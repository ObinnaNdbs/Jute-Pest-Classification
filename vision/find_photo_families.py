"""Group images that are copies of the same photo, even after zoom, rotation or color changes.

Neural networks used for preprocessing only: they decide which images count as the same
photo for splitting, and are never inputs to the classifier. Run in .venv-vision.

  embed, calibrate   DINOv2 similarity alone. Kept as evidence: at a threshold safe enough
                     to avoid merging different photos, it misses most rotated or zoomed copies.
  embed8             DINOv2 embeddings of all 8 flip/90-degree versions of every image
  calibrate-match    LightGlue keypoint matches + RANSAC on known copies vs. hard look-alikes
  link               each image's nearest neighbors (at their best orientation) checked with
                     LightGlue; inlier counts saved to photo_links.csv
  build              families = pairs with enough inliers, merged with the existing exact and
                     near-duplicate groups (connected components)

Usage (from the project root):
    .venv-vision\\Scripts\\python vision\\find_photo_families.py embed8
    .venv-vision\\Scripts\\python vision\\find_photo_families.py calibrate-match
    .venv-vision\\Scripts\\python vision\\find_photo_families.py link
    .venv-vision\\Scripts\\python vision\\find_photo_families.py build --strict 30 --loose 15
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import ransac
from skimage.transform import AffineTransform
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

PROJECT_DIR = Path(__file__).resolve().parent.parent
META = PROJECT_DIR / "features" / "meta.csv"
EMBED_FILE = PROJECT_DIR / "photo_embeddings.npy"
OUT = PROJECT_DIR / "photo_families.csv"
CALIBRATION = PROJECT_DIR / "results" / "photo_family_calibration.csv"
EMBED8_FILE = PROJECT_DIR / "photo_embeddings_8.npy"
LINKS = PROJECT_DIR / "photo_links.csv"
MODEL = "facebook/dinov2-base"
MATCHER = "ETH-CVG/lightglue_superpoint"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 64
MATCH_BATCH = 1  # the transformers LightGlue model crashes on batches of more than one pair
SEED = 7


class Embedder:
    def __init__(self):
        self.proc = AutoImageProcessor.from_pretrained(MODEL)
        self.model = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()

    @torch.inference_mode()
    def _batch(self, images):
        inputs = self.proc(images=images, return_tensors="pt").to(DEVICE)
        return torch.nn.functional.normalize(self.model(**inputs).last_hidden_state[:, 0], dim=1).cpu().numpy()

    def __call__(self, images):
        """Unit-length embeddings, averaged with the mirror image so flips look the same."""
        out = []
        for i in tqdm(range(0, len(images), BATCH), leave=False):
            chunk = images[i:i + BATCH]
            mean = self._batch(chunk) + self._batch([ImageOps.mirror(im) for im in chunk])
            out.append(mean / np.linalg.norm(mean, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)


def load_images(paths):
    images = []
    for p in paths:
        with Image.open(PROJECT_DIR / p) as img:
            images.append(img.convert("RGB"))
    return images


def augment(img, kind, rng):
    w, h = img.size
    if kind == "flip":
        return ImageOps.mirror(img)
    if kind == "rotate90":
        return img.rotate(rng.choice([90, 180, 270]))
    if kind == "rotate_free":
        return img.rotate(rng.uniform(10, 40), resample=Image.BICUBIC, fillcolor=None)
    if kind == "zoom":
        f = rng.uniform(0.6, 0.85)
        cw, ch = int(w * f), int(h * f)
        l, t = rng.randint(0, w - cw), rng.randint(0, h - ch)
        return img.crop((l, t, l + cw, t + ch)).resize((w, h), Image.BICUBIC)
    if kind == "color":
        out = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.4))
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.7, 1.3))
        return ImageEnhance.Contrast(out).enhance(rng.uniform(0.7, 1.3))
    if kind == "combined":
        return augment(augment(augment(img, "rotate_free", rng), "zoom", rng), "color", rng)
    raise ValueError(kind)


def embed():
    meta = pd.read_csv(META, keep_default_na=False)
    np.save(EMBED_FILE, Embedder()(load_images(meta["processed"])))
    print(f"Saved {len(meta)} embeddings to {EMBED_FILE.name}")


def calibrate():
    meta = pd.read_csv(META, keep_default_na=False)
    E = np.load(EMBED_FILE)
    rng, np_rng = random.Random(SEED), np.random.default_rng(SEED)
    # Resized images are the cleaner pool: they are not generated copies of each other.
    resized = meta.index[meta["source"] == "resized"].to_numpy()
    picks = np_rng.choice(resized, 600, replace=False)
    embedder = Embedder()
    images = load_images(meta["processed"].iloc[picks])
    rows, thresholds = [], [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
    positives = {}
    for kind in ("flip", "rotate90", "rotate_free", "zoom", "color", "combined"):
        aug = embedder([augment(im, kind, rng) for im in images])
        positives[kind] = (aug * E[picks]).sum(axis=1)

    classes = meta["class"].to_numpy()
    sims = E[picks] @ E[resized].T
    same = classes[picks][:, None] == classes[resized][None, :]
    self_hit = picks[:, None] == resized[None, :]
    dup = meta["dup_group"].to_numpy()
    known = dup[picks][:, None] == dup[resized][None, :]
    neg_same = sims[same & ~self_hit & ~known]
    neg_other = sims[~same]
    for t in thresholds:
        row = {"threshold": t, "negative_same_class_rate": float((neg_same >= t).mean()),
               "negative_other_class_rate": float((neg_other >= t).mean())}
        row.update({f"copy_found_{k}": float((v >= t).mean()) for k, v in positives.items()})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(CALIBRATION, index=False)
    pd.set_option("display.width", 220)
    print(table.round(4).to_string(index=False))


def dihedral(img, v):
    """One of the 8 flip/90-degree-rotation versions of an image."""
    out = ImageOps.mirror(img) if v >= 4 else img
    return out.rotate(90 * (v % 4), expand=True)


class Matcher:
    """LightGlue keypoint matches, checked with a RANSAC affine fit. Same photo = many agreeing matches."""

    def __init__(self):
        self.proc = AutoImageProcessor.from_pretrained(MATCHER)
        self.model = AutoModel.from_pretrained(MATCHER).to(DEVICE).eval()

    @torch.inference_mode()
    def inliers(self, pairs):
        inputs = self.proc(pairs, return_tensors="pt").to(DEVICE)
        out = self.model(**inputs)
        sizes = [[(a.height, a.width), (b.height, b.width)] for a, b in pairs]
        return [self._ransac(m["keypoints0"].cpu().numpy(), m["keypoints1"].cpu().numpy())
                for m in self.proc.post_process_keypoint_matching(out, sizes, threshold=0.2)]

    @staticmethod
    def _ransac(k0, k1):
        if len(k0) < 4:
            return 0
        _, mask = ransac((k0, k1), AffineTransform, min_samples=3, residual_threshold=4, max_trials=200, rng=SEED)
        return 0 if mask is None else int(mask.sum())

    def run(self, pairs):
        out = []
        for i in tqdm(range(0, len(pairs), MATCH_BATCH), leave=False):
            out += self.inliers(pairs[i:i + MATCH_BATCH])
        return np.array(out)


@torch.inference_mode()
def embed_raw(embedder, images):
    out = []
    for i in range(0, len(images), BATCH):
        out.append(embedder._batch(images[i:i + BATCH]))
    return np.concatenate(out).astype(np.float32)


def embed8():
    """Embeddings of all 8 flip/rotation versions of every image: lets a pair be compared at its best orientation."""
    meta = pd.read_csv(META, keep_default_na=False)
    images = load_images(meta["processed"])
    embedder = Embedder()
    np.save(EMBED8_FILE, np.stack([embed_raw(embedder, [dihedral(im, v) for im in images])
                                   for v in tqdm(range(8))], axis=1))
    print(f"Saved {EMBED8_FILE.name}")


def best_orientation(a_vec, b_vecs):
    return int(np.argmax(b_vecs @ a_vec))


def calibrate_match():
    """Inlier counts for known copies vs. hard look-alikes, to choose the inlier threshold."""
    meta = pd.read_csv(META, keep_default_na=False)
    E8 = np.load(EMBED8_FILE)
    rng, np_rng = random.Random(SEED), np.random.default_rng(SEED)
    resized = meta.index[meta["source"] == "resized"].to_numpy()
    picks = np_rng.choice(resized, 300, replace=False)
    images = load_images(meta["processed"].iloc[picks])
    embedder, matcher = Embedder(), Matcher()
    results = {}
    for kind in ("flip", "rotate90", "rotate_free", "zoom", "color", "combined"):
        aug = [augment(im, kind, rng) for im in images]
        vecs = embed_raw(embedder, aug)
        pairs = [[a, dihedral(im, best_orientation(v, E8[p]))] for a, im, v, p in zip(aug, images, vecs, picks)]
        results[kind] = matcher.run(pairs)

    classes, dup = meta["class"].to_numpy(), meta["dup_group"].to_numpy()
    all_images = load_images(meta["processed"])
    neg_pairs, sims = [], []
    for p in picks:
        s = (E8[resized, 0] @ E8[p, 0])
        order = resized[np.argsort(-s)]
        j = next(k for k in order if k != p and classes[k] == classes[p] and dup[k] != dup[p])
        v = best_orientation(E8[j, 0], E8[p])
        neg_pairs.append([all_images[j], dihedral(all_images[p], v)])
        sims.append(float(E8[j, 0] @ E8[p, v]))
    negatives = matcher.run(neg_pairs)

    rows = []
    for t in (5, 8, 10, 12, 15, 20, 30, 40, 60, 100):
        row = {"min_inliers": t, "closest_same_class_lookalike_linked": float((negatives >= t).mean())}
        row.update({f"copy_found_{k}": float((v >= t).mean()) for k, v in results.items()})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(CALIBRATION.with_name("photo_family_match_calibration.csv"), index=False)
    pd.set_option("display.width", 220)
    print(f"median embedding similarity of the look-alike pairs: {np.median(sims):.3f}")
    print(table.round(3).to_string(index=False))


@torch.inference_mode()
def candidate_pairs(E8, k, min_sim):
    """Each image's k most similar other images (best orientation) with similarity >= min_sim."""
    E = torch.from_numpy(E8).to(DEVICE)
    n = len(E)
    found = {}
    for start in tqdm(range(0, n, 256), desc="candidates"):
        anchor = E[start:start + 256, 0]
        best, orient = torch.stack([anchor @ E[:, v].T for v in range(8)]).max(dim=0)
        best[torch.arange(len(anchor)), torch.arange(start, start + len(anchor))] = -1
        sims, idx = best.topk(k, dim=1)
        for a, (row_s, row_i) in enumerate(zip(sims.cpu().numpy(), idx.cpu().numpy())):
            i = start + a
            for s, j in zip(row_s, row_i):
                j = int(j)
                if s >= min_sim and (j, i) not in found:
                    found[(i, j)] = (float(s), int(orient[a, j]))
    return found


def link(k, min_sim):
    """Check every candidate pair with the keypoint matcher and save the inlier counts."""
    meta = pd.read_csv(META, keep_default_na=False)
    candidates = candidate_pairs(np.load(EMBED8_FILE), k, min_sim)
    print(f"{len(candidates)} candidate pairs", flush=True)
    images = load_images(meta["processed"])
    keys = list(candidates)
    matcher = Matcher()
    inliers = []
    for start in tqdm(range(0, len(keys), MATCH_BATCH), desc="matching"):
        chunk = keys[start:start + MATCH_BATCH]
        inliers += matcher.inliers([[images[i], dihedral(images[j], candidates[(i, j)][1])] for i, j in chunk])
    pd.DataFrame({"i": [i for i, _ in keys], "j": [j for _, j in keys],
                  "similarity": [candidates[key][0] for key in keys],
                  "orientation": [candidates[key][1] for key in keys], "inliers": inliers}).to_csv(LINKS, index=False)
    print(f"Saved {LINKS.name}")


def components(pairs, n, dup):
    first = pd.Series(np.arange(n)).groupby(dup).transform("first").to_numpy()
    rows = np.concatenate([pairs["i"].to_numpy(), np.arange(n)])
    cols = np.concatenate([pairs["j"].to_numpy(), first])
    graph = coo_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def build(strict, loose):
    """Families from pairs with at least this many inliers, merged with the existing duplicate groups."""
    meta = pd.read_csv(META, keep_default_na=False)
    links_table = pd.read_csv(LINKS)
    out = meta[["original", "split", "class", "source"]].copy()
    for name, t in (("strict", strict), ("loose", loose)):
        pairs = links_table[links_table["inliers"] >= t]
        links = len(pairs)
        fam = components(pairs, len(meta), meta["dup_group"].to_numpy())
        out[f"family_{name}"] = fam
        size = out.groupby(f"family_{name}")["original"].transform("size")
        n_classes = out.groupby(f"family_{name}")["class"].transform("nunique")
        out[f"family_{name}_size"] = size
        out[f"family_{name}_classes"] = n_classes
        print(f"\n{name} (inliers >= {t}): {links} links, {out[f'family_{name}'].nunique()} families; "
              f"largest {int(size.max())}; images in class-crossing families {int((n_classes > 1).sum())}")
        print(out.groupby("source")[f"family_{name}"].nunique().to_string())
    out.to_csv(OUT, index=False)
    print(f"\nSaved {OUT.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["embed", "calibrate", "embed8", "calibrate-match", "link", "build"])
    parser.add_argument("--strict", type=int, default=30, help="min inliers, strict families")
    parser.add_argument("--loose", type=int, default=15, help="min inliers, loose families")
    parser.add_argument("--k", type=int, default=15, help="candidate neighbors per image")
    parser.add_argument("--min-sim", type=float, default=0.80)
    args = parser.parse_args()
    {"embed": embed, "calibrate": calibrate, "embed8": embed8, "calibrate-match": calibrate_match,
     "link": lambda: link(args.k, args.min_sim), "build": lambda: build(args.strict, args.loose)}[args.stage]()
