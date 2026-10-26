"""Find exact and near-duplicate images and define the cleaned (leakage-free) split.

Detection:
  1. Exact duplicates: identical file bytes (MD5 of the original files).
  2. Near-duplicates: re-encoded, resized, cropped, rotated, or flipped copies
     (the dataset contains augmented "Aug_..." files). Two stages:
       a. Candidates: perceptual hash (64-bit DCT pHash) of each 256x256 image,
          also compared against horizontally/vertically flipped hashes;
          pairs within PHASH_MAX_DIST bits become candidates.
       b. Verification: ORB keypoints matched between the pair, then RANSAC
          fits one affine transform (shift/scale/rotation/crop). The number of
          matches that agree with it (inliers) is the evidence:
            >= STRONG_INLIERS   -> "near" (confirmed near-duplicate)
            >= POSSIBLE_INLIERS -> "possible" (recorded, not acted on)
     pHash alone is unreliable here: small insects on white backgrounds hash
     alike regardless of species. With 6-14 inliers, visual inspection found
     both real augmented copies and different crickets photographed alike,
     and no pixel-level check separated them, so only strong matches are used.
     Every strong match inspected was a real copy. The cleaned split is
     therefore conservative: it removes leakage we are sure of, and the
     "possible" pairs in duplicate_pairs.csv mark where more may remain.
  3. Exact and confirmed near-duplicate links are merged into groups
     (connected components), so A~B and B~C puts A, B, C in one group.

Cleaned split (in_clean_split column):
  - Label conflicts (one group containing several classes): all members dropped.
  - Groups spanning several splits: only copies in the most evaluation-critical
    split are kept (test > val > train). This keeps the test set nearly
    identical to the original, so original-vs-clean differences isolate the
    effect of training on leaked copies.
  - Within a split, one copy per group is kept (prefer non-"Aug_" files).

Outputs:
  duplicate_audit.csv     - one row per image: dup_group, flags, clean-split decision
  duplicate_pairs.csv     - every accepted duplicate link with its evidence
  figures/duplicate_examples.png - sample near-duplicate pairs (incl. label conflicts)

Usage:
    python audit_duplicates.py
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image, ImageDraw
from scipy.fft import dctn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent
SRC_ROOT = PROJECT_DIR / "Jute_Pest_Dataset" / "Jute_Pest_Dataset"
MANIFEST = PROJECT_DIR / "preprocess_manifest.csv"
OUT_AUDIT = PROJECT_DIR / "duplicate_audit.csv"
OUT_PAIRS = PROJECT_DIR / "duplicate_pairs.csv"
OUT_EXAMPLES = PROJECT_DIR / "figures" / "duplicate_examples.png"

PHASH_MAX_DIST = 14       # candidate threshold (bits out of 64)
STRONG_INLIERS = 15       # RANSAC inliers to confirm a near-duplicate
POSSIBLE_INLIERS = 6      # weaker evidence: listed in duplicate_pairs.csv only
# Two ORB settings: sparse finds strong corners, dense recovers low-texture copies.
ORB_SETTINGS = [dict(n_keypoints=400, fast_threshold=0.05),
                dict(n_keypoints=1000, fast_threshold=0.02)]
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}  # lower = kept first
FLIPS = ["none", "horizontal", "vertical"]


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.float64) / 255


def flip(gray: np.ndarray, how: int) -> np.ndarray:
    return [gray, gray[:, ::-1], gray[::-1, :]][how]


def phash(gray: np.ndarray) -> np.uint64:
    small = Image.fromarray((gray * 255).astype(np.uint8)).resize((32, 32), Image.LANCZOS)
    dct = dctn(np.asarray(small, dtype=np.float64), norm="ortho")[:8, :8].ravel()
    bits = dct > np.median(dct[1:])  # skip the DC term when picking the median
    return np.packbits(bits).view(">u8")[0]


def file_fingerprints(original: str, processed: str):
    md5 = hashlib.md5((SRC_ROOT / original).read_bytes()).hexdigest()
    gray = load_gray(PROJECT_DIR / processed)
    return md5, [phash(flip(gray, f)) for f in range(len(FLIPS))]


def ransac_inliers(path_a: Path, path_b: Path, flip_b: int) -> int:
    """Max RANSAC-consistent ORB matches between a and (flipped) b over ORB_SETTINGS."""
    gray_a, gray_b = load_gray(path_a), flip(load_gray(path_b), flip_b)
    best = 0
    for setting in ORB_SETTINGS:
        try:
            orb_a, orb_b = ORB(**setting), ORB(**setting)
            orb_a.detect_and_extract(gray_a)
            orb_b.detect_and_extract(gray_b)
            matches = match_descriptors(orb_a.descriptors, orb_b.descriptors,
                                        cross_check=True, max_ratio=0.8)
            if len(matches) < 4:
                continue
            src = orb_a.keypoints[matches[:, 0]][:, ::-1]  # (row, col) -> (x, y)
            dst = orb_b.keypoints[matches[:, 1]][:, ::-1]
            _, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                                residual_threshold=3, max_trials=500, rng=0)
            if inliers is not None:
                best = max(best, int(inliers.sum()))
        except RuntimeError:  # too few keypoints in a flat image
            continue
    return best


def phash_candidates(hashes: np.ndarray, md5: np.ndarray, chunk: int = 1000) -> pd.DataFrame:
    """All pairs (a < b) whose pHash, allowing b to be flipped, is within PHASH_MAX_DIST."""
    n = len(hashes)
    rows = []
    for start in range(0, n, chunk):
        a_hash = hashes[start:start + chunk, 0, None]
        best = np.full((len(a_hash), n), 65)
        best_flip = np.zeros_like(best)
        for f in range(len(FLIPS)):
            dist = np.bitwise_count(a_hash ^ hashes[None, :, f]).astype(int)
            better = dist < best
            best[better], best_flip[better] = dist[better], f
        ii, jj = np.nonzero(best <= PHASH_MAX_DIST)
        for i, j in zip(ii + start, jj):
            if i < j and md5[i] != md5[j]:  # identical files are already exact duplicates
                rows.append((i, j, best[i - start, j], best_flip[i - start, j]))
    return pd.DataFrame(rows, columns=["a", "b", "phash_dist", "flip"])


def choose_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Decide which images stay in the cleaned split (see module docstring)."""
    df["in_clean_split"] = True
    df["clean_exclusion_reason"] = ""
    for _, g in df[df["group_size"] > 1].groupby("dup_group"):
        if g["class"].nunique() > 1:
            df.loc[g.index, ["in_clean_split", "clean_exclusion_reason"]] = [False, "label_conflict"]
            continue
        keep_split = min(g["split"], key=SPLIT_PRIORITY.get)
        other = g.index[g["split"] != keep_split]
        df.loc[other, ["in_clean_split", "clean_exclusion_reason"]] = [False, "copy_in_" + keep_split]
        same = g[g["split"] == keep_split].copy()
        same["is_aug"] = same["original"].map(lambda p: Path(p).name.startswith("Aug_"))
        keeper = same.sort_values(["is_aug", "original"]).index[0]
        dropped = same.index.drop(keeper)
        df.loc[dropped, ["in_clean_split", "clean_exclusion_reason"]] = [False, "within_split_copy"]
    return df


def save_examples(df: pd.DataFrame, pairs: pd.DataFrame, n: int = 8):
    """Grid of near-duplicate pairs for the report: all label conflicts plus a sample."""
    near = pairs[pairs["type"] == "near"]
    conflicts = near[~near["same_class"]].drop_duplicates("b")
    sample = near[near["same_class"]].sample(max(0, n - len(conflicts)), random_state=0)
    show = pd.concat([conflicts, sample])
    cell, gap, label_h = 160, 10, 16
    sheet = Image.new("RGB", (2 * cell + 3 * gap, len(show) * (cell + label_h + gap) + gap), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (_, r) in enumerate(show.iterrows()):
        y = gap + i * (cell + label_h + gap)
        for j, idx in enumerate([r["a"], r["b"]]):
            with Image.open(PROJECT_DIR / df.at[idx, "processed"]) as img:
                sheet.paste(img.convert("RGB").resize((cell, cell)), (gap + j * (cell + gap), y + label_h))
            draw.text((gap + j * (cell + gap), y), f"{df.at[idx, 'split']}/{df.at[idx, 'class']}"[:30], fill="black")
    OUT_EXAMPLES.parent.mkdir(exist_ok=True)
    sheet.save(OUT_EXAMPLES)


def main():
    df = pd.read_csv(MANIFEST)[["split", "class", "original", "processed"]]
    n = len(df)

    fp = Parallel(n_jobs=-1)(delayed(file_fingerprints)(o, p) for o, p in
                             tqdm(zip(df["original"], df["processed"]), total=n, desc="Hashing"))
    df["md5"] = [m for m, _ in fp]
    hashes = np.array([h for _, h in fp], dtype=np.uint64)

    # Exact duplicates: link every member of an MD5 group to its first member.
    exact = []
    for _, idx in df.groupby("md5").groups.items():
        exact += [(idx[0], j, 0, 0, -1) for j in idx[1:]]
    exact = pd.DataFrame(exact, columns=["a", "b", "phash_dist", "flip", "inliers"]).assign(type="exact")

    cand = phash_candidates(hashes, df["md5"].to_numpy())
    cand["inliers"] = Parallel(n_jobs=-1, batch_size=16)(
        delayed(ransac_inliers)(PROJECT_DIR / df.at[a, "processed"], PROJECT_DIR / df.at[b, "processed"], f)
        for a, b, f in tqdm(zip(cand["a"], cand["b"], cand["flip"]), total=len(cand), desc="Verifying"))
    near = cand[cand["inliers"] >= STRONG_INLIERS].assign(type="near")
    possible = cand[cand["inliers"].between(POSSIBLE_INLIERS, STRONG_INLIERS - 1)].assign(type="possible")

    pairs = pd.concat([exact, near, possible], ignore_index=True)
    linked = pairs[pairs["type"] != "possible"]
    graph = coo_matrix((np.ones(len(linked)), (linked["a"], linked["b"])), shape=(n, n))
    _, df["dup_group"] = connected_components(graph, directed=False)
    grp = df.groupby("dup_group")
    df["group_size"] = grp["split"].transform("size")
    df["cross_split_dup"] = grp["split"].transform("nunique") > 1
    df["label_conflict"] = grp["class"].transform("nunique") > 1
    df["has_exact_dup"] = df.duplicated("md5", keep=False)
    df = choose_clean(df)

    pairs["same_class"] = df["class"].to_numpy()[pairs["a"]] == df["class"].to_numpy()[pairs["b"]]
    pairs["cross_split"] = df["split"].to_numpy()[pairs["a"]] != df["split"].to_numpy()[pairs["b"]]
    pairs["flip"] = pairs["flip"].map(lambda f: FLIPS[f] if f >= 0 else "")
    out_pairs = pairs.assign(original_a=df["original"].to_numpy()[pairs["a"]],
                             original_b=df["original"].to_numpy()[pairs["b"]])
    out_pairs.drop(columns=["a", "b"]).to_csv(OUT_PAIRS, index=False)
    df.drop(columns=["processed"]).to_csv(OUT_AUDIT, index=False)
    save_examples(df, pairs)

    # ---- Summary ----
    multi = df[df["group_size"] > 1]
    exact_groups = df[df["has_exact_dup"]].groupby("md5")["split"].nunique()
    print(f"\nPairs checked by ORB+RANSAC: {len(cand)}   confirmed near-duplicate links: {len(near)}   "
          f"possible (not acted on): {len(possible)}")
    print(f"Exact-duplicate groups: {len(exact_groups)} ({(exact_groups > 1).sum()} cross splits)")
    print(f"Test images with an exact copy in train/val: "
          f"{(df['has_exact_dup'] & df['cross_split_dup'] & (df['split'] == 'test')).sum()}")
    print(f"All duplicate groups (exact + near): {multi['dup_group'].nunique()} groups, {len(multi)} images, "
          f"largest group {df['group_size'].max()}")
    print(f"  crossing splits: {multi[multi['cross_split_dup']]['dup_group'].nunique()} groups")
    print(f"  label conflicts: {multi[multi['label_conflict']]['dup_group'].nunique()} groups "
          f"({multi['label_conflict'].sum()} images)")
    for split in SPLIT_PRIORITY:
        s = df[df["split"] == split]
        print(f"  {split:5s} images with a copy in another split: {s['cross_split_dup'].sum()}")
    aug = df["original"].map(lambda p: Path(p).name.startswith("Aug_"))
    print(f"Files named 'Aug_*' (augmented copies): {aug.sum()}")

    print("\nCleaned split:")
    summary = df.groupby("split").agg(original=("split", "size"), clean=("in_clean_split", "sum"))
    summary["removed"] = summary["original"] - summary["clean"]
    print(summary.reindex(list(SPLIT_PRIORITY)[::-1]).to_string())
    print("\nRemoval reasons:", df["clean_exclusion_reason"].value_counts().drop("", errors="ignore").to_dict())
    print(f"\nWrote {OUT_AUDIT.name}, {OUT_PAIRS.name}, figures/{OUT_EXAMPLES.name}")


if __name__ == "__main__":
    main()
