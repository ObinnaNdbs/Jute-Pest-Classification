"""Extract handcrafted HOG, HSV-histogram, and LBP features from Jute_Pest_256.

Feature families (each saved separately so they can be ablated or combined):
  HOG  (shape)   - grayscale, 9 orientations, 16x16-pixel cells, 2x2-cell
                   blocks, L2-Hys normalization -> 15*15*2*2*9 = 8100 dims.
                   Computed on the full 256x256 image (HOG needs a fixed grid).
  HSV  (color)   - joint 3D histogram, 16 hue x 4 saturation x 4 value bins
                   = 256 dims, L1-normalized. Hue gets the most bins because
                   pest classes differ mainly in hue; few value bins reduce
                   sensitivity to brightness.
  LBP  (texture) - rotation-invariant uniform LBP at three scales:
                   (P=8, R=1), (P=16, R=2), (P=24, R=3) -> 10+18+26 = 54 dims,
                   each scale's histogram L1-normalized. Small radius captures
                   fine hair; larger radii capture coarser shell/wing texture.

HSV and LBP histograms are counted only inside the real-content box recorded
in preprocess_manifest.csv, so mirror/blur padding does not skew the color or
texture proportions. (For the 5,823 native 256x256 images the box is the full
image.) LBP codes are computed on the full image and then cropped, so the box
edges have proper neighbors.

Outputs (in features/):
  features.npz   - arrays hog, hsv, lbp (float32), label (int), row-aligned
  meta.csv       - one row per image: split, class, label, paths, image source,
                   and the duplicate-audit columns (dup_group, in_clean_split, ...)
  params.json    - every extraction setting, for reproducibility

Requires preprocess_images.py and audit_duplicates.py to have run first.

Usage:
    python extract_features.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from skimage.util import img_as_ubyte
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent
MANIFEST = PROJECT_DIR / "preprocess_manifest.csv"
DUP_AUDIT = PROJECT_DIR / "duplicate_audit.csv"
OUT_DIR = PROJECT_DIR / "features"
AUDIT_COLUMNS = ["dup_group", "group_size", "has_exact_dup", "cross_split_dup", "label_conflict",
                 "in_clean_split", "clean_exclusion_reason"]
META_COLUMNS = ["split", "class", "label", "source", "processed", "original", "action", *AUDIT_COLUMNS]

PARAMS = {
    "image_size": 256,
    "hog": {"orientations": 9, "pixels_per_cell": [16, 16], "cells_per_block": [2, 2],
            "block_norm": "L2-Hys"},
    "hsv": {"bins": [16, 4, 4]},
    "lbp": {"scales": [[8, 1], [16, 2], [24, 3]], "method": "uniform"},
    "histograms_use_content_box": True,
}


def hog_features(gray: np.ndarray) -> np.ndarray:
    p = PARAMS["hog"]
    return hog(gray, orientations=p["orientations"], pixels_per_cell=tuple(p["pixels_per_cell"]),
               cells_per_block=tuple(p["cells_per_block"]), block_norm=p["block_norm"],
               feature_vector=True)


def hsv_features(rgb_box: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(rgb_box).reshape(-1, 3)  # H, S, V all in [0, 1]
    hist, _ = np.histogramdd(hsv, bins=PARAMS["hsv"]["bins"], range=[(0, 1)] * 3)
    return hist.ravel() / hist.sum()


def lbp_features(gray_u8: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    hists = []
    for n_points, radius in PARAMS["lbp"]["scales"]:
        codes = local_binary_pattern(gray_u8, n_points, radius, method=PARAMS["lbp"]["method"])
        codes = codes[top:bottom, left:right]
        # 'uniform' yields P+1 uniform codes plus one bin for all non-uniform patterns.
        hist = np.bincount(codes.astype(np.int64).ravel(), minlength=n_points + 2)
        hists.append(hist / hist.sum())
    return np.concatenate(hists)


def extract_one(path: Path, box: tuple[int, int, int, int]):
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"))
    gray = rgb2gray(rgb)
    left, top, right, bottom = box
    return (hog_features(gray).astype(np.float32),
            hsv_features(rgb[top:bottom, left:right]).astype(np.float32),
            lbp_features(img_as_ubyte(gray), box).astype(np.float32))


def main():
    meta = pd.read_csv(MANIFEST)
    audit = pd.read_csv(DUP_AUDIT, keep_default_na=False)[["original", *AUDIT_COLUMNS]]
    meta = meta.merge(audit, on="original", how="left", validate="one_to_one")
    assert meta["dup_group"].notna().all(), "duplicate_audit.csv is stale; rerun audit_duplicates.py"
    classes = sorted(meta["class"].unique())
    meta["label"] = meta["class"].map({c: i for i, c in enumerate(classes)})
    # Native 256x256 images and resized ones appear to come from different
    # sources and score very differently, so results are reported per source.
    meta["source"] = np.where(meta["action"] == "copied", "native_256", "resized")

    if not PARAMS["histograms_use_content_box"]:
        meta[["box_left", "box_top"]] = 0
        meta[["box_right", "box_bottom"]] = PARAMS["image_size"]
    boxes = meta[["box_left", "box_top", "box_right", "box_bottom"]].itertuples(index=False, name=None)

    jobs = (delayed(extract_one)(PROJECT_DIR / p, tuple(int(v) for v in b))
            for p, b in zip(meta["processed"], boxes))
    results = Parallel(n_jobs=-1, batch_size=32, return_as="generator")(jobs)
    hog_x, hsv_x, lbp_x = zip(*tqdm(results, total=len(meta), desc="Extracting"))

    feats = {"hog": np.stack(hog_x), "hsv": np.stack(hsv_x), "lbp": np.stack(lbp_x)}
    for name, x in feats.items():
        assert np.isfinite(x).all(), f"non-finite values in {name}"

    OUT_DIR.mkdir(exist_ok=True)
    np.savez(OUT_DIR / "features.npz", label=meta["label"].to_numpy(), **feats)
    meta[META_COLUMNS].to_csv(OUT_DIR / "meta.csv", index=False)
    (OUT_DIR / "params.json").write_text(json.dumps({**PARAMS, "classes": classes}, indent=2))

    print(f"\nExtracted features for {len(meta)} images ({len(classes)} classes)")
    for name, x in feats.items():
        n_const = int((x.std(axis=0) == 0).sum())
        print(f"  {name.upper():4s} {x.shape[1]:5d} dims   constant columns: {n_const}")
    print(f"  ALL  {sum(x.shape[1] for x in feats.values()):5d} dims combined")
    print(f"Saved to {OUT_DIR.name}/ (features.npz, meta.csv, params.json)")


if __name__ == "__main__":
    main()
