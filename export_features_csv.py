"""Export features/features.npz to CSV files you can open in Excel or VS Code.

Writes to features/csv/:
  hsv.csv          - all images x 1024 HSV histogram bins
  lbp.csv          - all images x 54 LBP histogram bins
  hog_preview.csv  - first 50 images x 2352 HOG values (the full table is several hundred MB)

Each row starts with split, class and the image file, so you can see which image it is.

Usage:
    python export_features_csv.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).parent
FEATURE_DIR = PROJECT_DIR / "features"
OUT_DIR = FEATURE_DIR / "csv"
HOG_PREVIEW_ROWS = 50


def main():
    d = np.load(FEATURE_DIR / "features.npz")
    meta = pd.read_csv(FEATURE_DIR / "meta.csv", keep_default_na=False)[["split", "class", "original"]]
    OUT_DIR.mkdir(exist_ok=True)

    hsv_cols = [f"h{h}_s{s}_v{v}" for h in range(16) for s in range(8) for v in range(8)]
    lbp_cols = [f"lbp_P{p}_R{r}_code{c}" for p, r in [(8, 1), (16, 2), (24, 3)] for c in range(p + 2)]

    pd.concat([meta, pd.DataFrame(d["hsv"], columns=hsv_cols)], axis=1).to_csv(
        OUT_DIR / "hsv.csv", index=False, float_format="%.5f")
    pd.concat([meta, pd.DataFrame(d["lbp"], columns=lbp_cols)], axis=1).to_csv(
        OUT_DIR / "lbp.csv", index=False, float_format="%.5f")
    hog = pd.DataFrame(d["hog"][:HOG_PREVIEW_ROWS], columns=[f"hog_{i}" for i in range(d["hog"].shape[1])])
    pd.concat([meta.iloc[:HOG_PREVIEW_ROWS], hog], axis=1).to_csv(
        OUT_DIR / "hog_preview.csv", index=False, float_format="%.4f")

    print(f"Wrote hsv.csv, lbp.csv, hog_preview.csv to {OUT_DIR.relative_to(PROJECT_DIR)}/")


if __name__ == "__main__":
    main()
