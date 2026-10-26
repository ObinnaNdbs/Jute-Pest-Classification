"""Audit image properties across the Jute Pest dataset.

Prints a compact summary (sizes, aspect ratios, color modes, formats, corrupt
files) and writes per-image metadata to image_audit.csv for later use.

Usage:
    python check_images.py [dataset_root]
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).parent
DEFAULT_ROOT = PROJECT_DIR / "Jute_Pest_Dataset" / "Jute_Pest_Dataset"
OUT_CSV = PROJECT_DIR / "image_audit.csv"
SPLITS = ["train", "val", "test"]
EXPECTED_SIZE = (256, 256)


def audit(root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for path in sorted((root / split).rglob("*")):
            if not path.is_file():
                continue
            row = {"split": split, "class": path.parent.name,
                   "path": path.relative_to(root).as_posix(), "ext": path.suffix.lower(),
                   "width": None, "height": None, "mode": None, "format": None, "error": None}
            try:
                with Image.open(path) as img:
                    row.update(width=img.width, height=img.height, mode=img.mode, format=img.format)
                    img.verify()  # catches truncated / corrupt files
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["size"] = df["width"].astype("Int64").astype(str) + "x" + df["height"].astype("Int64").astype(str)
    df["shape"] = "non-square"
    df.loc[df["width"] == df["height"], "shape"] = "square (other)"
    df.loc[(df["width"] == EXPECTED_SIZE[0]) & (df["height"] == EXPECTED_SIZE[1]), "shape"] = "256x256"
    return df


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    df = audit(root)
    df.to_csv(OUT_CSV, index=False)
    ok = df[df["error"].isna()]

    print(f"Dataset root: {root}")
    print(f"Total files: {len(df)}   readable: {len(ok)}   corrupt: {len(df) - len(ok)}\n")

    print("Image shape category by split:")
    print(pd.crosstab(ok["split"], ok["shape"], margins=True).reindex(SPLITS + ["All"]), "\n")

    print("Top 10 sizes overall:")
    print(ok["size"].value_counts().head(10).to_string(), "\n")
    print(f"Distinct sizes: {ok['size'].nunique()}")
    print(f"Width range:  {ok['width'].min()} - {ok['width'].max()}")
    print(f"Height range: {ok['height'].min()} - {ok['height'].max()}\n")

    print("Shape category by class (all splits):")
    print(pd.crosstab(ok["class"], ok["shape"], margins=True), "\n")

    print("Color modes:", ok["mode"].value_counts().to_dict())
    print("Format by extension:", (ok["ext"] + " -> " + ok["format"]).value_counts().to_dict())

    if len(df) > len(ok):
        print("\nCorrupt files:")
        print(df[df["error"].notna()][["path", "error"]].to_string(index=False))

    print(f"\nPer-image details written to {OUT_CSV.name}")


if __name__ == "__main__":
    main()
