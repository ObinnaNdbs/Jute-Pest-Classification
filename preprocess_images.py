"""Standardize every Jute Pest image to 256x256 RGB.

Strategy:
  - Native 256x256 JPEGs are copied byte-for-byte (no re-encoding, no change).
  - Non-square images are padded to square with symmetric (mirror) reflection,
    then resized. Center-cropping can cut off pests that fill the frame or sit
    off-center, and stretching distorts the shapes that HOG measures. Constant
    (black/gray) padding would add an artificial edge for HOG, a fake color
    spike for the HSV histogram, and flat regions for LBP; mirror padding
    extends the real background instead. Panoramas wider than 2:1 would get
    the pest mirrored several times, so they use blur-fill padding instead.
  - Square images of other sizes are resized directly.
  - EXIF orientation is applied first so photos are upright as taken.
  - Resampling uses Lanczos (anti-aliased downscaling, sharp upscaling).
  - Processed images are saved as PNG (lossless) to avoid a second round of
    JPEG compression artifacts.

Output mirrors the original split/class folders under Jute_Pest_256/ and a
manifest (preprocess_manifest.csv) maps each original file to its output.

Usage:
    python preprocess_images.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent
SRC_ROOT = PROJECT_DIR / "Jute_Pest_Dataset" / "Jute_Pest_Dataset"
DST_ROOT = PROJECT_DIR / "Jute_Pest_256"
MANIFEST = PROJECT_DIR / "preprocess_manifest.csv"
PREVIEW = PROJECT_DIR / "figures" / "preprocess_examples.png"
SPLITS = ["train", "val", "test"]
TARGET = 256
MAX_MIRROR_RATIO = 2.0  # wider/taller than 2:1 -> blur-fill padding instead of mirror


def pad_to_square(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], str]:
    """Pad the short side so the image becomes square.

    Mirror padding is used up to MAX_MIRROR_RATIO. Beyond that, mirroring
    reflects the whole image several times and duplicates the pest, so the
    padding is filled with a heavily blurred, stretched copy of the image
    instead (keeps background colors, adds no copies and no hard edge).

    Returns the padded image, the box (left, top, right, bottom) where the
    original content sits in padded-image coordinates, and the method used.
    """
    w, h = img.size
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    box = (left, top, left + w, top + h)

    if side / min(h, w) <= MAX_MIRROR_RATIO:
        pad = ((top, side - h - top), (left, side - w - left), (0, 0))
        return Image.fromarray(np.pad(np.asarray(img), pad, mode="symmetric")), box, "mirror"

    canvas = img.resize((side, side), Image.BILINEAR).filter(ImageFilter.GaussianBlur(side / 15))
    canvas.paste(img, (left, top))
    return canvas, box, "blur"


def process(src: Path, dst_dir: Path) -> dict:
    with Image.open(src) as raw:
        orig_w, orig_h, fmt = raw.width, raw.height, raw.format
        orientation = raw.getexif().get(0x0112, 1)

        if (orig_w, orig_h) == (TARGET, TARGET) and fmt == "JPEG" and orientation == 1:
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            return dict(action="copied", orig_w=orig_w, orig_h=orig_h, exif_rotated=False,
                        content_box=(0, 0, TARGET, TARGET), dst=dst)

        img = ImageOps.exif_transpose(raw).convert("RGB")

    if img.width != img.height:
        img, box, how = pad_to_square(img)
        action = f"{how}-padded+resized"
    else:
        box = (0, 0, img.width, img.height)
        action = "resized" if img.width != TARGET else "re-encoded"

    # Scale the content box into 256x256 coordinates for later masking.
    scale = TARGET / img.width
    box = tuple(int(round(v * scale)) for v in box)
    if img.width != TARGET:
        img = img.resize((TARGET, TARGET), Image.LANCZOS)

    # A few folders hold different photos named e.g. large.jpg and large.jpeg;
    # keep the original extension in the name so they don't overwrite each other.
    same_stem = [p for p in src.parent.iterdir() if p.stem == src.stem]
    name = src.stem if len(same_stem) == 1 else src.stem + "_" + src.suffix.lstrip(".").lower()
    dst = dst_dir / (name + ".png")
    img.save(dst, optimize=False)
    return dict(action=action, orig_w=orig_w, orig_h=orig_h, exif_rotated=orientation != 1,
                content_box=box, dst=dst)


def make_preview(manifest: pd.DataFrame, n: int = 6):
    """Side-by-side grid of original vs. processed for padded images."""
    padded = manifest[manifest["action"].str.contains("padded")]
    # Pick a spread of aspect ratios, including the most extreme.
    padded = padded.assign(ar=padded[["orig_w", "orig_h"]].max(axis=1) / padded[["orig_w", "orig_h"]].min(axis=1))
    padded = padded.sort_values("ar")
    picks = padded.iloc[np.linspace(0, len(padded) - 1, n).astype(int)]

    cell, gap = 256, 8
    sheet = Image.new("RGB", (2 * cell + 3 * gap, n * (cell + gap) + gap), "white")
    for i, (_, row) in enumerate(picks.iterrows()):
        with Image.open(SRC_ROOT / row["original"]) as o:
            orig = ImageOps.exif_transpose(o).convert("RGB")
        orig.thumbnail((cell, cell), Image.LANCZOS)
        y = gap + i * (cell + gap)
        sheet.paste(orig, (gap + (cell - orig.width) // 2, y + (cell - orig.height) // 2))
        with Image.open(PROJECT_DIR / row["processed"]) as p:
            sheet.paste(p.convert("RGB"), (2 * gap + cell, y))
    PREVIEW.parent.mkdir(exist_ok=True)
    sheet.save(PREVIEW)


def main():
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)  # start clean so reruns never leave stale files

    rows = []
    sources =[p for split in SPLITS for p in sorted((SRC_ROOT / split).rglob("*")) if p.is_file()]
    for src in tqdm(sources, desc="Preprocessing"):
        rel = src.relative_to(SRC_ROOT)
        split, cls = rel.parts[0], rel.parts[1]
        dst_dir = DST_ROOT / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        info = process(src, dst_dir)
        rows.append({
            "split": split,
            "class": cls,
            "original": rel.as_posix(),
            "processed": info["dst"].relative_to(PROJECT_DIR).as_posix(),
            "orig_w": info["orig_w"],
            "orig_h": info["orig_h"],
            "action": info["action"],
            "exif_rotated": info["exif_rotated"],
            # Region of the 256x256 output that is real (non-padded) content.
            "box_left": info["content_box"][0],
            "box_top": info["content_box"][1],
            "box_right": info["content_box"][2],
            "box_bottom": info["content_box"][3],
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False)

    # Sanity check: every output is 256x256 RGB and nothing collided.
    assert manifest["processed"].is_unique, "two inputs mapped to the same output file"
    for p in manifest["processed"]:
        with Image.open(PROJECT_DIR / p) as img:
            assert img.size == (TARGET, TARGET) and img.mode == "RGB", p

    make_preview(manifest)

    print(f"\nProcessed {len(manifest)} images -> {DST_ROOT.name}/")
    print(pd.crosstab(manifest["split"], manifest["action"], margins=True).reindex(SPLITS + ["All"]))
    print(f"EXIF-rotated: {manifest['exif_rotated'].sum()}")
    print(f"All outputs verified {TARGET}x{TARGET} RGB.")
    print(f"Manifest: {MANIFEST.name}   Preview: figures/{PREVIEW.name}")


if __name__ == "__main__":
    main()
