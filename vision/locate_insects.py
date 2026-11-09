"""Find the insect in every image with neural networks, then crop to it.

Neural networks are used here for preprocessing only. Their outputs decide which
pixels get measured; they are never inputs to the classifier. Run in .venv-vision.

  - Grounding DINO: text-prompted detector. One fixed prompt for every image, so no
    class label is used and train, val and test are treated identically.
  - SAM 2: turns each detected box into a precise insect mask.

Box rule: the highest-scoring box, merged with boxes scoring at least 80% of it (for
swarms of mites, aphids or termites). A box more than 4x the top box's area is not
merged, so one whole-frame detection can't undo a tight crop.

Crop rule: union box + 15% margin, expanded to a square inside the image, resized to
256x256 with Lanczos. Works from the original files, so crops have no padding.
Images with no detection keep the existing 256x256 image and are flagged found=False.

Outputs:
  insect_boxes.csv          one row per image, same order as preprocess_manifest.csv
  insect_masks.npz          packed 256x256 masks: crop_mask (in the crop), full_mask (in
                            the Jute_Pest_256 image, False over padding)
  Jute_Pest_Crop256/        the crops

Usage (from the project root):
    .venv-vision\\Scripts\\python vision\\locate_insects.py --sample   # contact sheets only
    .venv-vision\\Scripts\\python vision\\locate_insects.py            # all images
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                          Sam2Processor)

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_DIR / "Jute_Pest_Dataset" / "Jute_Pest_Dataset"
MANIFEST = PROJECT_DIR / "preprocess_manifest.csv"
CROP_ROOT = PROJECT_DIR / "Jute_Pest_Crop256"
SHEET_DIR = PROJECT_DIR / "figures" / "fair"

DETECTOR = "IDEA-Research/grounding-dino-base"
SEGMENTER = "facebook/sam2.1-hiera-large"
PROMPT = "insect. bug. beetle. caterpillar. moth. mite. aphid. termite. cricket."
BOX_THRESHOLD, TEXT_THRESHOLD = 0.25, 0.20
MERGE_SCORE, MERGE_MAX_AREA = 0.8, 4.0
MARGIN, TARGET = 0.15, 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Locator:
    def __init__(self):
        self.det_proc = AutoProcessor.from_pretrained(DETECTOR)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR).to(DEVICE).eval()
        self.seg_proc = Sam2Processor.from_pretrained(SEGMENTER)
        self.seg = Sam2Model.from_pretrained(SEGMENTER).to(DEVICE).eval()

    @torch.inference_mode()
    def boxes(self, img: Image.Image):
        """Detected boxes (xyxy, original pixels) and scores, best first."""
        inputs = self.det_proc(images=img, text=PROMPT, return_tensors="pt").to(DEVICE)
        out = self.det(**inputs)
        res = self.det_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[img.size[::-1]])[0]
        order = res["scores"].argsort(descending=True)
        return res["boxes"][order].cpu().numpy(), res["scores"][order].cpu().numpy()

    @torch.inference_mode()
    def mask(self, img: Image.Image, boxes: np.ndarray) -> np.ndarray:
        """Union of one SAM 2 mask per box, at original image size."""
        inputs = self.seg_proc(images=img, input_boxes=[[b.tolist() for b in boxes]], return_tensors="pt").to(DEVICE)
        out = self.seg(**inputs, multimask_output=False)
        masks = self.seg_proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        return masks.squeeze(1).numpy().astype(bool).any(axis=0)


def select_boxes(boxes, scores):
    area = lambda b: max(b[2] - b[0], 1) * max(b[3] - b[1], 1)
    top = area(boxes[0])
    keep = [i for i, (b, s) in enumerate(zip(boxes, scores))
            if i == 0 or (s >= MERGE_SCORE * scores[0] and area(b) <= MERGE_MAX_AREA * top)]
    return boxes[keep]


def square_crop(box, w, h):
    """Square (left, top, right, bottom) around box + margin, kept inside the image."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    side = int(round(min(max(bw, bh) * (1 + 2 * MARGIN), w, h)))
    side = max(side, 1)
    left = int(round((x0 + x1) / 2 - side / 2))
    top = int(round((y0 + y1) / 2 - side / 2))
    left, top = min(max(left, 0), w - side), min(max(top, 0), h - side)
    return left, top, left + side, top + side


def resize_mask(mask: np.ndarray, size) -> np.ndarray:
    return np.asarray(Image.fromarray(mask).resize(size, Image.NEAREST), dtype=bool)


def load_original(row) -> Image.Image:
    with Image.open(SRC_ROOT / row["original"]) as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")


def process(locator, row):
    img = load_original(row)
    w, h = img.size
    boxes, scores = locator.boxes(img)
    info = dict(original=row["original"], found=len(boxes) > 0, score=float(scores[0]) if len(boxes) else np.nan,
                n_boxes=len(boxes), orig_w=w, orig_h=h)

    if not len(boxes):
        with Image.open(PROJECT_DIR / row["processed"]) as existing:
            crop = existing.convert("RGB")
        keep = np.zeros((TARGET, TARGET), dtype=bool)
        info.update(crop_left=0, crop_top=0, crop_right=w, crop_bottom=h, upscale=np.nan, mask_share=np.nan, box_share=np.nan)
        return info, crop, ~keep, keep, img, None

    chosen = select_boxes(boxes, scores)
    union = (chosen[:, 0].min(), chosen[:, 1].min(), chosen[:, 2].max(), chosen[:, 3].max())
    mask = locator.mask(img, chosen)
    left, top, right, bottom = square_crop(union, w, h)
    crop = img.crop((left, top, right, bottom)).resize((TARGET, TARGET), Image.LANCZOS)
    crop_mask = resize_mask(mask[top:bottom, left:right], (TARGET, TARGET))

    # Same geometry as Jute_Pest_256: the original sits inside the recorded content box.
    box_w, box_h = int(row["box_right"] - row["box_left"]), int(row["box_bottom"] - row["box_top"])
    full_mask = np.zeros((TARGET, TARGET), dtype=bool)
    full_mask[int(row["box_top"]):int(row["box_bottom"]), int(row["box_left"]):int(row["box_right"])] = resize_mask(mask, (box_w, box_h))

    info.update(crop_left=left, crop_top=top, crop_right=right, crop_bottom=bottom, upscale=TARGET / (right - left),
                mask_share=float(mask.mean()), box_share=float((union[2] - union[0]) * (union[3] - union[1]) / (w * h)))
    return info, crop, crop_mask, full_mask, img, union


def tile(info, crop, crop_mask, img, union, cell=150):
    left = img.copy()
    if union is not None:
        ImageDraw.Draw(left).rectangle(list(union), outline=(255, 40, 40), width=max(2, img.width // 120))
    left.thumbnail((cell, cell), Image.LANCZOS)
    tinted = np.asarray(crop, dtype=np.float32)
    tinted[~crop_mask] = tinted[~crop_mask] * 0.35
    right = Image.fromarray(tinted.astype(np.uint8)).resize((cell, cell), Image.BILINEAR)
    pair = Image.new("RGB", (2 * cell + 4, cell), "white")
    pair.paste(left, ((cell - left.width) // 2, (cell - left.height) // 2))
    pair.paste(right, (cell + 4, 0))
    return pair


def sample_sheets(locator, manifest):
    """Contact sheets of train images only: original with box, beside crop with mask."""
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    train = manifest[manifest["split"] == "train"]
    for name, rows in (("native", train[train["action"] == "copied"]), ("resized", train[train["action"] != "copied"])):
        picks = rows.sample(60, random_state=7)
        tiles, misses = [], 0
        for _, row in tqdm(picks.iterrows(), total=len(picks), desc=name):
            info, crop, crop_mask, _, img, union = process(locator, row)
            misses += not info["found"]
            tiles.append(tile(info, crop, crop_mask, img, union))
        w, h = tiles[0].size
        sheet = Image.new("RGB", (6 * (w + 8) + 8, 10 * (h + 8) + 8), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, (8 + (i % 6) * (w + 8), 8 + (i // 6) * (h + 8)))
        sheet.save(SHEET_DIR / f"crop_check_{name}.png")
        print(f"{name}: no detection on {misses} of {len(picks)}; saved {SHEET_DIR / f'crop_check_{name}.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="only write contact sheets from train images")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST)
    locator = Locator()
    if args.sample:
        sample_sheets(locator, manifest)
        return

    rows, crop_masks, full_masks = [], [], []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Locating"):
        info, crop, crop_mask, full_mask, _, _ = process(locator, row)
        dst = CROP_ROOT / Path(row["processed"]).relative_to("Jute_Pest_256").with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        crop.save(dst, optimize=False)
        info["crop_path"] = dst.relative_to(PROJECT_DIR).as_posix()
        rows.append(info)
        crop_masks.append(np.packbits(crop_mask))
        full_masks.append(np.packbits(full_mask))

    table = pd.DataFrame(rows)
    assert table["crop_path"].is_unique, "two images mapped to the same crop file"
    table.to_csv(PROJECT_DIR / "insect_boxes.csv", index=False)
    np.savez_compressed(PROJECT_DIR / "insect_masks.npz", crop_mask=np.stack(crop_masks), full_mask=np.stack(full_masks))
    print(f"Found an insect in {int(table['found'].sum())} of {len(table)} images")


if __name__ == "__main__":
    main()
