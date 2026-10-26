"""Report / presentation figures describing the data and the features.

Writes to figures/:
  class_distribution.png    - images per class, stacked by split
  source_by_class.png       - share of native 256x256 vs resized images per class
  sample_grid.png           - one example image per class
  feature_visualization.png - what HOG, LBP, and the hue histogram see in a few pests

(preprocess_examples.png and duplicate_examples.png are written by
preprocess_images.py and audit_duplicates.py.)

Usage:
    python make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import hsv_to_rgb  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.color import rgb2gray, rgb2hsv  # noqa: E402
from skimage.exposure import rescale_intensity  # noqa: E402
from skimage.feature import hog, local_binary_pattern  # noqa: E402
from skimage.util import img_as_ubyte  # noqa: E402

from extract_features import PARAMS  # noqa: E402

PROJECT_DIR = Path(__file__).parent
META = PROJECT_DIR / "features" / "meta.csv"
FIG_DIR = PROJECT_DIR / "figures"

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
SPLITS = ["train", "val", "test"]
FEATURE_EXAMPLES = ["Jute Red Mite", "Black Hairy", "Jute Stem Weevil"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
})


def style_barh(ax, title, subtitle):
    ax.set_title(f"{title}\n", loc="left", fontsize=13, color=INK)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.tick_params(length=0)


def class_distribution(meta):
    counts = pd.crosstab(meta["class"], meta["split"])[SPLITS]
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 7))
    left = np.zeros(len(counts))
    for split, color in zip(SPLITS, SERIES):
        ax.barh(counts.index, counts[split], left=left, height=0.62, color=color,
                edgecolor=SURFACE, linewidth=1.5, label=split)
        left += counts[split].to_numpy()
    for y, total in enumerate(left):
        ax.text(total + 8, y, f"{int(total)}", va="center", fontsize=9, color=INK_2)
    style_barh(ax, "Images per pest class",
               f"{len(meta):,} images, 17 classes; train sizes range "
               f"{counts['train'].min()}-{counts['train'].max()}")
    ax.set_xlabel("Images")
    ax.set_xlim(0, left.max() * 1.1)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def source_by_class(meta):
    share = pd.crosstab(meta["class"], meta["source"], normalize="index")[["native_256", "resized"]]
    share = share.sort_values("resized")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(share.index, share["native_256"], height=0.62, color=SERIES[0], edgecolor=SURFACE,
            linewidth=1.5, label="native 256x256")
    ax.barh(share.index, share["resized"], left=share["native_256"], height=0.62, color=SERIES[1],
            edgecolor=SURFACE, linewidth=1.5, label="resized from another size")
    for y, r in enumerate(share["resized"]):
        ax.text(1.015, y, f"{r:.0%}", va="center", fontsize=9, color=INK_2)
    style_barh(ax, "Image source varies by class",
               "Share of each class's images that were resized; resized images score far lower in baselines")
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.text(1.015, len(share) - 0.35, "resized", fontsize=8, color=MUTED)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "source_by_class.png", dpi=150)
    plt.close(fig)


def sample_grid(meta):
    classes = sorted(meta["class"].unique())
    cols = 6
    rows = int(np.ceil(len(classes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.35))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, cls in zip(axes.ravel(), classes):
        pool = meta[(meta["class"] == cls) & (meta["split"] == "train") & meta["in_clean_split"]]
        pick = pool.sort_values("original").iloc[len(pool) // 2]
        with Image.open(PROJECT_DIR / pick["processed"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(cls, fontsize=9, color=INK, pad=4)
    fig.suptitle("One training example per class (256x256 after preprocessing)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_grid.png", dpi=150)
    plt.close(fig)


def feature_visualization(meta):
    p_hog, p_lbp = PARAMS["hog"], PARAMS["lbp"]
    n_hue = PARAMS["hsv"]["bins"][0]
    fig, axes = plt.subplots(len(FEATURE_EXAMPLES), 4, figsize=(13, 3.3 * len(FEATURE_EXAMPLES)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    for row, cls in zip(axes, FEATURE_EXAMPLES):
        pick = meta[(meta["class"] == cls) & (meta["source"] == "native_256") & meta["in_clean_split"]].iloc[0]
        with Image.open(PROJECT_DIR / pick["processed"]) as img:
            rgb = np.asarray(img.convert("RGB"))
        gray = rgb2gray(rgb)
        _, hog_img = hog(gray, orientations=p_hog["orientations"], pixels_per_cell=tuple(p_hog["pixels_per_cell"]),
                         cells_per_block=tuple(p_hog["cells_per_block"]), block_norm=p_hog["block_norm"],
                         visualize=True)
        n_points, radius = p_lbp["scales"][0]
        lbp = local_binary_pattern(img_as_ubyte(gray), n_points, radius, method=p_lbp["method"])
        hue = rgb2hsv(rgb)[..., 0].ravel()
        hist, edges = np.histogram(hue, bins=n_hue, range=(0, 1))
        centers = (edges[:-1] + edges[1:]) / 2

        row[0].imshow(rgb)
        row[1].imshow(rescale_intensity(hog_img, in_range=(0, hog_img.max() * 0.6)), cmap="gray_r")
        row[2].imshow(lbp, cmap="gray")
        for ax in row[:3]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        row[0].set_ylabel(cls, fontsize=11, color=INK)
        # Bars are drawn in the hue they count, so the histogram reads as a color profile.
        row[3].bar(centers, hist / hist.sum(), width=0.8 / n_hue,
                   color=hsv_to_rgb(np.column_stack([centers, np.full(n_hue, 0.75), np.full(n_hue, 0.85)])))
        row[3].set_xlim(0, 1)
        row[3].set_xticks([0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1],
                          ["red", "yellow", "green", "cyan", "blue", "magenta", "red"], fontsize=8)
        row[3].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        row[3].grid(axis="y", color=GRID, linewidth=1)
        row[3].set_axisbelow(True)
        row[3].tick_params(length=0)
        row[3].tick_params(axis="x", pad=6)
        for side in ["top", "right", "left"]:
            row[3].spines[side].set_visible(False)
        row[3].spines["bottom"].set_color(BASELINE)
    for ax, title in zip(axes[0], ["Image", "HOG (shape: edge directions)",
                                   "LBP codes (texture), P=8, R=1", "Hue distribution (color)"]):
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_visualization.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    meta = pd.read_csv(META, keep_default_na=False)
    meta["in_clean_split"] = meta["in_clean_split"].astype(str) == "True"
    class_distribution(meta)
    source_by_class(meta)
    sample_grid(meta)
    feature_visualization(meta)
    print(f"Wrote figures to {FIG_DIR.name}/")


if __name__ == "__main__":
    main()
