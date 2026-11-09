"""Clustering: which pests look alike in feature space?

Uses the best feature mix from the 17-class grid, standardized (PCA on HOG if the
grid chose it), on all 7,236 images. No labels are used to build any grouping.

  - Ward hierarchical clustering of each class's average vector -> dendrogram
  - t-SNE 2D map, by image source and one panel per species
  - K-means, scored with adjusted Rand index (ARI: 0 = chance, 1 = perfect) against
    species, insect group and image source

Writes results/clustering_ari.csv and figures/final/{dendrogram,tsne_by_source,tsne_by_species}.png.

Usage:
    python exp_clustering.py
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from evaluation import INK, INK_2, MUTED, RESULTS_DIR, SURFACE, plt
from experiment_common import FIGURE_DIR, PCA_CHOICE_FILE, SEED, build_pipeline
from jute_data import SPLIT, load_data

# Insect group of each class (species names are in the dataset's file names).
GROUPS = {
    "Beet Armyworm": "moth/caterpillar", "Black Hairy": "moth/caterpillar", "Cutworm": "moth/caterpillar",
    "Jute Hairy": "moth/caterpillar", "Jute Semilooper": "moth/caterpillar", "Pod Borer": "moth/caterpillar",
    "Scopula Emissaria": "moth/caterpillar", "Field Cricket": "cricket", "Jute Aphid": "aphid/mealybug",
    "Mealybug": "aphid/mealybug", "Jute Red Mite": "mite", "Yellow Mite": "mite", "Jute Stem Girdler": "beetle",
    "Jute Stem Weevil": "beetle", "Leaf Beetle": "beetle", "Termite": "termite",
    "Termite odontotermes (Rambur)": "termite",
}
SOURCE_COLORS = {"native_256": "#2a78d6", "resized": "#eb6834"}
SOURCE_LABELS = {"native_256": "Native 256×256", "resized": "Resized"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_best():
    grid = pd.read_csv(RESULTS_DIR / "grid_summary.csv")
    best = grid[(grid["split"] == SPLIT) & (grid["eval_set"] == "cv")].sort_values("macro_f1", ascending=False).iloc[0]
    features = best["features"].split("+")
    n_pca = int(PCA_CHOICE_FILE.read_text()) if best["pca"] else None
    data = load_data(features, split=SPLIT, combine_train_val=True, fair_all_copies=True)
    X = np.vstack([data.X_train, data.X_test])
    meta = pd.concat([data.meta_train, data.meta_test], ignore_index=True)
    Z = build_pipeline(data, "passthrough", hog_pca=n_pca).fit_transform(X)
    print(f"Clustering on {best['features']} (HOG PCA {n_pca}): {Z.shape}", flush=True)
    return np.asarray(Z, dtype=float), meta, data.class_names, best["features"]


def plot_dendrogram(link, class_names, path):
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    style(ax)
    labels = [f"{c}  ({GROUPS[c]})" for c in class_names]
    dendrogram(link, labels=labels, orientation="right", color_threshold=0, above_threshold_color=INK_2, ax=ax)
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=INK_2)
    ax.tick_params(axis="x", length=0, labelcolor=MUTED)
    ax.set_xlabel("Ward linkage distance (further left = merged sooner = more alike)", color=INK_2)
    ax.set_title("Which pests look alike: hierarchical clustering of class averages", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_by_source(xy, meta, path):
    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor=SURFACE)
    style(ax)
    for source, color in SOURCE_COLORS.items():
        m = meta["source"].to_numpy() == source
        ax.scatter(xy[m, 0], xy[m, 1], s=5, color=color, linewidths=0, alpha=0.7, label=SOURCE_LABELS[source])
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=10, labelcolor=INK_2, markerscale=3, loc="upper right")
    ax.set_title("t-SNE map of all images, by image source", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_by_species(xy, meta, class_names, path):
    fig, axes = plt.subplots(3, 6, figsize=(15, 8), facecolor=SURFACE)
    classes = meta["class"].to_numpy()
    for ax, name in zip(axes.flat, class_names):
        style(ax)
        ax.scatter(xy[:, 0], xy[:, 1], s=1, color="#d8d6d0", linewidths=0)
        m = classes == name
        ax.scatter(xy[m, 0], xy[m, 1], s=4, color="#2a78d6", linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        short = name.replace("Termite odontotermes (Rambur)", "Termite odontotermes")
        ax.set_title(f"{short}\n{GROUPS[name]}", fontsize=9, color=INK_2, loc="left")
    for ax in list(axes.flat)[len(class_names):]:
        ax.axis("off")
    fig.suptitle("t-SNE map: where each species lands", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def kmeans_ari(Z, truth):
    k = len(np.unique(truth))
    return adjusted_rand_score(truth, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z))


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    Z, meta, class_names, mix = load_best()
    species = meta["class"].to_numpy()
    group = meta["class"].map(GROUPS).to_numpy()
    source = meta["source"].to_numpy()

    means = np.stack([Z[species == c].mean(axis=0) for c in class_names])
    link = linkage(means, method="ward")
    plot_dendrogram(link, class_names, FIGURE_DIR / "dendrogram.png")

    n_groups = len(set(GROUPS.values()))
    class_groups = [GROUPS[c] for c in class_names]
    print("ARI, class dendrogram cut into", n_groups, "groups vs insect group:",
          round(adjusted_rand_score(class_groups, fcluster(link, n_groups, "maxclust")), 3))

    rows = []
    for name, mask in (("all images", np.ones(len(meta), dtype=bool)), ("native only", source == "native_256"),
                       ("resized only", source == "resized")):
        row = {"images": name, "n": int(mask.sum()), "ari_species": kmeans_ari(Z[mask], species[mask]),
               "ari_insect_group": kmeans_ari(Z[mask], group[mask])}
        if name == "all images":
            row["ari_source"] = kmeans_ari(Z[mask], source[mask])
        rows.append(row)
    table = pd.DataFrame(rows).assign(features=mix)
    table.to_csv(RESULTS_DIR / "clustering_ari.csv", index=False)
    print(table.to_string(index=False), flush=True)

    reduced = PCA(50, random_state=SEED).fit_transform(Z) if Z.shape[1] > 50 else Z
    xy = TSNE(perplexity=30, init="pca", learning_rate="auto", random_state=SEED, n_jobs=-1).fit_transform(reduced)
    plot_by_source(xy, meta, FIGURE_DIR / "tsne_by_source.png")
    plot_by_species(xy, meta, class_names, FIGURE_DIR / "tsne_by_species.png")
    print("Saved figures to figures/final/")


if __name__ == "__main__":
    main()
