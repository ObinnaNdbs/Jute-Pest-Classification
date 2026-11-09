"""Do the pests form chunks, and does that depend on the image set?

exp_clustering.py uses each image set's own best feature mix and every copy of every photo,
so its numbers can't be compared across image sets, and copies of one photo form chunks of
their own. Here full, crop and insect-only all get the same features, and each is measured
two ways: one image per distinct photo (the fair view), and every copy (to show what copies do).
No labels are used to build any grouping.

  K-means             k = 17 vs. species, k = 6 vs. insect group, k = 2 vs. image source (ARI: 0 = chance, 1 = perfect)
  nearest neighbor    is the closest image from a different photo the same species / group?
  subsamples          200 random 80% samples of the photos, the same ones for every image set
  Ward dendrograms    of class averages, and the first class each class joins
  t-SNE maps          per image set and insect group

Writes results/fair_cluster_compare.csv, fair_cluster_differences.csv, fair_cluster_merges.csv,
fair_cluster_by_group.csv and figures/cluster_compare/.

Usage:
    python exp_cluster_compare.py
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, pairwise_distances

from evaluation import INK, INK_2, SURFACE, plt
from exp_clustering import GROUPS, style
from exp_matrix import IMAGE_SETS, OUT_DIR, all_rows
from experiment_common import PROJECT_DIR, SEED, build_pipeline
from jute_data import FEATURE_FAMILIES, load_data

FIGURE_DIR = PROJECT_DIR / "figures" / "cluster_compare"
MIXES = {"hsv+lbp": (["hsv", "lbp"], None), "hog+hsv+lbp": (list(FEATURE_FAMILIES), 50)}
CHART_MIX = "hog+hsv+lbp"
VIEWS = ("one per photo", "every copy")
TITLES = {"full": "Full image", "crop": "Crop", "insect_only": "Insect only"}
GROUP_ORDER = ["moth/caterpillar", "beetle", "termite", "mite", "aphid/mealybug", "cricket"]
N_SUBSAMPLES, SUBSAMPLE_SHARE = 200, 0.8
HIGHLIGHT, BACKGROUND = "#2a78d6", "#d8d6d0"
COMPARISONS = (("crop", "full"), ("insect_only", "full"), ("insect_only", "crop"))


def embed(images, mix, view):
    """Standardized features (PCA on HOG if the mix uses it), fit on the rows being clustered."""
    features, n_pca = MIXES[mix]
    data = load_data(features, split="fair", combine_train_val=True, images=images, fair_all_copies=True)
    X, _, meta = all_rows(data)
    keep = meta["fair_rep"].to_numpy() if view == "one per photo" else np.ones(len(meta), dtype=bool)
    Z = build_pipeline(data, "passthrough", hog_pca=n_pca).fit_transform(X[keep])
    return np.asarray(Z, dtype=float), meta[keep].reset_index(drop=True), data.class_names


def kmeans_ari(Z, truth, n_init=10, seed=SEED):
    k = len(np.unique(truth))
    return adjusted_rand_score(truth, KMeans(k, n_init=n_init, random_state=seed).fit_predict(Z))


def nearest_other_photo(Z, family):
    """Index of each image's closest neighbor among images of a different photo."""
    nearest = np.empty(len(Z), dtype=int)
    for start in range(0, len(Z), 500):
        block = np.arange(start, min(start + 500, len(Z)))
        dist = pairwise_distances(Z[block], Z)
        dist[family[block][:, None] == family[None, :]] = np.inf
        nearest[block] = dist.argmin(axis=1)
    return nearest


def share_of_pairs_matching(labels):
    """Chance that two different images, picked at random, share a label."""
    counts = np.unique(labels, return_counts=True)[1]
    return float((counts * (counts - 1)).sum() / (len(labels) * (len(labels) - 1)))


def subsample_scores(Z, species, group, chosen, seed):
    return kmeans_ari(Z[chosen], species[chosen], 3, seed), kmeans_ari(Z[chosen], group[chosen], 3, seed)


def class_structure(Z, meta, class_names):
    """Ward tree of class averages, and how often a class's closest class is in its insect group."""
    species = meta["class"].to_numpy()
    means = np.stack([Z[species == c].mean(axis=0) for c in class_names])
    link = linkage(means, method="ward")
    groups = [GROUPS[c] for c in class_names]
    dist = pairwise_distances(means)
    np.fill_diagonal(dist, np.inf)
    closest = dist.argmin(axis=1)
    merges = [{"class": c, "group": g, "closest_class": class_names[j], "same_group": groups[j] == g}
              for c, g, j in zip(class_names, groups, closest)]
    return link, merges, adjusted_rand_score(groups, fcluster(link, len(set(groups)), "maxclust"))


def tsne(Z):
    reduced = PCA(50, random_state=SEED).fit_transform(Z) if Z.shape[1] > 50 else Z
    return TSNE(perplexity=30, init="pca", learning_rate="auto", random_state=SEED, n_jobs=-1).fit_transform(reduced)


def plot_tsne(maps, view, path):
    fig, axes = plt.subplots(len(IMAGE_SETS), len(GROUP_ORDER), figsize=(15, 8), facecolor=SURFACE)
    for row, images in zip(axes, IMAGE_SETS):
        xy, group = maps[images]
        size = 6 if view == "one per photo" else 2
        for ax, name in zip(row, GROUP_ORDER):
            style(ax)
            on = group == name
            ax.scatter(xy[~on, 0], xy[~on, 1], s=size * 0.6, color=BACKGROUND, linewidths=0)
            ax.scatter(xy[on, 0], xy[on, 1], s=size, color=HIGHLIGHT, linewidths=0)
            ax.set_xticks([]); ax.set_yticks([])
        row[0].set_ylabel(TITLES[images], fontsize=11, color=INK)
    for ax, name in zip(axes[0], GROUP_ORDER):
        ax.set_title(name, fontsize=11, color=INK_2, loc="left")
    fig.suptitle(f"t-SNE maps of pest groups, HOG + HSV + LBP, {view}", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_dendrograms(trees, path):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True, facecolor=SURFACE)
    for ax, images in zip(axes, IMAGE_SETS):
        style(ax)
        link, class_names = trees[images]
        labels = [f"{c.replace('Termite odontotermes (Rambur)', 'Termite odontotermes')}  ({GROUPS[c]})" for c in class_names]
        dendrogram(link, labels=labels, orientation="right", color_threshold=0, above_threshold_color=INK_2, ax=ax)
        ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=INK_2)
        ax.tick_params(axis="x", length=0, labelcolor=INK_2)
        ax.set_xlabel("Ward distance (further left = more alike)", color=INK_2)
        ax.set_title(TITLES[images], loc="left", color=INK, fontsize=12)
    fig.suptitle("Which pests look alike, one image per photo, HOG + HSV + LBP", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows, merge_rows, difference_rows, group_rows = [], [], [], []
    rng = np.random.default_rng(0)
    for mix in MIXES:
        for view in VIEWS:
            embedded = {images: embed(images, mix, view) for images in IMAGE_SETS}
            meta = embedded["full"][1]
            for images in IMAGE_SETS:
                assert (embedded[images][1]["original"] == meta["original"]).all()
            species = meta["class"].to_numpy()
            group = meta["class"].map(GROUPS).to_numpy()
            source, family = meta["source"].to_numpy(), meta["family"].to_numpy()
            samples = None
            if view == "one per photo":
                size = int(SUBSAMPLE_SHARE * len(meta))
                samples = [rng.choice(len(meta), size, replace=False) for _ in range(N_SUBSAMPLES)]

            spread, maps, trees = {}, {}, {}
            for images in IMAGE_SETS:
                Z, _, class_names = embedded[images]
                nearest = nearest_other_photo(Z, family)
                link, merges, cut_ari = class_structure(Z, meta, class_names)
                row = {"mix": mix, "view": view, "images": images, "n": len(meta),
                       "ari_species": kmeans_ari(Z, species), "ari_insect_group": kmeans_ari(Z, group), "ari_source": kmeans_ari(Z, source),
                       "nn_same_species": (species[nearest] == species).mean(), "nn_same_species_chance": share_of_pairs_matching(species),
                       "nn_same_group": (group[nearest] == group).mean(), "nn_same_group_chance": share_of_pairs_matching(group),
                       "classes_nearest_in_same_group": sum(m["same_group"] for m in merges), "tree_cut_ari_vs_group": cut_ari}
                if samples is not None:
                    spread[images] = np.array(Parallel(n_jobs=-1)(
                        delayed(subsample_scores)(Z, species, group, chosen, SEED + i) for i, chosen in enumerate(samples)))
                    row.update(ari_species_low=np.percentile(spread[images][:, 0], 2.5), ari_species_high=np.percentile(spread[images][:, 0], 97.5),
                               ari_group_low=np.percentile(spread[images][:, 1], 2.5), ari_group_high=np.percentile(spread[images][:, 1], 97.5))
                rows.append(row)
                merge_rows += [{"mix": mix, "view": view, "images": images, **m} for m in merges]
                for name in GROUP_ORDER:
                    of_group = group == name
                    group_rows.append({"mix": mix, "view": view, "images": images, "group": name, "n": int(of_group.sum()),
                                       "nn_same_group": (group[nearest[of_group]] == name).mean(),
                                       "nn_same_group_chance": (of_group.sum() - 1) / (len(group) - 1),
                                       "nn_same_species": (species[nearest[of_group]] == species[of_group]).mean()})
                if mix == CHART_MIX:
                    maps[images] = (tsne(Z), group)
                    trees[images] = (link, class_names)
                print(f"{mix} | {view} | {images}: done", flush=True)
            if samples is not None:
                for a, b in COMPARISONS:
                    for k, target in enumerate(("species", "insect group")):
                        diff = spread[a][:, k] - spread[b][:, k]
                        difference_rows.append({"mix": mix, "compared": f"{a} minus {b}", "target": target, "mean_difference": diff.mean(),
                                                "low": np.percentile(diff, 2.5), "high": np.percentile(diff, 97.5)})
            if mix == CHART_MIX:
                plot_tsne(maps, view, FIGURE_DIR / f"tsne_groups_{view.replace(' ', '_')}.png")
                if view == "one per photo":
                    plot_dendrograms(trees, FIGURE_DIR / "dendrograms_one_per_photo.png")

    table, differences, merged = pd.DataFrame(rows), pd.DataFrame(difference_rows), pd.DataFrame(merge_rows)
    table.to_csv(OUT_DIR / "fair_cluster_compare.csv", index=False)
    differences.to_csv(OUT_DIR / "fair_cluster_differences.csv", index=False)
    merged.to_csv(OUT_DIR / "fair_cluster_merges.csv", index=False)
    by_group = pd.DataFrame(group_rows)
    by_group.to_csv(OUT_DIR / "fair_cluster_by_group.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 30, "display.max_rows", 100)
    print(table.round(3).to_string(index=False))
    print(differences.round(3).to_string(index=False))
    print(by_group[(by_group["mix"] == CHART_MIX) & (by_group["view"] == "one per photo")].round(3).to_string(index=False))
    print(merged[(merged["mix"] == CHART_MIX) & (merged["view"] == "one per photo")].to_string(index=False))


if __name__ == "__main__":
    main()
