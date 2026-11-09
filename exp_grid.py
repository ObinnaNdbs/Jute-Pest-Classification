"""Step 3: every feature mix on every model, on both tracks.

11 configurations (7 mixes without PCA + the 4 HOG mixes with PCA on HOG) x 6 models
x 2 tracks = 132 grouped-CV runs on train+val. The 17-class track is also scored on the
supplied val set so it lines up with the midpoint tables. Finished runs are skipped,
so an interrupted grid can be restarted.

Writes results/grid_summary.csv and heatmaps to figures/final/.

Usage:
    python exp_grid.py            # needs results/pca_choice.txt from exp_pca.py
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap

from evaluation import (BLUE_RAMP, INK, INK_2, MUTED, RESULTS_DIR, SURFACE, consolidate_results, load_results,
                        plt, run_cv, run_experiment)
from experiment_common import (FEATURE_MIXES, FIGURE_DIR, LINEAR_SVM_MAX_ITER, N_WORKERS, PCA_CHOICE_FILE, TRACKS,
                               build_pipeline, make_classifiers, mix_name)
from jute_data import SPLIT, load_data

MODEL_ORDER = ["log_reg", "knn", "linear_svm", "rbf_svm", "decision_tree", "random_forest"]
MODEL_LABELS = ["Logistic reg.", "KNN", "Linear SVM", "RBF SVM", "Decision tree", "Random Forest"]


def split_name(track):
    return SPLIT if track == "all" else f"{SPLIT}_native12"


def run_one(track, features, use_pca, n_pca, model_name):
    run_name = f"grid_{model_name}" + ("_pca" if use_pca else "")
    n = n_pca if use_pca else None
    notes = f"HOG PCA {n}" if use_pca else "no PCA"
    if model_name == "linear_svm":
        notes += f"; iteration limit {LINEAR_SVM_MAX_ITER} per class pair"
    data = load_data(features, split=SPLIT, combine_train_val=True, subset=track)
    run_cv(build_pipeline(data, make_classifiers()[model_name], hog_pca=n), data, name=run_name, n_jobs=1,
           notes=f"{notes}; grouped 5-fold CV on train+val")
    if track == "all" and SPLIT == "original":
        data = load_data(features, split="original")
        run_experiment(build_pipeline(data, make_classifiers()[model_name], hog_pca=n), data, name=run_name,
                       eval_set="val", notes=notes)


def finished_runs():
    try:
        table = load_results()
    except (FileNotFoundError, ValueError):
        return set()
    return set(zip(table["model"], table["features"], table["split"], table["eval_set"]))


def heatmap(values, row_labels, title, path):
    fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    low, high = np.nanmin(values), np.nanmax(values)
    mesh = ax.pcolormesh(values, cmap=cmap, vmin=low, vmax=high, edgecolors=SURFACE, linewidth=2)
    best = np.nanmax(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            share = (v - low) / max(high - low, 1e-9)
            ax.text(j + 0.5, i + 0.5, f"{v:.3f}".lstrip("0"), ha="center", va="center", fontsize=9,
                    fontweight="bold" if v == best else "normal", color="#ffffff" if share >= 0.55 else INK)
    ax.set_xlim(0, values.shape[1])
    ax.set_ylim(values.shape[0], 0)
    ax.set_xticks(np.arange(values.shape[1]) + 0.5, MODEL_LABELS, fontsize=9, color=INK_2)
    ax.set_yticks(np.arange(values.shape[0]) + 0.5, row_labels, fontsize=9, color=INK_2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    n_pca = int(PCA_CHOICE_FILE.read_text())
    configs = [(f, False) for f in FEATURE_MIXES] + [(f, True) for f in FEATURE_MIXES if "hog" in f]
    done = finished_runs()
    jobs = []
    for track in TRACKS:
        for features, use_pca in configs:
            for model_name in MODEL_ORDER:
                run_name = f"grid_{model_name}" + ("_pca" if use_pca else "")
                if (run_name, mix_name(features), split_name(track), "cv") not in done:
                    jobs.append((track, features, use_pca, n_pca, model_name))
    print(f"{len(jobs)} of {len(configs) * len(MODEL_ORDER) * len(TRACKS)} grid runs to do", flush=True)
    Parallel(n_jobs=N_WORKERS, batch_size=1, verbose=5)(delayed(run_one)(*j) for j in jobs)

    results = consolidate_results()
    grid = results[results["model"].str.startswith("grid_")].copy()
    grid["classifier"] = grid["model"].str.removeprefix("grid_").str.removesuffix("_pca")
    grid["pca"] = grid["model"].str.endswith("_pca")
    keep = ["split", "eval_set", "classifier", "features", "pca", "n_features", "accuracy", "macro_f1",
            "macro_f1_native_256", "macro_f1_resized", "fit_seconds"]
    grid[keep].sort_values(["split", "eval_set", "macro_f1"], ascending=[True, True, False]).to_csv(
        RESULTS_DIR / "grid_summary.csv", index=False)

    row_keys = [(mix_name(f), p) for f, p in configs]
    row_labels = [m.upper().replace("+", " + ") + (f" (PCA {n_pca})" if p else "") for m, p in row_keys]

    def matrix(split, eval_set, column):
        sub = grid[(grid["split"] == split) & (grid["eval_set"] == eval_set)]
        lookup = {(r.features, r.pca, r.classifier): getattr(r, column) for r in sub.itertuples()}
        return np.array([[lookup.get((m, p, c), np.nan) for c in MODEL_ORDER] for m, p in row_keys])

    heatmap(matrix(split_name("all"), "cv", "macro_f1"), row_labels,
            "17 classes, all images: grouped CV macro-F1", FIGURE_DIR / "grid_17class_cv.png")
    heatmap(matrix(split_name("all"), "cv", "macro_f1_resized"), row_labels,
            "17 classes: grouped CV macro-F1 on resized images only", FIGURE_DIR / "grid_17class_cv_resized.png")
    if SPLIT == "original":
        heatmap(matrix("original", "val", "macro_f1"), row_labels,
                "17 classes, all images: validation macro-F1", FIGURE_DIR / "grid_17class_val.png")
    heatmap(matrix(split_name("native12"), "cv", "macro_f1"), row_labels,
            "Native-image classes only: grouped CV macro-F1", FIGURE_DIR / "grid_native12_cv.png")

    n_rows = len(grid[grid["eval_set"] == "cv"])
    print(f"\nGrid CV rows recorded: {n_rows} (expected {len(configs) * len(MODEL_ORDER) * len(TRACKS)})")
    for split in (split_name("all"), split_name("native12")):
        print(f"\nTop 5 for {split} (CV):")
        print(grid[(grid["split"] == split) & (grid["eval_set"] == "cv")].sort_values("macro_f1", ascending=False)
              [["classifier", "features", "pca", "macro_f1", "macro_f1_native_256", "macro_f1_resized"]]
              .head(5).to_string(index=False))


if __name__ == "__main__":
    main()
