"""Step 2: does PCA make HOG useful? HOG alone, grouped 5-fold CV, both tracks.

Component counts 10-1000 plus no PCA, on logistic regression, RBF SVM and Random
Forest. The count used by exp_grid.py is the one with the best mean macro-F1 over the
three models on the 17-class track (the native12 track sits too near the ceiling).

Writes results/pca_sweep.csv, results/pca_choice.txt and figures/final/pca_sweep.png.

Usage:
    python exp_pca.py
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from evaluation import INK, INK_2, MUTED, RESULTS_DIR, SURFACE, plt, run_cv
from experiment_common import (FIGURE_DIR, N_WORKERS, PCA_CHOICE_FILE, TRACKS, build_pipeline,
                               make_classifiers)
from jute_data import SPLIT, load_data

COMPONENTS = [10, 15, 25, 50, 100, 200, 500, 1000, None]
MODELS = ["log_reg", "rbf_svm", "random_forest"]
MODEL_LABELS = {"log_reg": "Logistic regression", "rbf_svm": "RBF SVM", "random_forest": "Random Forest"}
SERIES_COLORS = {"log_reg": "#2a78d6", "rbf_svm": "#eb6834", "random_forest": "#1baf7a"}
TRACK_LABELS = {"all": "All 17 classes, all images", "native12": "12 classes, native images only"}


def run_one(track, model_name, n):
    data = load_data(["hog"], split=SPLIT, combine_train_val=True, subset=track)
    pipe = build_pipeline(data, make_classifiers()[model_name], hog_pca=n)
    suffix = "nopca" if n is None else f"n{n}"
    r = run_cv(pipe, data, name=f"pca_{model_name}_{suffix}", n_jobs=1)
    return {"track": track, "model": model_name, "components": n, "macro_f1": r["macro_f1"],
            "macro_f1_native_256": r["macro_f1_native_256"], "macro_f1_resized": r["macro_f1_resized"],
            "fit_seconds": r["fit_seconds"]}


def plot(table, path):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(11.5, 4.8), facecolor=SURFACE)
    positions = np.arange(len(COMPONENTS))
    for ax, track in zip(axes, TRACKS):
        ax.set_facecolor(SURFACE)
        sub = table[table["track"] == track]
        ends = {}
        for model in MODELS:
            line = sub[sub["model"] == model].set_index(sub[sub["model"] == model]["components"].fillna(-1))
            y = [line.loc[-1 if c is None else c, "macro_f1"] for c in COMPONENTS]
            ax.plot(positions, y, color=SERIES_COLORS[model], linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODEL_LABELS[model])
            ends[model] = y[-1]
        gap = 0.035  # keep the end labels from overlapping
        placed = []
        for model, y in sorted(ends.items(), key=lambda kv: kv[1]):
            y_text = max([y] + [p + gap for p in placed])
            placed.append(y_text)
            ax.text(positions[-1] + 0.15, y_text, MODEL_LABELS[model], va="center", fontsize=9, color=INK_2)
        ax.set_xticks(positions, [("No PCA" if c is None else str(c)) for c in COMPONENTS], fontsize=9, color=INK_2)
        ax.set_xlim(-0.3, len(COMPONENTS) + 0.75)
        ax.set_xlabel("PCA components on HOG", color=INK_2)
        ax.set_title(TRACK_LABELS[track], loc="left", color=INK, fontsize=11)
        ax.grid(axis="y", color="#e8e6e1", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Grouped 5-fold CV macro-F1", color=INK_2)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(t, m, n) for t in TRACKS for m in MODELS for n in COMPONENTS]
    rows = Parallel(n_jobs=N_WORKERS, batch_size=1)(delayed(run_one)(*j) for j in jobs)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "pca_sweep.csv", index=False)

    means = table[(table["track"] == "all") & table["components"].notna()].groupby("components")["macro_f1"].mean()
    choice = int(means.idxmax())
    PCA_CHOICE_FILE.write_text(str(choice))
    plot(table, FIGURE_DIR / "pca_sweep.png")
    print(table.to_string())
    print(f"\nChosen HOG PCA components: {choice}")


if __name__ == "__main__":
    main()
