"""Why did two linear SVM runs never finish?

A multi-class SVC trains one binary SVM per class pair. This refits each pair on its
own, fold by fold, with an iteration limit, for the HOG and HOG+LBP setups with PCA on
HOG. It does this on both tracks, because the same setups finished in seconds on all 17
classes and hung on native12.

Writes results/svm_convergence.csv and prints which pairs hit the limit.

Usage:
    python exp_svm_convergence.py
"""

import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from evaluation import RESULTS_DIR
from experiment_common import LINEAR_SVM_MAX_ITER, PCA_CHOICE_FILE, SEED, TRACKS, build_pipeline, mix_name
from jute_data import cv_folds, load_data

MIXES = [["hog"], ["hog", "lbp"]]


def fit_pair(Z, y, a, b, weights, info):
    mask = (y == a) | (y == b)
    clf = SVC(kernel="linear", class_weight={a: weights[a], b: weights[b]}, max_iter=LINEAR_SVM_MAX_ITER,
              random_state=SEED)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Z[mask], y[mask])
    iterations = int(np.ravel(clf.n_iter_)[0])
    return {**info, "class_a": a, "class_b": b, "n": int(mask.sum()), "iterations": iterations,
            "hit_limit": iterations >= LINEAR_SVM_MAX_ITER, "support_vector_share": clf.n_support_.sum() / mask.sum(),
            "train_accuracy": clf.score(Z[mask], y[mask]), "seconds": time.perf_counter() - start}


def main():
    n_pca = int(PCA_CHOICE_FILE.read_text())
    jobs, names = [], {}
    for track in TRACKS:
        for features in MIXES:
            data = load_data(features, split="original", combine_train_val=True, subset=track)
            names[track] = data.class_names
            for fold, (train_idx, _) in enumerate(cv_folds(data)):
                Z = np.asarray(build_pipeline(data, "passthrough", hog_pca=n_pca).fit_transform(data.X_train[train_idx]),
                               dtype=float)
                y = data.y_train[train_idx]
                classes = np.unique(y)
                weights = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y)))
                info = {"track": track, "features": mix_name(features), "fold": fold}
                jobs += [delayed(fit_pair)(Z, y, a, b, weights, info)
                         for i, a in enumerate(classes) for b in classes[i + 1:]]

    print(f"{len(jobs)} pair fits", flush=True)
    table = pd.DataFrame(Parallel(n_jobs=16, verbose=1)(jobs))
    table["class_a"] = [names[t][c] for t, c in zip(table["track"], table["class_a"])]
    table["class_b"] = [names[t][c] for t, c in zip(table["track"], table["class_b"])]
    table.to_csv(RESULTS_DIR / "svm_convergence.csv", index=False)

    print(table.groupby(["track", "features"]).agg(
        pairs=("n", "size"), hit_limit=("hit_limit", "sum"), median_iter=("iterations", "median"),
        max_iter=("iterations", "max"), median_sv_share=("support_vector_share", "median"),
        total_seconds=("seconds", "sum")).round(3).to_string())
    print("\nPairs that hit the limit, or the slowest 10 if none did:")
    stuck = table[table["hit_limit"]]
    shown = stuck if len(stuck) else table.sort_values("iterations", ascending=False).head(10)
    print(shown.sort_values("seconds", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
