"""What does each change give, and why was the model so good?

One fixed model everywhere, so a difference in score comes from the data, not from
tuning: Random Forest (300 trees) with PCA 25 on HOG.

  matrix   image set (full, crop, insect_only) x split (original, clean, fair) x
           classes (all 17, native only), scored on each split's own test set
  probes   background only vs. insect only; train on one image source, test on the other;
           original-split test images scored by whether their photo was in training

"fair_5fold" rotates through the 5 family-grouped folds, so every distinct photo is scored
exactly once (one image per photo). It is far less noisy than the single fair test set.

Images where no insect was found are left out of the background probe, for every
image set, so all three compare the same photos.

Writes results/fair_matrix.csv, results/fair_probes.csv and results/fair_copy_probe.csv.

Usage:
    python exp_matrix.py
"""

import dataclasses

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluation import evaluate, run_experiment
from experiment_common import PROJECT_DIR, SEED, TRACKS, build_pipeline
from jute_data import FEATURE_FAMILIES, load_data

OUT_DIR = PROJECT_DIR / "results"
IMAGE_SETS = ("full", "crop", "insect_only")
SPLITS = ("original", "clean", "fair")
HOG_PCA = 25
MIN_TRAIN, MIN_TEST = 20, 3  # images per class needed for a source-to-source comparison


def forest():
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)


def summarize(result):
    return {k: result[k] for k in ("n_eval", "accuracy", "macro_f1", "macro_f1_native_256", "macro_f1_resized")}


def all_rows(data):
    """Every image in a fair-split dataset; load it with fair_all_copies=True."""
    return (np.vstack([data.X_train, data.X_test]), np.concatenate([data.y_train, data.y_test]),
            pd.concat([data.meta_train, data.meta_test], ignore_index=True))


def rotate_folds(data, train_rows=None, test_rows=None, hog_pca=HOG_PCA):
    """Hold out each fair fold in turn and score its photos, one image each, so every photo is scored once.

    Returns the pooled evaluate() result and the number of training images in the last fold.
    """
    X, y, meta = all_rows(data)
    train_rows = np.ones(len(y), dtype=bool) if train_rows is None else train_rows
    test_rows = np.ones(len(y), dtype=bool) if test_rows is None else test_rows
    fold, rep = meta["fair_fold"].to_numpy(), meta["fair_rep"].to_numpy()
    truth, predicted, scored = [], [], []
    for k in range(5):
        tr, te = train_rows & (fold != k), test_rows & (fold == k) & rep
        pipe = build_pipeline(data, forest(), hog_pca=hog_pca).fit(X[tr], y[tr])
        truth.append(y[te])
        predicted.append(pipe.predict(X[te]))
        scored.append(meta[te])
    result = evaluate(np.concatenate(truth), np.concatenate(predicted), data.class_names, run_name="rotate_folds",
                      features=data.features, split=data.split, eval_set="fair_5fold",
                      meta=pd.concat(scored, ignore_index=True), save=False)
    return result, int(tr.sum())


def matrix():
    rows = []
    for images in IMAGE_SETS:
        for split in SPLITS:
            for track in TRACKS:
                data = load_data(FEATURE_FAMILIES, split=split, combine_train_val=True, subset=track, images=images)
                pipe = build_pipeline(data, forest(), hog_pca=HOG_PCA)
                r = run_experiment(pipe, data, name=f"fixed_{images}", eval_set="test", save=False)
                row = {"images": images, "split": split, "classes": track, "n_classes": len(data.class_names),
                       "n_train": len(data.y_train), **summarize(r)}
                if split == "fair":
                    copies = load_data(FEATURE_FAMILIES, split=split, combine_train_val=True, subset=track,
                                       images=images, fair_all_copies=True)
                    a = evaluate(copies.y_test, pipe.predict(copies.X_test), copies.class_names, run_name=f"fixed_{images}",
                                 features=copies.features, split=copies.split, eval_set="all_copies", meta=copies.meta_test,
                                 save=False)
                    row.update(all_copies_n=len(copies.y_test), all_copies_macro_f1=a["macro_f1"])
                rows.append(row)
        for track in TRACKS:
            data = load_data(FEATURE_FAMILIES, split="fair", combine_train_val=True, subset=track, images=images,
                             fair_all_copies=True)
            r, n_train = rotate_folds(data)
            rows.append({"images": images, "split": "fair_5fold", "classes": track, "n_classes": len(data.class_names),
                         "n_train": n_train, **summarize(r)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "fair_matrix.csv", index=False)
    return table


def only_found(data, found):
    train, test = data.meta_train["original"].isin(found).to_numpy(), data.meta_test["original"].isin(found).to_numpy()
    return dataclasses.replace(data, X_train=data.X_train[train], y_train=data.y_train[train],
                               meta_train=data.meta_train[train].reset_index(drop=True),
                               X_test=data.X_test[test], y_test=data.y_test[test],
                               meta_test=data.meta_test[test].reset_index(drop=True))


def background_probe():
    """Can color and texture of the background alone name the pest?"""
    boxes = pd.read_csv(PROJECT_DIR / "insect_boxes.csv", keep_default_na=False)
    found = set(boxes.loc[boxes["found"].astype(str) == "True", "original"])
    rows = []
    for track in TRACKS:
        for images in ("full", "insect_only", "background"):
            data = only_found(load_data(["hsv", "lbp"], split="original", combine_train_val=True, subset=track, images=images), found)
            r = run_experiment(build_pipeline(data, forest()), data, name=f"probe_{images}", eval_set="test", save=False)
            rows.append({"probe": "background", "pixels_used": images, "split": "original", "classes": track,
                         "n_classes": len(data.class_names), "n_train": len(data.y_train), **summarize(r)})
            data = only_found(load_data(["hsv", "lbp"], split="fair", combine_train_val=True, subset=track, images=images,
                                        fair_all_copies=True), found)
            r, n_train = rotate_folds(data, hog_pca=None)
            rows.append({"probe": "background", "pixels_used": images, "split": "fair_5fold", "classes": track,
                         "n_classes": len(data.class_names), "n_train": n_train, **summarize(r)})
    return rows


def domain_probe():
    """Train on one image source, test on the other, on classes that have enough of both.

    Rotates through the fair folds, so no photo is ever on both sides.
    """
    rows = []
    for images in ("full", "crop"):
        data = load_data(FEATURE_FAMILIES, split="fair", combine_train_val=True, images=images, fair_all_copies=True)
        _, y, meta = all_rows(data)
        source, rep = meta["source"].to_numpy(), meta["fair_rep"].to_numpy()
        for a in ("native_256", "resized"):
            for b in ("native_256", "resized"):
                train_counts = np.bincount(y[source == a], minlength=len(data.class_names))
                test_counts = np.bincount(y[(source == b) & rep], minlength=len(data.class_names))
                classes = np.where((train_counts >= MIN_TRAIN) & (test_counts >= MIN_TEST))[0]
                r, n_train = rotate_folds(data, (source == a) & np.isin(y, classes), (source == b) & np.isin(y, classes))
                rows.append({"probe": "domain", "pixels_used": images, "split": "fair_5fold", "classes": f"{a} -> {b}",
                             "n_classes": len(classes), "n_train": n_train, **summarize(r)})
    return rows


def copy_probe():
    """On the original split, score test images separately by whether their photo was in training."""
    rows = []
    for track in TRACKS:
        data = load_data(FEATURE_FAMILIES, split="original", combine_train_val=True, subset=track)
        pred = build_pipeline(data, forest(), hog_pca=HOG_PCA).fit(data.X_train, data.y_train).predict(data.X_test)
        in_training = data.meta_test["family"].isin(data.meta_train["family"]).to_numpy()
        for source in ("all", "native_256", "resized"):
            of_source = np.ones(len(pred), dtype=bool) if source == "all" else (data.meta_test["source"] == source).to_numpy()
            for label, seen in (("photo in training", in_training), ("photo not in training", ~in_training)):
                chosen = of_source & seen
                if chosen.any():
                    rows.append({"classes": track, "source": source, "group": label, "n": int(chosen.sum()),
                                 "distinct_photos": data.meta_test.loc[chosen, "family"].nunique(),
                                 "accuracy": (data.y_test[chosen] == pred[chosen]).mean()})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "fair_copy_probe.csv", index=False)
    return table


def main():
    pd.set_option("display.width", 220)
    print(matrix().round(3).to_string(index=False))
    probes = pd.DataFrame(background_probe() + domain_probe())
    probes.to_csv(OUT_DIR / "fair_probes.csv", index=False)
    print(probes.round(3).to_string(index=False))
    print(copy_probe().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
