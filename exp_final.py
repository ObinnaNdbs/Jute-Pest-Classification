"""Steps 4-8: class weights, tuning, final test, cross-source check, leakage.

Two stages, so the test set is only touched after every choice is locked in:

    python exp_final.py select   # class weights + tuning with grouped CV; writes results/final_models.json
    python exp_final.py test     # scores those fixed models once on test; writes results/final_results.csv

"select" takes the best configuration of each classifier from the grid and keeps the
top 3 classifiers per track. Models are ranked by CV, never by test score.
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from evaluation import (RESULTS_DIR, consolidate_results, evaluate, plot_confusion_matrix, run_cv,
                        run_experiment)
from experiment_common import (FIGURE_DIR, N_WORKERS, PCA_CHOICE_FILE, TRACKS, build_pipeline,
                               make_classifiers)
from jute_data import SPLIT, cv_folds, load_data

MODELS_FILE = RESULTS_DIR / "final_models.json"
PARAM_GRIDS = {
    "log_reg": {"C": [0.01, 0.1, 1, 10]},
    "knn": {"n_neighbors": [1, 3, 5, 9, 15], "weights": ["uniform", "distance"]},
    "linear_svm": {"C": [0.01, 0.1, 1, 10]},
    "rbf_svm": {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.0001, 0.001, 0.01]},
    "decision_tree": {"max_depth": [10, 20, None], "min_samples_leaf": [1, 5, 10]},
    "random_forest": {"n_estimators": [300], "max_features": ["sqrt", 0.1], "min_samples_leaf": [1, 2, 5],
                      "max_depth": [None, 20]},
}


def top_configs(track, n=3):
    """Best grid configuration of each classifier, top n classifiers. Ties go to the faster fit."""
    grid = pd.read_csv(RESULTS_DIR / "grid_summary.csv")
    split = SPLIT if track == "all" else f"{SPLIT}_native12"
    grid = grid[(grid["split"] == split) & (grid["eval_set"] == "cv")]
    best = grid.sort_values(["macro_f1", "fit_seconds"], ascending=[False, True]).drop_duplicates("classifier")
    return best.head(n).to_dict("records")


def small_class_recall(result, data, n=3):
    """Mean recall over the n classes with the fewest training images."""
    smallest = np.argsort(np.bincount(data.y_train, minlength=len(data.class_names)))[:n]
    return float(np.mean([result[f"recall[{data.class_names[i]}]"] for i in smallest]))


def select():
    n_pca = int(PCA_CHOICE_FILE.read_text())
    chosen, weight_rows = [], []
    for track in TRACKS:
        for cfg in top_configs(track):
            classifier, features, use_pca = cfg["classifier"], cfg["features"].split("+"), bool(cfg["pca"])
            n = n_pca if use_pca else None
            data = load_data(features, split=SPLIT, combine_train_val=True, subset=track)

            weight = None
            if classifier != "knn":  # KNN has no class_weight option
                scores = {}
                for option, label in (("balanced", "balanced"), (None, "none")):
                    r = run_cv(build_pipeline(data, make_classifiers(option)[classifier], hog_pca=n), data,
                               name=f"weights_{label}_{classifier}", n_jobs=N_WORKERS)
                    scores[option] = r["macro_f1"]
                    weight_rows.append({"track": track, "classifier": classifier, "features": cfg["features"],
                                        "pca": use_pca, "class_weight": label, "cv_macro_f1": r["macro_f1"],
                                        "cv_accuracy": r["accuracy"],
                                        "recall_3_smallest_classes": small_class_recall(r, data)})
                weight = "balanced" if scores["balanced"] >= scores[None] else None

            pipe = build_pipeline(data, make_classifiers(weight)[classifier], hog_pca=n)
            step = pipe.steps[-1][0]
            search = GridSearchCV(pipe, {f"{step}__{k}": v for k, v in PARAM_GRIDS[classifier].items()},
                                  scoring="f1_macro", cv=cv_folds(data), n_jobs=N_WORKERS, refit=False)
            search.fit(data.X_train, data.y_train)
            best = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}
            chosen.append({"track": track, "classifier": classifier, "features": features, "hog_pca": n,
                           "class_weight": weight, "params": best, "cv_macro_f1": float(search.best_score_)})
            print(f"{track}: {classifier} on {cfg['features']} (PCA {n}) weight={weight} "
                  f"-> {best}  CV macro-F1 {search.best_score_:.3f}", flush=True)

    pd.DataFrame(weight_rows).to_csv(RESULTS_DIR / "class_weights.csv", index=False)
    MODELS_FILE.write_text(json.dumps(chosen, indent=2))
    consolidate_results()


def final_pipeline(data, cfg):
    clf = make_classifiers(cfg["class_weight"])[cfg["classifier"]].set_params(**cfg["params"])
    return build_pipeline(data, clf, hog_pca=cfg["hog_pca"])


def resized_same_classes(features, class_names):
    """Resized images of the native-track classes, relabeled to match them.

    On the original split that is every resized image. On the fair split it is only the
    test side, one image per photo, so no copy of a training photo can be scored.
    """
    if SPLIT == "fair":
        full = load_data(features, split="fair")
        X, meta = full.X_test, full.meta_test
    else:
        full = load_data(features, split="original")
        X = np.vstack([full.X_train, full.X_val, full.X_test])
        meta = pd.concat([full.meta_train, full.meta_val, full.meta_test], ignore_index=True)
    mask = ((meta["source"] == "resized") & meta["class"].isin(class_names)).to_numpy()
    index = {c: i for i, c in enumerate(class_names)}
    meta = meta[mask].reset_index(drop=True)
    return X[mask], meta["class"].map(index).to_numpy(), meta


def confused_pairs(cm, class_names, top=10):
    off = np.array(cm, dtype=int)
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:top]
    rows = [np.unravel_index(i, off.shape) for i in order]
    return pd.DataFrame([{"true": class_names[a], "predicted": class_names[b], "count": int(off[a, b])}
                         for a, b in rows if off[a, b] > 0], columns=["true", "predicted", "count"])


def test():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    models = json.loads(MODELS_FILE.read_text())
    rows = []
    for track in TRACKS:
        configs = sorted((m for m in models if m["track"] == track), key=lambda m: -m["cv_macro_f1"])
        for rank, cfg in enumerate(configs):
            features, classifier = cfg["features"], cfg["classifier"]
            notes = json.dumps({k: cfg[k] for k in ("hog_pca", "class_weight", "params")})

            data = load_data(features, split=SPLIT, combine_train_val=True, subset=track)
            pipe = final_pipeline(data, cfg)
            r = run_experiment(pipe, data, name=f"final_{classifier}", eval_set="test", notes=notes)
            row = {"track": track, "classifier": classifier, "features": "+".join(features), "hog_pca": cfg["hog_pca"],
                   "class_weight": cfg["class_weight"], "params": json.dumps(cfg["params"]),
                   "cv_macro_f1": cfg["cv_macro_f1"], "test_accuracy": r["accuracy"], "test_macro_f1": r["macro_f1"],
                   "test_macro_f1_native_256": r["macro_f1_native_256"], "test_macro_f1_resized": r["macro_f1_resized"],
                   "chosen_by_cv": rank == 0}

            if rank == 0:
                tag = "17class" if track == "all" else "native12"
                plot_confusion_matrix(r["confusion_matrix"], data.class_names, path=FIGURE_DIR / f"cm_final_{tag}.png",
                                      title=f"Final model on test: {classifier}, {tag}")
                confused_pairs(r["confusion_matrix"], data.class_names).to_csv(
                    RESULTS_DIR / f"confused_pairs_{tag}.csv", index=False)

            if track == "native12":
                X, y, meta = resized_same_classes(features, data.class_names)
                cross = evaluate(y, pipe.predict(X), data.class_names, run_name=f"cross_source_{classifier}",
                                 features=features, split=data.split, eval_set="resized_all_splits", meta=meta,
                                 n_features=data.n_features, notes="trained on native12 train+val")
                row.update({"cross_source_n": len(y), "cross_source_accuracy": cross["accuracy"],
                            "cross_source_macro_f1": cross["macro_f1"]})

            if SPLIT == "fair":
                copies = load_data(features, split="fair", combine_train_val=True, subset=track, fair_all_copies=True)
                a = evaluate(copies.y_test, pipe.predict(copies.X_test), copies.class_names,
                             run_name=f"final_all_copies_{classifier}", features=features, split=copies.split,
                             eval_set="test_all_copies", meta=copies.meta_test, n_features=copies.n_features, notes=notes)
                row.update({"all_copies_test_n": len(copies.y_test), "all_copies_test_macro_f1": a["macro_f1"]})
            else:
                clean = load_data(features, split="clean", combine_train_val=True, subset=track)
                c = run_experiment(final_pipeline(clean, cfg), clean, name=f"final_{classifier}", eval_set="test", notes=notes)
                row.update({"clean_test_accuracy": c["accuracy"], "clean_test_macro_f1": c["macro_f1"],
                            "clean_test_macro_f1_native_256": c["macro_f1_native_256"],
                            "clean_test_macro_f1_resized": c["macro_f1_resized"]})
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "final_results.csv", index=False)
    consolidate_results()
    print(table.drop(columns=["params"]).to_string(index=False))


if __name__ == "__main__":
    {"select": select, "test": test}[sys.argv[1]]()
