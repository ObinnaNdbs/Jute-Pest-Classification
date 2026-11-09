"""Does tuning each image set separately beat one fixed model?

Takes the models exp_final.py picked for each image set (three per class set) and scores
them across all 5 fair folds, one image per photo, next to exp_matrix.py's fixed model.
Every score uses the same resampled photos as exp_bootstrap.py, so the 95% intervals on
scores and differences are directly comparable.

Tuned settings were chosen by grouped CV on folds 1-4, so those folds are not fully unseen
by the tuned models. Their 5-fold scores lean slightly optimistic. Fold 0 alone is the
fair test score in final_results.csv.

Writes results/fair_tuned_5fold.csv.

Usage:
    python exp_tuned_5fold.py    # needs results_<images>_fair/final_models.json
"""

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from exp_bootstrap import N_DRAWS, macro_f1
from exp_final import final_pipeline
from exp_matrix import HOG_PCA, IMAGE_SETS, OUT_DIR, all_rows, forest
from experiment_common import PROJECT_DIR, TRACKS, build_pipeline
from jute_data import FEATURE_FAMILIES, load_data

SOURCES = ("native_256", "resized")
COMPARISONS = (("crop", "full"), ("insect_only", "full"), ("insect_only", "crop"))


def fit_predict(pipe, X, y, train, test):
    return pipe.fit(X[train], y[train]).predict(X[test])


def out_of_fold(pipe, data, n_jobs):
    """Hold out each fold in turn. Returns one row per distinct photo."""
    X, y, meta = all_rows(data)
    fold, rep = meta["fair_fold"].to_numpy(), meta["fair_rep"].to_numpy()
    splits = [(fold != k, (fold == k) & rep) for k in range(5)]
    preds = Parallel(n_jobs=n_jobs)(delayed(fit_predict)(clone(pipe), X, y, a, b) for a, b in splits)
    return pd.concat([pd.DataFrame({"original": meta.loc[b, "original"].to_numpy(), "source": meta.loc[b, "source"].to_numpy(),
                                    "fold": k, "y": y[b], "pred": p})
                      for k, ((_, b), p) in enumerate(zip(splits, preds))], ignore_index=True)


def fixed_model(images, track):
    data = load_data(FEATURE_FAMILIES, split="fair", combine_train_val=True, subset=track, images=images, fair_all_copies=True)
    return out_of_fold(build_pipeline(data, forest(), hog_pca=HOG_PCA), data, n_jobs=1)


def tuned_model(images, track, cfg):
    data = load_data(cfg["features"], split="fair", combine_train_val=True, subset=track, images=images, fair_all_copies=True)
    pipe = final_pipeline(data, cfg)
    is_forest = cfg["classifier"] == "random_forest"
    if is_forest:  # a forest uses every core itself, so its folds run one at a time
        pipe.set_params(**{f"{pipe.steps[-1][0]}__n_jobs": -1})
    return out_of_fold(pipe, data, n_jobs=1 if is_forest else 5)


def describe(d, draws):
    y, pred = d["y"].to_numpy(), d["pred"].to_numpy()
    boot = np.array([macro_f1(y[i], pred[i]) for i in draws])
    of_fold = [macro_f1(y[d["fold"] == k], pred[d["fold"] == k]) for k in range(5)]
    by_source = {s: macro_f1(y[d["source"] == s], pred[d["source"] == s]) if (d["source"] == s).any() else np.nan for s in SOURCES}
    return {"macro_f1": macro_f1(y, pred), "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
            "fold_min": min(of_fold), "fold_max": max(of_fold), "macro_f1_native_256": by_source["native_256"],
            "macro_f1_resized": by_source["resized"], "n_photos": len(d)}, boot


def difference(label, track, point_a, point_b, boot_a, boot_b):
    diff = boot_a - boot_b
    return {"classes": track, "kind": "difference", "compared": label, "macro_f1": point_a - point_b,
            "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5)}


def main():
    rng = np.random.default_rng(0)  # same draws as exp_bootstrap.py
    rows = []
    for track in TRACKS:
        fixed, tuned, config = {}, {}, {}
        for images in IMAGE_SETS:
            cfgs = [c for c in json.loads((PROJECT_DIR / f"results_{images}_fair" / "final_models.json").read_text()) if c["track"] == track]
            config[images] = max(cfgs, key=lambda c: c["cv_macro_f1"])
            fixed[images] = fixed_model(images, track)
            tuned[images] = {c["classifier"]: (c, tuned_model(images, track, c)) for c in cfgs}
            print(f"{track} {images}: done", flush=True)
        reference = fixed[IMAGE_SETS[0]]
        n = len(reference)
        draws = [rng.integers(0, n, n) for _ in range(N_DRAWS)]

        points, boots = {}, {}
        for images in IMAGE_SETS:
            assert (fixed[images]["original"] == reference["original"]).all()
            summary, boot = describe(fixed[images], draws)
            points[images, "fixed"], boots[images, "fixed"] = summary["macro_f1"], boot
            rows.append({"classes": track, "kind": "score", "images": images, "model": "fixed: random_forest",
                         "features": "hog+hsv+lbp", "hog_pca": HOG_PCA, "chosen_by_cv": False, **summary})
            for classifier, (cfg, d) in tuned[images].items():
                assert (d["original"] == reference["original"]).all()
                summary, boot = describe(d, draws)
                chosen = cfg is config[images]
                if chosen:
                    points[images, "tuned"], boots[images, "tuned"] = summary["macro_f1"], boot
                rows.append({"classes": track, "kind": "score", "images": images, "model": f"tuned: {classifier}",
                             "features": "+".join(cfg["features"]), "hog_pca": cfg["hog_pca"], "chosen_by_cv": chosen, **summary})
            rows.append(difference(f"{images}: tuned minus fixed", track, points[images, "tuned"], points[images, "fixed"],
                                   boots[images, "tuned"], boots[images, "fixed"]))
        for a, b in COMPARISONS:
            rows.append(difference(f"tuned: {a} minus {b}", track, points[a, "tuned"], points[b, "tuned"],
                                   boots[a, "tuned"], boots[b, "tuned"]))

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "fair_tuned_5fold.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 100)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
