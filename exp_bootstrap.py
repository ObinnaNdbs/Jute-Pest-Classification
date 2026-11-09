"""How noisy is the fair 5-fold score, and do the image sets really differ?

Uses exp_matrix.py's fixed model (Random Forest, PCA 25 on HOG). Each distinct photo gets one
out-of-fold prediction. Photos are then resampled with replacement 2,000 times, the same
draws for every image set, which gives 95% intervals on each score and on the differences.

Writes results/fair_bootstrap.csv and results/fair_per_class.csv.

Usage:
    python exp_bootstrap.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from exp_matrix import HOG_PCA, IMAGE_SETS, OUT_DIR, all_rows, forest
from experiment_common import TRACKS, build_pipeline
from jute_data import FEATURE_FAMILIES, load_data

N_DRAWS = 2000
COMPARISONS = (("crop", "full"), ("insect_only", "full"), ("insect_only", "crop"))


def macro_f1(y, pred):
    return f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)


def out_of_fold(images, track):
    data = load_data(FEATURE_FAMILIES, split="fair", combine_train_val=True, subset=track, images=images,
                     fair_all_copies=True)
    X, y, meta = all_rows(data)
    fold, rep = meta["fair_fold"].to_numpy(), meta["fair_rep"].to_numpy()
    parts = []
    for k in range(5):
        train, test = fold != k, (fold == k) & rep
        pipe = build_pipeline(data, forest(), hog_pca=HOG_PCA).fit(X[train], y[train])
        parts.append(pd.DataFrame({"original": meta.loc[test, "original"].to_numpy(), "fold": k,
                                   "y": y[test], "pred": pipe.predict(X[test])}))
    return pd.concat(parts, ignore_index=True), data.class_names


def main():
    rng = np.random.default_rng(0)
    rows, per_class = [], []
    for track in TRACKS:
        preds = {}
        for images in IMAGE_SETS:
            preds[images], names = out_of_fold(images, track)
            assert (preds[images]["original"] == preds[IMAGE_SETS[0]]["original"]).all()
        n = len(preds[IMAGE_SETS[0]])
        draws = [rng.integers(0, n, n) for _ in range(N_DRAWS)]

        point, boot = {}, {}
        for images, d in preds.items():
            y, pred = d["y"].to_numpy(), d["pred"].to_numpy()
            point[images] = macro_f1(y, pred)
            boot[images] = np.array([macro_f1(y[i], pred[i]) for i in draws])
            by_fold = [macro_f1(y[d["fold"] == k], pred[d["fold"] == k]) for k in range(5)]
            rows.append({"classes": track, "compared": images, "macro_f1": point[images],
                         "ci_low": np.percentile(boot[images], 2.5), "ci_high": np.percentile(boot[images], 97.5),
                         "fold_min": min(by_fold), "fold_max": max(by_fold), "n_photos": n})
            per_class += [{"classes": track, "images": images, "class": c, "photos": int((y == i).sum()),
                           "f1": f1_score(y == i, pred == i, zero_division=0)} for i, c in enumerate(names)]
        for a, b in COMPARISONS:
            diff = boot[a] - boot[b]
            rows.append({"classes": track, "compared": f"{a} minus {b}", "macro_f1": point[a] - point[b],
                         "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5), "n_photos": n})

    table, classes = pd.DataFrame(rows), pd.DataFrame(per_class)
    table.to_csv(OUT_DIR / "fair_bootstrap.csv", index=False)
    classes.to_csv(OUT_DIR / "fair_per_class.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    wide = classes[classes["classes"] == "all"].pivot(index="class", columns="images", values="f1")[list(IMAGE_SETS)]
    wide["photos"] = classes[(classes["classes"] == "all") & (classes["images"] == "full")].set_index("class")["photos"]
    print(wide.sort_values("full").round(2).to_string())


if __name__ == "__main__":
    main()
