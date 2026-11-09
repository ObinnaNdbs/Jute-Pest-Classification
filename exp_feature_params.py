"""Step 1: compare HOG, HSV and LBP extraction settings before running the big grid.

One family at a time, on all 17 classes (train+val only; test stays untouched), with
grouped 5-fold CV. Native-only data is too close to the ceiling to rank settings.
Each setting is scored with Random Forest and logistic regression. HOG goes through
PCA(200) first for logistic regression, because cell size 8 gives up to 46,128 dims.

A setting counts as better only if its mean macro-F1 (over both models) beats the
current one by more than the current setting's fold-to-fold standard deviation.

Writes results/feature_params.csv and prints the recommendation; changing PARAMS in
extract_features.py is a separate, deliberate step.

Usage:
    python exp_feature_params.py
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from skimage.util import img_as_ubyte
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import RESULTS_DIR
from experiment_common import N_WORKERS, PROJECT_DIR, SEED, hog_pca_step
from extract_features import PARAMS, normalized, unpack_masks
from jute_data import IMAGES, SPLIT, cv_folds, load_data

HOG_SETTINGS = [{"cell": c, "orientations": o} for c in (8, 16, 32) for o in (9, 12)]
HSV_SETTINGS = [{"bins": b} for b in ((8, 4, 4), (16, 4, 4), (16, 8, 8), (32, 8, 8))]
LBP_SETTINGS = [{"scales": s} for s in (((8, 1),), ((16, 2),), ((24, 3),), ((8, 1), (16, 2), (24, 3)))]
CURRENT = {"hog": {"cell": PARAMS["hog"]["pixels_per_cell"][0], "orientations": PARAMS["hog"]["orientations"]},
           "hsv": {"bins": tuple(PARAMS["hsv"]["bins"])},
           "lbp": {"scales": tuple(tuple(s) for s in PARAMS["lbp"]["scales"])}}


def read_image(row):
    """Returns the RGB image and a boolean mask of the pixels the histograms may count."""
    with Image.open(PROJECT_DIR / row["processed"]) as img:
        rgb = np.asarray(img.convert("RGB"))
    if "mask" in row:
        return rgb, unpack_masks(row["mask"][None])[0]
    counted = np.zeros(rgb.shape[:2], dtype=bool)
    counted[int(row["box_top"]):int(row["box_bottom"]), int(row["box_left"]):int(row["box_right"])] = True
    return rgb, counted


def extract(row, family, cfg):
    rgb, counted = read_image(row)
    if family == "hog":
        return hog(rgb2gray(rgb), orientations=cfg["orientations"], pixels_per_cell=(cfg["cell"], cfg["cell"]),
                   cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True).astype(np.float32)
    if family == "hsv":
        pixels = rgb2hsv(rgb[counted].reshape(1, -1, 3)).reshape(-1, 3)
        hist, _ = np.histogramdd(pixels, bins=cfg["bins"], range=[(0, 1)] * 3)
        return normalized(hist.ravel()).astype(np.float32)
    gray = img_as_ubyte(rgb2gray(rgb))
    hists = []
    for n_points, radius in cfg["scales"]:
        codes = local_binary_pattern(gray, n_points, radius, method="uniform")[counted]
        hists.append(normalized(np.bincount(codes.astype(np.int64).ravel(), minlength=n_points + 2)))
    return np.concatenate(hists).astype(np.float32)


def models(family):
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    return {"random_forest": rf,
            "log_reg": make_pipeline(hog_pca_step(200), lr) if family == "hog" else make_pipeline(StandardScaler(), lr)}


def score_fold(model, X, y, source, train_idx, test_idx):
    pred = clone(model).fit(X[train_idx], y[train_idx]).predict(X[test_idx])
    truth, src = y[test_idx], source[test_idx]
    macro = f1_score(truth, pred, average="macro", labels=np.unique(truth), zero_division=0)
    resized = src == "resized"
    return macro, f1_score(truth[resized], pred[resized], average="macro", labels=np.unique(truth[resized]),
                           zero_division=0)


def main():
    data = load_data(["lbp"], split=SPLIT, combine_train_val=True)
    if IMAGES == "full":
        boxes = pd.read_csv(PROJECT_DIR / "preprocess_manifest.csv")[
            ["processed", "box_left", "box_top", "box_right", "box_bottom"]]
        rows = data.meta_train[["processed"]].merge(boxes, on="processed", how="left", validate="one_to_one")
        assert rows["box_left"].notna().all() and (rows["processed"] == data.meta_train["processed"]).all()
    else:  # crops fill the whole frame
        rows = data.meta_train[["processed"]].assign(box_left=0, box_top=0, box_right=256, box_bottom=256)
    rows = rows.to_dict("records")
    if IMAGES == "insect_only":  # histograms count only the insect, as in extract_features.py
        located = pd.read_csv(PROJECT_DIR / "insect_boxes.csv", keep_default_na=False)["original"]
        row_of = pd.Series(np.arange(len(located)), index=located)[data.meta_train["original"]].to_numpy()
        packed = np.load(PROJECT_DIR / "insect_masks.npz")["crop_mask"]
        for row, i in zip(rows, row_of):
            row["mask"] = packed[i]
    y, source, folds = data.y_train, data.meta_train["source"].to_numpy(), cv_folds(data)

    records = []
    for family, settings in (("lbp", LBP_SETTINGS), ("hsv", HSV_SETTINGS), ("hog", HOG_SETTINGS)):
        for cfg in settings:
            X = np.stack(Parallel(n_jobs=-1, batch_size=32)(delayed(extract)(r, family, cfg) for r in rows))
            for name, model in models(family).items():
                scores = Parallel(n_jobs=min(5, N_WORKERS))(
                    delayed(score_fold)(model, X, y, source, a, b) for a, b in folds)
                macro, resized = np.array(scores).T
                records.append({"family": family, "setting": str(cfg), "dims": X.shape[1], "model": name,
                                "cv_macro_f1": macro.mean(), "cv_macro_f1_std": macro.std(),
                                "cv_macro_f1_resized": resized.mean(), "current": cfg == CURRENT[family]})
                print(f"{family} {cfg} {name}: {macro.mean():.3f} +/- {macro.std():.3f}"
                      f"  resized {resized.mean():.3f}  ({X.shape[1]} dims)", flush=True)
            del X

    table = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "feature_params.csv", index=False)

    print("\nRecommendation (mean over both models):")
    for family, group in table.groupby("family"):
        means = group.groupby("setting")["cv_macro_f1"].mean().sort_values(ascending=False)
        base = group[group["current"]]
        base_mean, noise = base["cv_macro_f1"].mean(), base["cv_macro_f1_std"].mean()
        best, best_mean = means.index[0], means.iloc[0]
        verdict = "CHANGE" if best_mean - base_mean > noise else "keep current"
        print(f"  {family}: best {best} = {best_mean:.3f}; current = {base_mean:.3f} (noise {noise:.3f}) -> {verdict}")


if __name__ == "__main__":
    main()
