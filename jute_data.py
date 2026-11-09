"""Load extracted features for modeling.

Example:
    from jute_data import load_data, cv_folds

    data = load_data(features=["hog", "hsv"], split="clean")
    data.X_train.shape, data.y_train.shape     # numpy arrays
    data.meta_test["source"]                   # per-row metadata as DataFrames
    for train_idx, val_idx in cv_folds(data):  # grouped, stratified 5-fold CV on train
        ...

Features are returned UNSCALED. Put StandardScaler / PCA inside an sklearn
Pipeline so they are fit on training folds only; scaling all the data up front
leaks test statistics into training.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_DIR = Path(__file__).parent
FEATURE_DIR = PROJECT_DIR / "features"
FEATURE_FAMILIES = ("hog", "hsv", "lbp")
SPLIT_SCHEMES = ("original", "clean", "fair")
SUBSETS = ("all", "native12")
FAIR_FILE = PROJECT_DIR / "fair_split.csv"
# Which images and split the experiment scripts use; set these to rerun them unchanged.
# Images: full, crop, insect_only or background (see extract_features.py).
IMAGES = os.environ.get("JUTE_IMAGES", "full")
SPLIT = os.environ.get("JUTE_SPLIT", "original")
# A class joins the native12 subset if it has enough native 256x256 images to
# train on and to score on.
NATIVE_MIN_TRAIN, NATIVE_MIN_TEST = 250, 15


@dataclass
class JuteData:
    features: list[str]
    split: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray | None
    y_val: np.ndarray | None
    X_test: np.ndarray
    y_test: np.ndarray
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame | None
    meta_test: pd.DataFrame
    class_names: list[str]
    feature_slices: dict[str, slice]  # which columns of X belong to which family

    @property
    def groups_train(self) -> np.ndarray:
        """Group id per training row for grouped CV: photo family on the fair split, else duplicate group."""
        return self.meta_train["family" if self.split.startswith("fair") else "dup_group"].to_numpy()

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def __repr__(self):
        val = "merged into train" if self.X_val is None else len(self.y_val)
        return (f"JuteData(features={self.features}, split={self.split!r}, dims={self.n_features}, "
                f"train={len(self.y_train)}, val={val}, test={len(self.y_test)})")


def feature_dir(images: str) -> Path:
    return FEATURE_DIR if images == "full" else FEATURE_DIR / images


@lru_cache(maxsize=6)
def _load_all(images: str = "full"):
    arrays = np.load(feature_dir(images) / "features.npz")
    feats = {name: arrays[name] for name in FEATURE_FAMILIES}
    meta = pd.read_csv(feature_dir(images) / "meta.csv", keep_default_na=False)
    if FAIR_FILE.exists():
        meta = meta.merge(pd.read_csv(FAIR_FILE, keep_default_na=False), on="original", how="left", validate="one_to_one")
        meta["fair_rep"] = meta["fair_rep"].astype(str) == "True"
    for col in ["has_exact_dup", "cross_split_dup", "label_conflict", "in_clean_split"]:
        meta[col] = meta[col].astype(str) == "True"
    assert len(meta) == len(arrays["label"]) and (meta["label"].to_numpy() == arrays["label"]).all(), \
        "features.npz and meta.csv are out of sync; rerun extract_features.py"
    class_names = meta.drop_duplicates("label").sort_values("label")["class"].tolist()
    return feats, meta, class_names


def native_classes(images: str = "full") -> list[str]:
    """Classes with at least NATIVE_MIN_TRAIN native train and NATIVE_MIN_TEST native test images.

    Counted on the original split for every split scheme, so the same 12 classes are
    compared. On the fair split no class has that many distinct native test photos.
    """
    _, meta, class_names = _load_all(images)
    native = meta[meta["source"] == "native_256"]
    train = native[native["split"] == "train"]["class"].value_counts()
    test = native[native["split"] == "test"]["class"].value_counts()
    return [c for c in class_names
            if train.get(c, 0) >= NATIVE_MIN_TRAIN and test.get(c, 0) >= NATIVE_MIN_TEST]


def load_data(features=FEATURE_FAMILIES, split: str = "original", combine_train_val: bool = False,
              subset: str = "all", images: str | None = None, fair_all_copies: bool = False) -> JuteData:
    """Return train/val/test arrays for a combination of feature families.

    features:           any non-empty subset of ("hog", "hsv", "lbp"), in any order.
    split:              "original" - the supplied train/val/test folders as-is.
                        "clean"    - duplicates and label conflicts removed (see
                                     audit_duplicates.py); test stays almost unchanged.
                        "fair"     - every copy of a photo stays on one side (see
                                     make_fair_split.py). No val set; use cv_folds. Test has
                                     one image per photo unless fair_all_copies=True.
    combine_train_val:  merge val into train (e.g. when tuning with cross-validation
                        instead of the supplied val split); X_val/y_val become None.
    subset:             "all"      - all 17 classes and every image.
                        "native12" - native 256x256 images only, from the classes in
                                     native_classes(). Labels are renumbered 0-11 and the
                                     returned .split reads e.g. "original_native12". Supplied
                                     val covers few of these classes, so use
                                     combine_train_val=True and cv_folds.
    images:             which image set's features to load; defaults to JUTE_IMAGES.
    """
    images = IMAGES if images is None else images
    features = [features] if isinstance(features, str) else list(features)
    unknown = set(features) - set(FEATURE_FAMILIES)
    if not features or unknown:
        raise ValueError(f"features must be a non-empty subset of {FEATURE_FAMILIES}, got {features}")
    if split not in SPLIT_SCHEMES:
        raise ValueError(f"split must be one of {SPLIT_SCHEMES}, got {split!r}")
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")

    feats, meta, class_names = _load_all(images)
    if split == "fair" and "fair_split" not in meta:
        raise FileNotFoundError("fair_split.csv is missing; run make_fair_split.py")
    X = np.hstack([feats[f] for f in features])
    y = meta["label"].to_numpy()

    slices, start = {}, 0
    for f in features:
        slices[f] = slice(start, start + feats[f].shape[1])
        start += feats[f].shape[1]

    if split == "original":
        keep = np.ones(len(meta), dtype=bool)
    elif split == "clean":
        keep = meta["in_clean_split"].to_numpy()
    else:
        keep = (meta["fair_split"] != "dropped").to_numpy()
    split_name = split
    if subset == "native12":
        kept_classes = native_classes(images=images)
        keep = keep & (meta["source"] == "native_256").to_numpy() & meta["class"].isin(kept_classes).to_numpy()
        y = meta["class"].map({c: i for i, c in enumerate(kept_classes)}).fillna(-1).to_numpy(dtype=int)
        class_names, split_name = kept_classes, f"{split}_{subset}"
    column = "fair_split" if split == "fair" else "split"
    train_splits = ["train", "val"] if combine_train_val else ["train"]

    def part(splits, only=None):
        mask = keep & meta[column].isin(splits).to_numpy()
        if only is not None:
            mask = mask & only
        return X[mask], y[mask], meta[mask].reset_index(drop=True)

    X_tr, y_tr, m_tr = part(train_splits)
    X_va, y_va, m_va = (None, None, None) if combine_train_val or split == "fair" else part(["val"])
    X_te, y_te, m_te = part(["test"], meta["fair_rep"].to_numpy() if split == "fair" and not fair_all_copies else None)
    return JuteData(features, split_name, X_tr, y_tr, X_va, y_va, X_te, y_te, m_tr, m_va, m_te,
                    class_names, slices)


def cv_folds(data: JuteData, n_splits: int = 5, seed: int = 0):
    """Stratified K-fold over the training rows that never splits a duplicate group.

    Returns a list of (train_idx, val_idx) index arrays into data.X_train. Pass it
    as `cv=` to GridSearchCV / cross_val_score, or loop over it yourself.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(data.X_train, data.y_train, groups=data.groups_train))
