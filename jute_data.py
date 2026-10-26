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

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_DIR = Path(__file__).parent
FEATURE_DIR = PROJECT_DIR / "features"
FEATURE_FAMILIES = ("hog", "hsv", "lbp")
SPLIT_SCHEMES = ("original", "clean")


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
        """Duplicate-group id per training row, for grouped cross-validation."""
        return self.meta_train["dup_group"].to_numpy()

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def __repr__(self):
        val = "merged into train" if self.X_val is None else len(self.y_val)
        return (f"JuteData(features={self.features}, split={self.split!r}, dims={self.n_features}, "
                f"train={len(self.y_train)}, val={val}, test={len(self.y_test)})")


@lru_cache(maxsize=1)
def _load_all():
    arrays = np.load(FEATURE_DIR / "features.npz")
    feats = {name: arrays[name] for name in FEATURE_FAMILIES}
    meta = pd.read_csv(FEATURE_DIR / "meta.csv", keep_default_na=False)
    for col in ["has_exact_dup", "cross_split_dup", "label_conflict", "in_clean_split"]:
        meta[col] = meta[col].astype(str) == "True"
    assert len(meta) == len(arrays["label"]) and (meta["label"].to_numpy() == arrays["label"]).all(), \
        "features.npz and meta.csv are out of sync; rerun extract_features.py"
    class_names = meta.drop_duplicates("label").sort_values("label")["class"].tolist()
    return feats, meta, class_names


def load_data(features=FEATURE_FAMILIES, split: str = "original", combine_train_val: bool = False) -> JuteData:
    """Return train/val/test arrays for a combination of feature families.

    features:           any non-empty subset of ("hog", "hsv", "lbp"), in any order.
    split:              "original" - the supplied train/val/test folders as-is.
                        "clean"    - duplicates and label conflicts removed (see
                                     audit_duplicates.py); test stays almost unchanged.
    combine_train_val:  merge val into train (e.g. when tuning with cross-validation
                        instead of the supplied val split); X_val/y_val become None.
    """
    features = [features] if isinstance(features, str) else list(features)
    unknown = set(features) - set(FEATURE_FAMILIES)
    if not features or unknown:
        raise ValueError(f"features must be a non-empty subset of {FEATURE_FAMILIES}, got {features}")
    if split not in SPLIT_SCHEMES:
        raise ValueError(f"split must be one of {SPLIT_SCHEMES}, got {split!r}")

    feats, meta, class_names = _load_all()
    X = np.hstack([feats[f] for f in features])
    y = meta["label"].to_numpy()

    slices, start = {}, 0
    for f in features:
        slices[f] = slice(start, start + feats[f].shape[1])
        start += feats[f].shape[1]

    keep = np.ones(len(meta), dtype=bool) if split == "original" else meta["in_clean_split"].to_numpy()
    train_splits = ["train", "val"] if combine_train_val else ["train"]

    def part(splits):
        mask = keep & meta["split"].isin(splits).to_numpy()
        return X[mask], y[mask], meta[mask].reset_index(drop=True)

    X_tr, y_tr, m_tr = part(train_splits)
    X_va, y_va, m_va = (None, None, None) if combine_train_val else part(["val"])
    X_te, y_te, m_te = part(["test"])
    return JuteData(features, split, X_tr, y_tr, X_va, y_va, X_te, y_te, m_tr, m_va, m_te,
                    class_names, slices)


def cv_folds(data: JuteData, n_splits: int = 5, seed: int = 0):
    """Stratified K-fold over the training rows that never splits a duplicate group.

    Returns a list of (train_idx, val_idx) index arrays into data.X_train. Pass it
    as `cv=` to GridSearchCV / cross_val_score, or loop over it yourself.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(data.X_train, data.y_train, groups=data.groups_train))
