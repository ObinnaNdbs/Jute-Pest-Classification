"""One shared way to score every model, so all results are directly comparable.

Every run records the metrics from the proposal (slide 5):
  accuracy, macro-F1 (primary metric), macro precision / recall, weighted F1,
  per-class recall, confusion matrix, fit / predict time, feature count,
  plus accuracy and macro-F1 per image source (native_256 vs resized).

Each call appends one row to results/results.csv and saves the confusion
matrix to results/confusion/<run_id>.csv.

Why per-source matters: native 256x256 images score ~0.9+ macro-F1 while
resized images score ~0.3-0.5, and the splits mix them very differently
(train 84% native, val 32%, test 72%). An overall score mostly reflects that
mix, so always read macro_f1_native_256 and macro_f1_resized alongside it.

Example:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from jute_data import load_data
    from evaluation import run_experiment

    data = load_data(["hsv", "lbp"], split="original")
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    result = run_experiment(model, data, name="rbf_svm_balanced", eval_set="val")
    cv_result = run_cv(model, data, name="rbf_svm_balanced")   # 5-fold grouped CV on train
"""

import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "results"
SOURCES = ("native_256", "resized")

# Chart styling (light surface, recessive ink; sequential blue ramp).
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]})


def evaluate(y_true, y_pred, class_names, *, run_name: str, features, split: str, eval_set: str,
             meta: pd.DataFrame | None = None, fit_seconds: float | None = None,
             predict_seconds: float | None = None, n_features: int | None = None,
             params: str = "", notes: str = "", save: bool = True,
             results_dir: Path = RESULTS_DIR) -> dict:
    """Compute every project metric for one set of predictions.

    meta: the metadata rows matching y_true (e.g. data.meta_val); enables the
          per-source breakdown.
    Returns a dict with the flat metrics plus 'confusion_matrix' (ndarray).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # Macro averages run over the classes present in y_true, so a subset that
    # lacks a class (e.g. Beet Armyworm has no native_256 images) is not
    # penalized with a 0 for it.
    labels = np.unique(y_true)
    features = [features] if isinstance(features, str) else list(features)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = re.sub(r"[^\w+.-]", "_", f"{stamp}_{run_name}_{'+'.join(features)}_{split}_{eval_set}")
    row = {
        "run_id": run_id, "timestamp": stamp, "model": run_name, "features": "+".join(features),
        "split": split, "eval_set": eval_set, "n_eval": len(y_true), "n_features": n_features,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
    }
    if meta is not None:
        source = meta["source"].to_numpy()
        for s in SOURCES:
            m = source == s
            row[f"n_{s}"] = int(m.sum())
            row[f"accuracy_{s}"] = accuracy_score(y_true[m], y_pred[m]) if m.any() else np.nan
            row[f"macro_f1_{s}"] = (f1_score(y_true[m], y_pred[m], labels=np.unique(y_true[m]),
                                             average="macro", zero_division=0) if m.any() else np.nan)
    all_labels = np.arange(len(class_names))
    per_class = recall_score(y_true, y_pred, labels=all_labels, average=None, zero_division=0)
    present = np.isin(all_labels, labels)
    row.update({f"recall[{c}]": (r if p else np.nan) for c, r, p in zip(class_names, per_class, present)})
    row.update({"params": params, "notes": notes})

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    if save:
        (results_dir / "confusion").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            results_dir / "confusion" / f"{run_id}.csv")
        path = results_dir / "results.csv"
        table = pd.DataFrame([row])
        if path.exists():
            table = pd.concat([pd.read_csv(path), table], ignore_index=True)
        table.to_csv(path, index=False)

    print(f"[{run_id}] acc={row['accuracy']:.3f}  macro-F1={row['macro_f1']:.3f}"
          + (f"  (native_256 {row['macro_f1_native_256']:.3f} / resized {row['macro_f1_resized']:.3f})"
             if meta is not None else ""))
    return {**row, "confusion_matrix": cm}


def run_experiment(model, data, *, name: str, eval_set: str = "val", notes: str = "",
                   save: bool = True, results_dir: Path = RESULTS_DIR) -> dict:
    """Fit `model` on data.X_train, predict eval_set ("val" or "test"), and evaluate.

    The model is fitted in place, so it can be inspected afterwards. Keep the
    test set for final runs only; tune on val or with jute_data.cv_folds.
    """
    if eval_set not in ("val", "test"):
        raise ValueError("eval_set must be 'val' or 'test'")
    X_eval, y_eval, meta = getattr(data, f"X_{eval_set}"), getattr(data, f"y_{eval_set}"), getattr(data, f"meta_{eval_set}")
    if X_eval is None:
        raise ValueError("val was merged into train (combine_train_val=True); evaluate on 'test' or use cv_folds")

    t0 = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    t1 = time.perf_counter()
    y_pred = model.predict(X_eval)
    t2 = time.perf_counter()

    return evaluate(y_eval, y_pred, data.class_names, run_name=name, features=data.features,
                    split=data.split, eval_set=eval_set, meta=meta, fit_seconds=t1 - t0,
                    predict_seconds=t2 - t1, n_features=data.n_features,
                    params=" ".join(str(model).split()), notes=notes, save=save,
                    results_dir=results_dir)


def run_cv(model, data, *, name: str, n_splits: int = 5, seed: int = 0, n_jobs: int = -1,
           notes: str = "", save: bool = True, results_dir: Path = RESULTS_DIR) -> dict:
    """Grouped, stratified K-fold CV on the training rows, recorded as eval_set="cv".

    Out-of-fold predictions for every training row are scored with the same
    metrics as val/test (including the per-source breakdown). fit_seconds is
    the total wall time for all folds.
    """
    from sklearn.model_selection import cross_val_predict

    from jute_data import cv_folds

    t0 = time.perf_counter()
    y_pred = cross_val_predict(model, data.X_train, data.y_train,
                               cv=cv_folds(data, n_splits=n_splits, seed=seed), n_jobs=n_jobs)
    elapsed = time.perf_counter() - t0
    return evaluate(data.y_train, y_pred, data.class_names, run_name=name, features=data.features,
                    split=data.split, eval_set="cv", meta=data.meta_train, fit_seconds=elapsed,
                    n_features=data.n_features, params=" ".join(str(model).split()),
                    notes=notes or f"{n_splits}-fold grouped stratified CV, seed {seed}",
                    save=save, results_dir=results_dir)


def load_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """All recorded runs, best macro-F1 first."""
    return pd.read_csv(results_dir / "results.csv").sort_values("macro_f1", ascending=False)


def plot_confusion_matrix(cm, class_names, path=None, title="Confusion matrix", label_min=0.10):
    """Row-normalized confusion matrix (each row = recall breakdown for one true class).

    Only cells >= label_min are labeled so the off-diagonal confusions stand out.
    """
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    mesh = ax.pcolormesh(norm, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    for i in range(n):
        for j in range(n):
            if norm[i, j] >= label_min:
                ax.text(j + 0.5, i + 0.5, f"{norm[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=8, color="#ffffff" if norm[i, j] >= 0.5 else INK)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n) + 0.5, class_names, rotation=60, ha="right", fontsize=9, color=INK_2)
    ax.set_yticks(np.arange(n) + 0.5, class_names, fontsize=9, color=INK_2)
    ax.set_xlabel("Predicted class", color=INK_2)
    ax.set_ylabel("True class", color=INK_2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=13, pad=12)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Share of true class", color=INK_2)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
    return fig
