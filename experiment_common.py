"""Shared pieces for the exp_*.py scripts: feature mixes, classifiers, pipelines, chart style."""

from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import RESULTS_DIR
from jute_data import IMAGES, SPLIT

PROJECT_DIR = Path(__file__).parent
FIGURE_DIR = PROJECT_DIR / "figures" / ("final" if (IMAGES, SPLIT) == ("full", "original") else f"final_{IMAGES}_{SPLIT}")
SEED = 42
# HOG runs hold several GB per worker, so this stays well under the 24 cores.
N_WORKERS = 8
# Per class pair. Without a limit, two linear SVM runs did not finish in over an hour
# (see exp_svm_convergence.py).
LINEAR_SVM_MAX_ITER = 1_000_000

FEATURE_MIXES = [["hog"], ["hsv"], ["lbp"], ["hog", "hsv"], ["hog", "lbp"], ["hsv", "lbp"], ["hog", "hsv", "lbp"]]
TRACKS = ("all", "native12")
PCA_CHOICE_FILE = RESULTS_DIR / "pca_choice.txt"


def mix_name(features) -> str:
    return "+".join(features)


def make_classifiers(class_weight="balanced") -> dict:
    """The six classifiers, with the same default settings as model_comparison.py."""
    return {
        "log_reg": LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=SEED),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "linear_svm": SVC(kernel="linear", class_weight=class_weight, max_iter=LINEAR_SVM_MAX_ITER, random_state=SEED),
        "rbf_svm": SVC(kernel="rbf", class_weight=class_weight, random_state=SEED),
        "decision_tree": DecisionTreeClassifier(class_weight=class_weight, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=SEED),
    }


def hog_pca_step(n_components: int):
    """Standardize, then PCA. Whitened so each kept component has unit variance like the other columns."""
    return make_pipeline(StandardScaler(),
                         PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=SEED))


def build_pipeline(data, clf, hog_pca: int | None = None):
    """StandardScaler -> classifier, with PCA on the HOG columns only when hog_pca is set."""
    if hog_pca is None or "hog" not in data.features:
        return make_pipeline(StandardScaler(), clf)
    hog = data.feature_slices["hog"]
    hog_cols = np.arange(hog.start, hog.stop)
    rest_cols = np.setdiff1d(np.arange(data.n_features), hog_cols)
    steps = [("hog", hog_pca_step(hog_pca), hog_cols)]
    if len(rest_cols):
        steps.append(("rest", StandardScaler(), rest_cols))
    return make_pipeline(ColumnTransformer(steps), clf)
