"""
Classifier Comparison Script for HSV + LBP Features (Full Dimensions - No PCA) (CS 590)
Evaluates all primary classifiers on HSV + LBP features and saves confusion matrices to figures/all_models/
"""
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from jute_data import load_data
from evaluation import run_experiment, plot_confusion_matrix
from joblib import Parallel, delayed


def evaluate_classifier(name, clf, data):
    """Helper function to run a single classifier on HSV+LBP features and save its confusion matrix."""
    run_name = f"{name}_hsv_lbp_full_dims"
    print(f"Running experiment for {name} on HSV + LBP features...")

    # Pipeline: StandardScaler -> Classifier
    model = make_pipeline(
        StandardScaler(),
        clf
    )

    # Fits on train and evaluates on the validation set
    result = run_experiment(model, data, name=run_name, eval_set="val")

    # Save the confusion matrix inside figures/all_models/
    output_dir = "figures/all_models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cm_{name}_hsv_lbp.png")

    plot_confusion_matrix(result["confusion_matrix"], data.class_names, path=output_path)
    print(f"-> Saved confusion matrix for {name} to {output_path}")


def main():
    # 1. Load data with HSV and LBP features combined (310 total dimensions)
    print("Loading data for HSV + LBP features...")
    data = load_data(features=["hsv", "lbp"], split="original")
    print(f"Loaded: {data}")

    # 2. Define all primary classifiers to test
    classifiers = {
        "log_reg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "linear_svm": SVC(kernel="linear", class_weight="balanced", random_state=42),
        "rbf_svm": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        "decision_tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
    }

    print("\n--- Starting HSV + LBP Classifier Comparison (Parallel) ---")

    # 3. Run all classifiers concurrently across CPU cores using joblib
    Parallel(n_jobs=-1)(
        delayed(evaluate_classifier)(name, clf, data)
        for name, clf in classifiers.items()
    )

    print("\nAll models evaluated! Confusion matrices saved to figures/all_models/.")


if __name__ == "__main__":
    main()


#Made with the assistance of Google Gemini