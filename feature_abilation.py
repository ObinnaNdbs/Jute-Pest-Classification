"""
Feature Ablation Experiment Script (Full Dimensions - No PCA) for Logistic Regression
Tests all 7 feature combinations, evaluates on validation, and saves confusion matrices to figures/fa/
"""
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from jute_data import load_data
from evaluation import run_experiment, plot_confusion_matrix
from joblib import Parallel, delayed


def run_ablation_experiment(feats):
    """Helper function to run a single feature combination ablation experiment."""
    feat_label = "_".join(feats)
    run_name = f"logreg_ablation_{feat_label}_full_dims"

    print(f"Loading data for features: {feats}...")
    data = load_data(features=feats, split="original")
    print(f"Loaded: {data}")

    # Pipeline: StandardScaler -> Logistic Regression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    )

    print(f"Running experiment for {run_name}...")
    # Fits on train and evaluates on the validation set
    result = run_experiment(model, data, name=run_name, eval_set="val")

    # Save the confusion matrix inside figures/fa/
    output_dir = "figures/fa"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"cm_logreg_ablation_{feat_label}.png")

    plot_confusion_matrix(result["confusion_matrix"], data.class_names, path=output_path)
    print(f"-> Saved confusion matrix to {output_path}")


def main():
    # Define all 7 possible feature combinations (Experiment 1)
    feature_combinations = [
        ["hog"],
        ["hsv"],
        ["lbp"],
        ["hog", "hsv"],
        ["hog", "lbp"],
        ["hsv", "lbp"],
        ["hog", "hsv", "lbp"]
    ]

    print("--- Starting Feature Ablation Experiments (Full Dimensions, No PCA) ---")

    # Run all 7 combinations concurrently across CPU cores using joblib
    Parallel(n_jobs=-1)(
        delayed(run_ablation_experiment)(feats) for feats in feature_combinations
    )

    print("\nAll feature ablation runs completed and confusion matrices saved to figures/fa/!")


if __name__ == "__main__":
    main()

#Made with the assistance of Google Gemini