# Jute Pest Classification Without Deep Learning

CS 590 project (SIUE, Fall 2026) by Obinna Ndubuisi and Luke Laurenzaza. The task is to classify 17 jute pest species from images using **handcrafted features (HOG, HSV histograms, LBP) and classical ML only**. No neural networks are used anywhere in the pipeline.

The data preparation is finished: auditing, preprocessing, duplicate detection and feature extraction. This README explains what was done, what was found, and how to run the modeling experiments on top of it.

---

## Setup

```bash
git lfs install                 # once per machine; features.npz is stored with Git LFS
git clone https://github.com/ObinnaNdbs/Jute-Pest-Classification.git
cd Jute-Pest-Classification
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**Everything is already in the repository**, so you can start modeling immediately:
- the original dataset in `Jute_Pest_Dataset/`
- the preprocessed images in `Jute_Pest_256/`
- the extracted features in `features/`
- all audit outputs

You only need to rerun the pipeline if you change a preprocessing or feature setting.

Dataset: UCI Machine Learning Repository, "Jute Pest" (ID 920, DOI 10.24432/C5289P), licensed CC BY 4.0.

## Pipeline

These are the scripts that produced the data in the repository. Run them in this order to regenerate it; each one reads the previous step's output. On a 24-core machine the whole run takes about 7 minutes.

| # | Script | What it does | Output |
|---|---|---|---|
| 1 | `check_images.py` | Audits sizes, color modes and formats, and checks for corrupt files | `image_audit.csv` |
| 2 | `preprocess_images.py` | Converts every image to 256×256 RGB | `Jute_Pest_256/`, `preprocess_manifest.csv` |
| 3 | `audit_duplicates.py` | Finds exact and near-duplicates and defines the cleaned split | `duplicate_audit.csv`, `duplicate_pairs.csv` |
| 4 | `extract_features.py` | Extracts HOG, HSV and LBP features (about 20 s) | `features/` |
| 5 | `make_figures.py` | Makes report figures | `figures/` |

Modeling code uses two helper modules:

- **`jute_data.py`** loads any feature combination for either split scheme and builds cross-validation folds.
- **`evaluation.py`** scores every model the same way and logs each run to `results/`.

`features/features.npz` (233 MB) is stored with **Git LFS**, because GitHub rejects normal files over 100 MB. If it shows up as a tiny text file after cloning, run `git lfs pull`.

---

## What we found in the data

| Finding | Numbers | Consequence |
|---|---|---|
| **Image sizes vary** | 5,823 of 7,236 images (80%) are 256×256. The rest come in 215 sizes, from 103 px to 4592 px. | Every image is standardized to 256×256 (see below). |
| **Exact duplicates** | 99 groups of byte-identical files. 96 of them cross splits, and 17 test images have an exact copy in train/val. | Matches our proposal's audit. |
| **Near-duplicates** | The dataset includes **3,890 augmented files (`Aug_*`, 54%)**: flipped, rotated, cropped and color-shifted copies. With these counted, **78 test images (21%)** have a copy in train or val. | Leakage is much larger than exact matching shows. This is why the cleaned split exists. |
| **Label conflicts** | 3 groups (10 images) contain the same photo under two classes: Jute Aphid vs Yellow Mite, and Black Hairy vs Jute Hairy. | Those images are excluded from the cleaned split. |
| **Image source predicts difficulty** | Native 256×256 images score about 0.9+ macro-F1; resized images score about 0.3–0.5. | **Always report results per source** (see Evaluation protocol). |
| **Source mix differs by split** | Native 256×256 share: train 84%, val 32%, test 72%. | A model's overall score mostly reflects this mix. Val will always look worse than CV. |
| **Source is confounded with class** | All Beet Armyworm images are resized; Yellow Mite and one termite class are 100% native. | A model can partly learn the image source instead of the pest. Mention this in the discussion. |

Figures for the report are in `figures/`:

- `class_distribution.png`: images per class, by split
- `source_by_class.png`: share of native vs. resized images in each class
- `sample_grid.png`: one example image per class
- `feature_visualization.png`: what HOG, LBP and the hue histogram capture
- `preprocess_examples.png`: before/after examples of padding
- `duplicate_examples.png`: near-duplicate pairs, including the label conflicts

## Preprocessing (`preprocess_images.py`)

- **Native 256×256 JPEGs** (5,823) are copied byte-for-byte, so they are completely unchanged.
- **Square images of other sizes** (1,132, mostly 640×640) are resized with Lanczos resampling.
- **Non-square images** (281, mostly 3:2) are **mirror-padded to square** and then resized. Other options were rejected:
  - Center-cropping cut off pests that fill the frame or sit off-center.
  - Stretching distorts body shape, which is exactly what HOG measures.
  - Black or gray padding adds a fake edge for HOG, a fake color for the HSV histogram and flat texture for LBP.
- **Panoramas wider than 2:1** (6 images) use **blur-fill padding** instead, because mirroring them repeats the pest several times.
- Resized images are saved as PNG to avoid a second round of JPEG compression. `preprocess_manifest.csv` records what was done to each image, and where its real (unpadded) content sits.

## Duplicate detection (`audit_duplicates.py`)

1. **Exact:** MD5 hash of each original file.
2. **Near-duplicate candidates:** a perceptual hash (pHash) of each image, also compared against flipped versions.
3. **Verification:** ORB keypoint matching plus RANSAC, which fits one affine transform (shift, scale, rotation or crop) between the two images.
   - **15 or more consistent matches** count as a confirmed near-duplicate. Every pair inspected at this level was a real copy.
   - **6–14 matches** are listed in `duplicate_pairs.csv` as `possible`, but not acted on. That range mixes real augmented copies with *different* insects photographed alike (e.g. crickets on white), and no pixel-level check separated them reliably.
4. Confirmed links are merged into groups, stored in the `dup_group` column.

The **cleaned split** (`in_clean_split`) applies three rules:

- Label-conflict groups are removed entirely.
- In a group spanning several splits, only the copies in the most evaluation-critical split are kept (test over val over train). The test set therefore stays almost identical (379 → 355 images), so an original-vs-clean difference isolates the effect of training on leaked copies.
- Within a split, one copy per group is kept.

| | Original | Clean |
|---|---|---|
| train | 6,444 | 5,828 |
| val | 413 | 391 |
| test | 379 | 355 |

The cleaned split is **conservative**. It removes only the leakage we are sure of, so the measured leakage effect is a lower bound.

## Features (`extract_features.py`)

| Family | Captures | Dims | Settings |
|---|---|---|---|
| **HOG** | Shape: edges, outline, legs, orientation | 8,100 | Grayscale, 9 orientations, 16×16 px cells, 2×2 blocks, L2-Hys normalization |
| **HSV** | Color distribution | 256 | Joint histogram: 16 hue × 4 saturation × 4 value bins. Hue gets the most bins because the pests differ mainly in hue; few value bins reduce sensitivity to brightness. |
| **LBP** | Texture: hair, shell, wing surface | 54 | Rotation-invariant uniform LBP at three scales: (8 points, radius 1), (16, 2), (24, 3) |

- The HSV and LBP histograms are counted **only inside the real-content box**, so padding doesn't skew them.
- Features are saved **unscaled**.
- All settings are recorded in `features/params.json`. To try other settings (e.g. HOG cell size), edit `PARAMS` in `extract_features.py` and rerun it.

---

## Running experiments (for the modeling half)

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from jute_data import load_data, cv_folds
from evaluation import run_experiment, run_cv, load_results, plot_confusion_matrix

# Any feature combination, either split scheme
data = load_data(features=["hog", "hsv"], split="original")   # or split="clean"
print(data)   # JuteData(features=['hog', 'hsv'], split='original', dims=8356, train=6444, val=413, test=379)

# Scaling and PCA go INSIDE the pipeline, so they are fit on training data only
model = make_pipeline(StandardScaler(), PCA(n_components=0.95),
                      SVC(kernel="rbf", class_weight="balanced"))

run_cv(model, data, name="rbf_svm_pca")                          # 5-fold grouped CV on train
run_experiment(model, data, name="rbf_svm_pca", eval_set="val")  # fit on train, score val

# Hyperparameter search: use the grouped folds and macro-F1
search = GridSearchCV(model, {"svc__C": [1, 10, 100]}, cv=cv_folds(data),
                      scoring="f1_macro", n_jobs=-1)
search.fit(data.X_train, data.y_train)

# FINAL runs only: score the test set
result = run_experiment(search.best_estimator_, data, name="rbf_svm_final", eval_set="test")
plot_confusion_matrix(result["confusion_matrix"], data.class_names, path="figures/cm_rbf_svm.png")

load_results().head(10)   # every run so far, best macro-F1 first
```

Other `load_data` options:

- `combine_train_val=True` merges val into train, for example to run CV over both.
- `data.meta_train`, `data.meta_val` and `data.meta_test` hold the per-row metadata (class, source, file paths) for error analysis.
- `data.feature_slices` tells you which columns of X belong to which feature family.

### What every run records (`results/results.csv`)

| Column | Meaning |
|---|---|
| `macro_f1` | **Primary metric.** F1 averaged equally over classes, so small classes count as much as big ones. |
| `accuracy`, `macro_precision`, `macro_recall`, `weighted_f1` | Supporting metrics. `macro_recall` is the same as balanced accuracy. |
| `macro_f1_native_256`, `macro_f1_resized` (and `accuracy_*`, `n_*`) | The same scores computed per image source. **Read these alongside the overall score.** |
| `recall[<class>]` | Per-class recall, showing which pests get missed |
| `fit_seconds`, `predict_seconds`, `n_features` | Runtime and feature-size tradeoff |
| `model`, `features`, `split`, `eval_set`, `params`, `notes` | What was run |

Each run's confusion matrix is saved to `results/confusion/<run_id>.csv`.

### Evaluation protocol

1. **Scaler and PCA live inside the pipeline.** Never fit them on the full dataset.
2. **Tune with `run_cv` / `GridSearchCV(cv=cv_folds(data))`**, optimizing `f1_macro`. The folds never split a duplicate group. The supplied val split is also available, but it is small (413 images) and 68% resized, so treat it as a second opinion rather than the main tuning signal.
3. **Touch the test set only for final runs.** Use one test evaluation per final model.
4. **Report overall *and* per-source scores.** CV scores look high (about 0.88 for a simple baseline) because train is 84% native images. Val looks low (about 0.40) because it is mostly resized. Neither number alone is "the" accuracy.
5. **Run final models on both `split="original"` and `split="clean"`.** The difference is the leakage effect (slide 5, experiment 5).

### Experiment checklist (from the proposal, slide 5)

| # | Experiment | How |
|---|---|---|
| 1 | Feature ablation | Same model with `features=` set to `["hog"]`, `["hsv"]`, `["lbp"]`, pairs, and all three |
| 2 | Classifier comparison | Logistic Regression, KNN, Linear SVM, RBF SVM, Decision Tree, Random Forest on the best feature set(s) |
| 3 | Dimensionality reduction | Add `PCA(n_components=...)` to the pipeline; compare `macro_f1` vs `n_features` / `fit_seconds`. Most useful with HOG's 8,100 dims, especially for KNN. |
| 4 | Class imbalance | `class_weight="balanced"` vs. the default; compare `macro_f1` and the `recall[...]` columns for small classes (Beet Armyworm, Jute Stem Girdler) |
| 5 | Leakage sensitivity | Rerun the final models with `split="clean"` and compare to `split="original"` |

A quick sanity baseline (StandardScaler + Logistic Regression on HSV+LBP, clean split) scores **0.877 macro-F1 in CV** (native 0.967 / resized 0.311) and **0.396 on val** (native 0.854 / resized 0.247). Proper tuned models should beat this.

## Known limitations

- Near-duplicate detection favors precision over recall. Some heavily augmented copies (6–14 matches) remain in the cleaned split, so the leakage effect is underestimated.
- Image source is confounded with class (e.g. Beet Armyworm). Per-source reporting exposes this but can't remove it.
- The label conflicts suggest the original dataset contains some labeling errors beyond the 3 groups we could detect.
