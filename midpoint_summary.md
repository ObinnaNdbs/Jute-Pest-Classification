# Jute Pest Classification: What We've Done So Far (Midpoint)

CS 590 Machine Learning, SIUE, Fall 2026. Team: Obinna Ndubuisi and Luke Laurenzana.
Repository: https://github.com/ObinnaNdbs/Jute-Pest-Classification

## Project recap

- **Goal:** classify 17 jute pest species from images using handcrafted features (HOG, HSV histograms, LBP) and classical ML. No neural networks anywhere in the pipeline.
- **Prior work:** published results on this dataset all use deep learning:
  - JutePestDetect (DenseNet201): about 99%
  - Islam & Rahman (2024): transfer learning
  - Kiziloluk et al. (2025): DenseNet/DarkNet features + SVM, 96.779%
- **Dataset:** UCI Jute Pest (ID 920, CC BY 4.0). 7,236 images in 17 classes: train 6,444, val 413, test 379. Training images per class range from 199 (Beet Armyworm) to 676 (Jute Stem Weevil).
- **Primary metric:** macro-F1, because the classes are imbalanced.
- **Planned experiments:**
  1. Feature ablation
  2. Classifier comparison
  3. PCA
  4. Class imbalance (class weights)
  5. Leakage sensitivity (original vs. cleaned split)

---

## Part 1: Obinna's work (data preparation)

### 1. Image audit (`check_images.py`)
- All 7,236 images open, none are corrupt, and all are RGB.
- **Not all the same size:** only 5,823 (80%) were 256×256. The rest come in 215 sizes, from 103 px up to 4592 px.
  - 1,132 are square at other sizes (981 at 640×640, 87 at 416×416, 40 at 300×300).
  - 281 are non-square, mostly 3:2.
- Size is tied to class. All Beet Armyworm images are 640×640 (none are 256×256), and half of Jute Stem Weevil is 640×640.

### 2. Preprocessing to 256×256 (`preprocess_images.py`)
| What was done | Count |
|---|---|
| Already 256×256: copied unchanged | 5,823 |
| Square, other size: resized (Lanczos) | 1,132 |
| Non-square: mirror-padded to square, then resized | 275 |
| Panoramas wider than 2:1: blur-padded, then resized | 6 |

**Why padding instead of the alternatives:**
- **Cropping** cut off pests that fill the frame or sit off-center.
- **Stretching** distorts body shape, which HOG measures.
- **Black or gray bars** add a fake edge for HOG, a fake color for HSV and flat texture for LBP.
- **Mirror padding** extends the real background. On the extreme panoramas it repeated the pest (one photo turned into 5 copies of the mite), so those 6 use a blurred background instead.

Resized images are saved as PNG to avoid extra compression. The output is in `Jute_Pest_256/`.

### 3. Duplicate / leakage audit and cleaned split (`audit_duplicates.py`)
- **Exact duplicates** (identical files): 99 groups. **96 cross splits, and 17 test images have an exact copy in train or val.** This matches our proposal's audit.
- **Near-duplicates:** **3,890 files (54% of the dataset) are augmented copies named `Aug_*`** (flipped, rotated, cropped or color-shifted versions of other images).
  - **Method:** a perceptual hash (pHash) finds candidate pairs. Each pair is then confirmed with ORB keypoint matching plus RANSAC, requiring at least 15 geometrically consistent matches.
  - The stricter threshold is needed because pHash alone falsely matched different insects on white backgrounds, e.g. different crickets.
- **With near-duplicates counted, 78 test images (21%) have a copy in train or val**, not just the 17 exact ones.
- **Label conflicts:** 3 groups (10 images) are the same photo filed under two classes: Jute Aphid vs Yellow Mite, and Black Hairy vs Jute Hairy.
- **Cleaned split** (for experiment 5):
  - Label conflicts are removed entirely.
  - When copies sit in different splits, the copy in test (or else val) is kept and the training copies are dropped, so the test set stays nearly the same.
  - Within one split, one copy per group is kept.

| Split | Original | Cleaned |
|---|---|---|
| Train | 6,444 | 5,828 |
| Val | 413 | 391 |
| Test | 379 | 355 |

- **Limitation:** detection is conservative. Weaker matches (6–14 keypoints) are listed but not removed, so some leakage may remain.

### 4. Feature extraction (`extract_features.py`)
| Feature | Captures | Size per image | Settings |
|---|---|---|---|
| HOG | Shape / edges | 8,100 | Grayscale, 9 orientations, 16×16 px cells, 2×2 blocks |
| HSV histogram | Color | 256 | 16 hue × 4 saturation × 4 value bins |
| LBP | Texture | 54 | Rotation-invariant uniform LBP at 3 scales (radius 1, 2, 3) |

- The color and texture histograms ignore the padded area.
- All features are saved in `features/features.npz`. CSV exports are in `features/csv/`.
- Extraction takes about 20 seconds for all 7,236 images.

### 5. Tools and setup for the modeling half
- `jute_data.py`: loads any feature combination (e.g. HOG+HSV) on either the original or cleaned split in one line. It also makes 5-fold cross-validation folds that keep duplicate groups together.
- `evaluation.py`: scores every model the same way, logs each run to `results/results.csv` and saves its confusion matrix. Each run records:
  - accuracy, macro-F1, macro precision/recall, weighted F1
  - per-class recall
  - training and prediction time, and feature count
  - **scores split by image source** (native 256×256 vs. resized)
- Figures in `figures/`: class distribution, image source by class, one sample per class, feature visualizations (what HOG/LBP/HSV see), preprocessing examples, duplicate examples.
- The README, `requirements.txt` and the GitHub repository contain all data, so no downloads are needed.

### Key finding from the data work
**Image source drives difficulty.**
- In sanity checks, native 256×256 images scored about 0.9 macro-F1 and resized images about 0.3–0.5.
- The splits mix sources very differently (native 256×256 share: train 84%, val 32%, test 72%), so an overall score mostly reflects that mix.
- For that reason, every result is reported per source.

---

## Part 2: Luke's work (first modeling experiments)

All of Luke's runs use the original split, standardized features and `class_weight="balanced"`, scored on the **validation set** with default settings (no tuning yet).

### Experiment 1: Feature ablation (`feature_abilation.py`)
Logistic regression on all 7 feature combinations:

| Features | Dims | Val accuracy | Val macro-F1 | Native 256 macro-F1 | Resized macro-F1 |
|---|---|---|---|---|---|
| HSV + LBP | 310 | 0.571 | **0.469** | 0.899 | 0.328 |
| HOG + HSV + LBP | 8,410 | 0.508 | 0.402 | 0.793 | 0.282 |
| HSV | 256 | 0.482 | 0.401 | 0.925 | 0.246 |
| HOG + HSV | 8,356 | 0.482 | 0.383 | 0.784 | 0.270 |
| HOG + LBP | 8,154 | 0.416 | 0.326 | 0.661 | 0.240 |
| HOG | 8,100 | 0.380 | 0.297 | 0.619 | 0.220 |
| LBP | 54 | 0.332 | 0.256 | 0.481 | 0.197 |

- **Color (HSV) is the strongest single feature**, and texture (LBP) adds to it.
- **Adding HOG lowered logistic regression's scores**, probably because 8,100 dimensions is too many for a linear model on about 6,400 images. PCA (experiment 3) is planned to test this.

### Experiment 2: Classifier comparison (`model_comparison.py`)
Six classifiers on HSV + LBP (310 dims):

| Classifier | Val accuracy | Val macro-F1 | Native 256 macro-F1 | Resized macro-F1 | Fit time (s) |
|---|---|---|---|---|---|
| Random Forest (100 trees) | 0.702 | **0.569** | 0.914 | 0.423 | 0.87 |
| Linear SVM | 0.591 | 0.481 | 0.909 | 0.335 | 2.97 |
| Logistic Regression | 0.571 | 0.469 | 0.899 | 0.328 | 1.64 |
| Decision Tree | 0.552 | 0.468 | 0.923 | 0.337 | 1.75 |
| RBF SVM | 0.530 | 0.426 | 0.841 | 0.295 | 3.71 |
| KNN (k=5) | 0.470 | 0.369 | 0.931 | 0.219 | 0.04 |

- **Random Forest is best**, and most of its edge comes from the harder resized images.
- **Every model scores about 0.84–0.93 on native images but only 0.22–0.42 on resized ones.**
- **Random Forest's weakest classes** (val recall): Jute Semilooper 0.00, Pod Borer 0.00, Scopula Emissaria 0.17. These classes have very few val images, so the numbers are noisy.
- **Most common confusion:** Termite odontotermes (Rambur) predicted as Termite (12 images). Others were Beet Armyworm as Jute Aphid (6) and as Jute Stem Weevil (4).

Confusion matrices for every run are in `figures/fa/` (ablation) and `figures/all_models/` (classifier comparison).

---

## Caveats for these results
- **All scores are preliminary.** They're on the validation set (413 images, 68% resized) with default hyperparameters.
- **HSV + LBP was chosen and compared on the same validation set.** Final numbers will come from tuned models on the test set.
- **The ablation only used logistic regression.** HOG might do better with other classifiers or with PCA.
- **Bug note:** one logged row was lost, due to a write conflict when runs executed in parallel. It was a duplicate configuration, so no result is missing.

## Remaining work (to the final)
- **Experiment 3:** PCA, especially to make HOG usable.
- **Experiment 4:** class weights on vs. off, checking recall on the small classes.
- **Experiment 5:** rerun the final models on the cleaned split to measure the leakage effect.
- **Tuning** with grouped 5-fold cross-validation (macro-F1).
- **Final test-set evaluation** of the best models, reported overall and per source.
- **Error analysis:** confusion matrices, the most-confused pests and why.
