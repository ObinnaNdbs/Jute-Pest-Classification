# Jute Pest Classification: Full Experiment Results

CS 590 Machine Learning, SIUE, Fall 2026. Team: Obinna Ndubuisi and Luke Laurenzana.
Follows on from `midpoint_summary.md`. All numbers below use the updated features from section 1, so they differ from the midpoint tables.

## Main takeaways

1. **The fair score is about 0.49, not 0.91.** Sections 1 to 9 use the supplied split, where most test images have another version of the same photo in training. With every copy of a photo kept on one side, 17 classes score **0.490 macro-F1** and the 12 native classes score **0.577** (section 10). The model does learn something, but much less than 0.910 and 1.000 suggest.
2. **Dropping to 12 classes of native images gives a perfect-looking score (1.000 test macro-F1), but it doesn't carry over.** The same model scores **0.242** on resized photos of those same pests.
3. **Best result on the supplied split:** Random Forest on HOG + HSV + LBP (PCA on HOG), 17 classes: **0.910 test macro-F1, or 0.864 with known duplicates removed.** On resized images it gets 0.711, or 0.544 with known duplicates removed.
4. **Color does most of the work.** HSV alone nearly matches the best mix. PCA makes HOG usable but HOG adds little on top of color.
5. **Pests don't group by insect type in these features.** Beetles do not cluster with beetles.

**How to read the scores:** macro-F1 (0 to 1) averages over classes equally. "Native" means the image was already 256×256. "Resized" means it was a different size and we resized it. "Grouped CV" is 5-fold cross-validation on train+val that keeps known duplicates in one fold.

---

## 1. Feature settings (`exp_feature_params.py`)

Each family tested alone on all 17 classes with grouped CV, using Random Forest and logistic regression. A change counted only if it beat the old setting by more than fold-to-fold noise; settings within noise of each other went to the smaller one.

| Family | Old | New | Mean CV macro-F1, old → new |
|---|---|---|---|
| HOG | 9 orientations, 16 px cells, 8,100 dims | **12 orientations, 32 px cells, 2,352 dims** | 0.216 → 0.264 |
| HSV | 16×4×4 bins, 256 dims | **16×8×8 bins, 1,024 dims** | 0.861 → 0.895 |
| LBP | 3 scales, 54 dims | **no change** | 0.622 |

- Coarser HOG cells work better. Finer cells (8 px, up to 46,128 dims) were worse.
- The HSV gain is mostly logistic regression (0.801 → 0.866). Random Forest barely changed (0.921 → 0.924).
- Combining all 3 LBP scales clearly beats any single scale.

Details: `results/feature_params.csv`.

## 2. Choosing classes: 17 → 12 (`subset="native12"`)

Rule: a class stays if it has at least 250 native train images and at least 15 native test images. Only its native images are used.

| Outcome | Classes | Why |
|---|---|---|
| Dropped | Beet Armyworm | 0 native images |
| Dropped | Jute Red Mite, Mealybug, Jute Stem Weevil | 0 native test images |
| Dropped | Cutworm | 9 native test images |
| Kept (12) | Black Hairy, Field Cricket, Jute Aphid, Jute Hairy, Jute Semilooper, Jute Stem Girdler, Leaf Beetle, Pod Borer, Scopula Emissaria, Termite, Termite odontotermes, Yellow Mite | |

This leaves 4,128 train+val images and 265 test images. Supplied val covers only 4 of the 12 classes, so this set uses grouped CV instead of val.

**Why native images need caution:** every native file name looks machine-generated. 3,890 are named `Aug_<species>_#` and the other 1,933 follow `<name>_0_<random#>`, the pattern image-augmentation tools save. This suggests many native images are augmented copies of a smaller number of photos. Our duplicate check only linked 574 of the 3,890 `Aug_` files to another image, so grouped CV can't keep all related copies apart.

## 3. PCA on HOG (`exp_pca.py`)

HOG alone, grouped CV macro-F1, 17 classes:

| Components | Logistic reg. | RBF SVM | Random Forest |
|---|---|---|---|
| 10 | 0.133 | 0.278 | 0.323 |
| 15 | 0.169 | 0.337 | 0.355 |
| **25 (chosen)** | 0.204 | **0.388** | **0.384** |
| 50 | 0.218 | 0.374 | 0.353 |
| 100 | 0.225 | 0.328 | 0.312 |
| 200 | **0.236** | 0.297 | 0.267 |
| 500 | 0.200 | 0.256 | 0.210 |
| 1000 | 0.177 | 0.202 | 0.181 |
| No PCA (2,352 dims) | 0.213 | 0.356 | 0.293 |

- 25 components had the best average across the three models. **PCA helps every model:** Random Forest 0.293 → 0.384, RBF SVM 0.356 → 0.388, logistic regression 0.213 → 0.236 (at 200).
- HOG is still the weakest family even after PCA.
- PCA was applied to the HOG columns only, and whitened, so HOG doesn't overwhelm the color and texture columns.

Figure: `figures/final/pca_sweep.png`.

## 4. Every feature mix on every model (`exp_grid.py`)

11 configurations (7 mixes without PCA, 4 HOG mixes with PCA) × 6 models × 2 datasets = 132 grouped-CV runs.

**Best configuration of each model:**

| Model | 17 classes: best mix | CV macro-F1 | CV resized only | 12 native classes: best mix | CV macro-F1 |
|---|---|---|---|---|---|
| Random Forest | HOG+HSV+LBP (PCA) | **0.925** | 0.397 | HOG+HSV (PCA) | **1.000** |
| Linear SVM | HSV+LBP | 0.892 | 0.301 | HOG+HSV+LBP (PCA) | 0.999 |
| Logistic reg. | HOG+HSV+LBP (PCA) | 0.887 | 0.293 | HOG+HSV+LBP (PCA) | 0.998 |
| RBF SVM | HOG+HSV+LBP (PCA) | 0.861 | 0.261 | HOG+HSV+LBP (PCA) | 0.972 |
| KNN | HSV+LBP | 0.841 | 0.174 | HSV+LBP | 0.989 |
| Decision tree | HSV | 0.804 | 0.249 | HSV | 0.931 |

**Average over the six models, by mix (17 classes):**

| Mix | No PCA | With PCA on HOG |
|---|---|---|
| HSV | 0.856 | not applicable |
| HSV+LBP | 0.865 | not applicable |
| LBP | 0.614 | not applicable |
| HOG | 0.235 | 0.288 |
| HOG+LBP | 0.365 | 0.601 |
| HOG+HSV | 0.726 | 0.858 |
| HOG+HSV+LBP | 0.746 | 0.867 |

- This corrects the midpoint result that HOG hurts. Without PCA it does (0.746 vs 0.865). With PCA, adding HOG costs nothing (0.867 vs 0.865), though it barely helps either.
- KNN gains the most from PCA: all three features went from 0.469 to 0.841.
- On 12 native classes, many setups score 0.99+, so those rankings are effectively ties.
- On the supplied val set (17 classes, 68% resized), the best is Random Forest at 0.625.

Figures: `figures/final/grid_17class_cv.png`, `grid_17class_cv_resized.png`, `grid_17class_val.png`, `grid_native12_cv.png`. Full table: `results/grid_summary.csv`.

**Note on two runs.** Two linear SVM runs (12 classes, HOG and HOG+LBP with PCA) ran over an hour without finishing, while the other 130 took about 22 minutes in total. Refitting all 2,020 class-pair problems separately found the solver stalls on only 3 pairs, all HOG+LBP, all in fold 4, all involving Black Hairy. Those pairs were not unusually hard (90–95% separable), so this is a numerical stall rather than a difficult problem. It didn't reproduce for HOG alone when pairs were fitted separately, so it is only partly explained. Linear SVM now stops at 1,000,000 iterations per pair, and those two runs are marked in the results. Rechecking six finished runs with the limit changed macro-F1 by at most 0.00015. Details: `results/svm_convergence.csv`.

## 5. Class weights (`exp_final.py select`)

Top 3 models per dataset, grouped CV. "Small-class recall" is mean recall on the 3 classes with the fewest training images.

| Dataset | Model | Macro-F1: balanced / none | Small-class recall: balanced / none |
|---|---|---|---|
| 17 classes | Random Forest | 0.925 / 0.927 | **0.874 / 0.823** |
| 17 classes | Linear SVM | 0.892 / 0.892 | 0.827 / 0.827 |
| 17 classes | Logistic reg. | 0.887 / 0.886 | 0.819 / 0.808 |
| 12 native | All three | no meaningful difference | 1.000 / 1.000 |

Class weights barely change macro-F1. For Random Forest they raise small-class recall by 5 points. The rule fixed in advance was to take the higher macro-F1, so the final Random Forest uses no class weights, by a margin of only 0.0014.

## 6. Tuning and final test (`exp_final.py`)

Models were tuned with grouped CV on train+val, then each was scored **once** on test. The model marked ★ is the one CV chose for that dataset, not the one with the best test score.

| Dataset | Model and features | Settings | CV | Test accuracy | Test macro-F1 | Native | Resized |
|---|---|---|---|---|---|---|---|
| 17 classes | ★ Random Forest, HOG+HSV+LBP (PCA 25) | 300 trees, no class weights | 0.930 | 0.910 | **0.910** | 0.965 | 0.711 |
| 17 classes | Linear SVM, HSV+LBP | C=10, balanced | 0.892 | 0.810 | 0.791 | 0.873 | 0.581 |
| 17 classes | Logistic reg., HOG+HSV+LBP (PCA 25) | C=1, balanced | 0.886 | 0.810 | 0.791 | 0.867 | 0.587 |
| 12 native | ★ Random Forest, HOG+HSV (PCA 25) | 300 trees, no class weights | 1.000 | 1.000 | **1.000** | 1.000 | none |
| 12 native | Linear SVM, HOG+HSV+LBP (PCA 25) | C=0.1, balanced | 0.999 | 0.887 | 0.868 | 0.868 | none |
| 12 native | Logistic reg., HOG+HSV+LBP (PCA 25) | C=1, no class weights | 0.998 | 0.887 | 0.868 | 0.868 | none |

The linear models fall from 0.999 in CV to 0.868 on the 12-class test set. Test images come from outside the CV folds, so that drop is direct evidence that CV on native images is too optimistic. Random Forest did not drop on this small test set.

**Where the 17-class Random Forest goes wrong on test** (`results/confused_pairs_17class.csv`):

| True class | Predicted as | Images | Recall for true class |
|---|---|---|---|
| Scopula Emissaria | Jute Stem Weevil | 15 of 24 | 0.38 |
| Beet Armyworm | Jute Stem Weevil | 5 of 20 | 0.60 |
| Mealybug | Jute Red Mite | 2 of 24 | 0.79 |

10 of the 17 classes have recall 1.00, and 12 have at least 0.95. The 15 Scopula Emissaria errors are all copies of one photo that has no copy in training (section 10.4), so this is one photo misclassified 15 times. In the 12-class set, which has no Stem Weevil, Scopula is perfect.

Figures: `figures/final/cm_final_17class.png`, `cm_final_native12.png`.

## 7. Cross-source check: does the 12-class model learn the pest?

The 12-class models were trained only on native images, then asked to classify the **321 resized images of those same classes** (none used in training).

| Model | Native test macro-F1 | Resized images macro-F1 | Resized images accuracy |
|---|---|---|---|
| Random Forest | 1.000 | **0.242** | 0.340 |
| Linear SVM | 0.868 | 0.163 | 0.209 |
| Logistic reg. | 0.868 | 0.156 | 0.193 |

**A model that's perfect on native images is right only about a third of the time on other photos of the same species.** Random guessing across 12 classes would be about 0.08. The check can't fully separate whether the cause is augmented copies or different photo conditions, but the near-perfect native score clearly doesn't reflect general pest recognition.

Caveats: 194 of the 321 images are Jute Aphid, two classes have none, and several have only 6–13.

**Was the 1.000 caused by test images being copied into training?** A one-off check says not in any simple way:
- No test file is in train+val. The Random Forest was fit on train+val only.
- 2 of the 265 test images have an identical twin in training. With confirmed duplicates removed, the score is still 1.000 (244 of 244) and no test image has a twin.
- Comparing small grayscale versions of the images, allowing flips and 90° rotations, only 2 of 265 test images closely match a training image of the same class (correlation of at least 0.9), and those are the same 2 twins. For only 94 of 265 is the closest training image even the same class.

So we did not show that test images are near-copies of training images. This check would miss copies that were zoomed, cropped or rotated by other angles, so the augmented-copy explanation is still possible but unproven. The score comes from color and texture statistics, which work almost perfectly within native images and fail on resized photos.

**Correction:** the stronger check in section 10 settled this. **238 of the 265 test images have another version of the same photo in training.** The simple check above found only 2 because it could not see zoomed, cropped or freely rotated copies. The test files themselves were never used for training, but the 1.000 mostly reflects recognizing other versions of photos already seen.

## 8. Leakage: original vs. cleaned split

Same final models, trained and tested with confirmed duplicates and label conflicts removed.

| Dataset | Model | Test macro-F1: original → clean | Resized only: original → clean |
|---|---|---|---|
| 17 classes | Random Forest | 0.910 → **0.864** | 0.711 → **0.544** |
| 17 classes | Linear SVM | 0.791 → 0.740 | 0.581 → 0.442 |
| 17 classes | Logistic reg. | 0.791 → 0.733 | 0.587 → 0.426 |
| 12 native | Random Forest | 1.000 → 1.000 | none |
| 12 native | Linear SVM | 0.868 → 0.868 | none |
| 12 native | Logistic reg. | 0.868 → 0.869 | none |

Known duplicates inflate the 17-class scores by about 0.05, and resized images by about 0.14–0.17. This is a lower bound, because the duplicate check doesn't catch every copy.

## 9. Clustering: which pests look alike? (`exp_clustering.py`)

Uses HOG + HSV + LBP (PCA on HOG), all 7,236 images. No labels were used to form the groups.

**Methods:** Ward hierarchical clustering on each class's average vector, a t-SNE 2D map, and K-means. Adjusted Rand index (ARI) measures agreement with known groups: 0 is chance level and 1 is perfect. Insect groups used: beetle, moth/caterpillar, aphid/mealybug, mite, termite, cricket.

| Images | K-means ARI vs species | K-means ARI vs insect group |
|---|---|---|
| All | 0.050 | 0.027 |
| Native only | 0.102 | 0.022 |
| Resized only | 0.006 | 0.001 |

- **Cutting the class tree into 6 groups matches the real insect groups at ARI 0.07, close to chance.**
- **The three beetles do not group together.** Jute Stem Girdler pairs with Field Cricket, Jute Stem Weevil with Beet Armyworm, and Leaf Beetle with Jute Aphid and Yellow Mite. The two termites and the two mites also land apart.
- Some moths do group: Scopula Emissaria, Black Hairy and Jute Semilooper merge early.
- These features describe the look of an image (color, background, texture), not an insect's biology, so similar body plans don't end up close together.
- On the t-SNE map, native images form many tight little islands while resized images are scattered. That fits the idea that native images are clusters of augmented copies. Most species are spread across the map, though Termite and Jute Stem Girdler each have one fairly concentrated area.
- K-means with 2 clusters did not split the images by source (ARI 0.001).

Figures: `figures/final/dendrogram.png`, `tsne_by_source.png`, `tsne_by_species.png`.

---

## 10. Fair re-evaluation

A student in class suggested cropping each image to the insect. Looking into that raised a bigger question: why did the model score so high? This section answers three questions: what each change gives and why, why the original scores were so high, and what the score is once the dataset's bias is removed.

**Neural networks were used in preprocessing only.** The proposal said no neural networks anywhere, so this is a change we are disclosing. They were used to find the insect in each image and to group copies of the same photo. Their outputs never go into a classifier. The classifier inputs are still HOG, HSV and LBP, and the classifiers are still classical ML.

### 10.1 Short answers

1. **Why were the scores so high? Mostly copies of the same photo.** 324 of the 379 original test images have another version of their photo in training. The model gets 97.8% of those right and 50.9% of the other 55. Backgrounds add to it: on the original native set, the background alone identifies the pest at 0.961 macro-F1.
2. **What is the fair score?** **0.490 macro-F1 on 17 classes** (95% interval 0.453 to 0.522) and **0.577 on the 12 native classes** (0.519 to 0.621). Random guessing would score about 0.06 and 0.08. The model has real skill, but far less than 0.910 and 1.000.
3. **What does each change give?**
   - Removing known duplicates: 17 classes 0.910 → 0.864, 12 native classes stay at 1.000. That check misses most copies.
   - Removing all copies we could detect: 17 classes → 0.490, 12 native classes → 0.577. This is by far the biggest change.
   - Cropping to the insect, on the fair split: no measurable change.
   - Cropping and counting color and texture only on the insect: the best fair score, 0.515 on 17 classes and 0.635 on 12. That beats the crop alone, but its gain over the full image is within noise.
   - Tuning each image set separately: no gain over one fixed model (tuned minus fixed was −0.022 to +0.006). See 10.6.
   - Natural chunks of similar pests appear mainly on insect-only images, and they are mostly single species rather than wider insect groups. See 10.7.

**Other terms used here.** A *photo family* is a group of images that are versions of one photograph. The *fair split* keeps every family on one side. The *fair 5-fold score* holds out each of 5 groups of families in turn and scores each distinct photo once.

### 10.2 What we did

**Finding the insect (`vision/locate_insects.py`).** Grounding DINO, a text-prompted object detector, found insects using one fixed prompt for every image, so no class labels were used: "insect. bug. beetle. caterpillar. moth. mite. aphid. termite. cricket." SAM 2 then traced an exact outline of the insect inside each box. We took the top box, merged it with boxes scoring at least 80% as high (for swarms), added a 15% margin, made it square inside the image and resized it to 256×256.

- An insect was found in 7,099 of 7,236 images (native 98.3%, resized 97.4%). Yellow Mite was lowest at 90.9%, then Mealybug at 93.5% and Jute Red Mite at 94.9%. Images with no detection keep their existing version.
- Checked by eye on 60 native and 60 resized training images, against a required 80% good: native about 54 good, 4 partial, 2 no insect; resized about 53 good, 3 partial, 4 no insect.
- The median insect box covers 53% of a native image and 37% of a resized one.

**Finding copies of each photo (`vision/find_photo_families.py`).** DINOv2 image embeddings proposed similar pairs, across all 8 flip and 90° rotation orientations. LightGlue with RANSAC then counted keypoints that agree on one geometric transform. We tested this on images we edited ourselves, so we knew the right answer:

| Known edit | DINOv2 alone (similarity ≥ 0.95) | With LightGlue (≥ 15 matching points) |
|---|---|---|
| Flip | 100% found | 100% |
| 90° rotation | 24% | 100% |
| Color shift | 98% | 100% |
| Free rotation | 39% | 70% |
| Zoom | 45% | 69% |
| Several edits combined | 19% | 54% |

At 15 points, the closest same-class look-alikes linked 14.3% of the time, and some of those may be real copies. By eye, pairs with 15 to 20 points were the same photo 11 of 12 times, all 36 pairs at 20 or more were the same photo, and 8 to 14 points still included several real copies.

We used 15 points. A false link only keeps two photos on the same side, which can't raise a score. A missed copy can, so we chose the setting that catches more copies.

| Copy rule | Families | Native images → photos | Resized images → photos | Original test images with a relative in train/val |
|---|---|---|---|---|
| Before this work (ORB check) | | | | 78 of 379 |
| Strict (≥ 40 points) | 1,880 | 5,823 → 717 | 1,413 → 1,189 | 277 of 379 |
| **Used (≥ 15 points)** | 1,503 | 5,823 → 474 | 1,413 → 1,058 | **324 of 379** |

The largest family has 136 images. 275 images sit in families that span more than one class.

**The fair split (`make_fair_split.py`).** Whole families go to one of 5 stratified folds. Fold 0 is the test set and the rest is training. The 10 images with confirmed label conflicts are dropped. Train has 5,763 images. Test has 1,463 images, which is 303 distinct photos. Test scores use one image per photo, so a photo with 30 copies counts once. Across all folds there are 1,511 distinct photos.

**Image sets (`JUTE_IMAGES`).**

| Name | Image | HOG | HSV and LBP counted on |
|---|---|---|---|
| `full` | Existing 256×256 image | Whole image | Real content of the image |
| `crop` | Crop around the insect | Whole crop | Whole crop |
| `insect_only` | Crop around the insect | Whole crop, because masking adds false edges | Insect pixels only |
| `background` | Existing 256×256 image | Not used | Non-insect pixels only (probe only) |

**Changes from the original plan.** We added LightGlue because DINOv2 alone missed most rotations and zooms. We added `insect_only` to the full rerun. We kept the same 12 classes so comparisons are like for like, because no class has 15 distinct native test photos, which the original rule requires. We added the 5-fold score because the single test set has only 1 to 9 photos in some classes.

### 10.3 What each change gives

One fixed model throughout: Random Forest, 300 trees, HOG + HSV + LBP, PCA 25 on HOG. Only the listed factor changes. Macro-F1:

| Classes | Split | Full image | Crop | Insect only |
|---|---|---|---|---|
| 17 | Original | 0.910 | 0.908 | 0.880 |
| 17 | Known duplicates removed | 0.864 | 0.876 | 0.852 |
| 17 | Fair, single test set (303 photos) | 0.462 | 0.482 | 0.533 |
| 17 | **Fair 5-fold (1,511 photos)** | **0.490** | 0.481 | **0.515** |
| 12 native | Original | 1.000 | 0.996 | 0.905 |
| 12 native | Known duplicates removed | 1.000 | 1.000 | 0.882 |
| 12 native | Fair, single test set (66 photos) | 0.545 | 0.524 | 0.617 |
| 12 native | **Fair 5-fold (325 photos)** | **0.577** | 0.558 | **0.635** |

The fair 5-fold score is the one to quote, because the single test set is small. On 17 classes it is 0.510 / 0.502 / 0.560 on native images and 0.369 / 0.361 / 0.405 on resized images (full / crop / insect only).

**How noisy is this?** Resampling the photos 2,000 times, using the same draws for each image set (`exp_bootstrap.py`):

| Comparison | 17 classes | 12 native classes |
|---|---|---|
| Full image | 0.490 (0.453 to 0.522) | 0.577 (0.519 to 0.621) |
| Crop | 0.481 (0.446 to 0.514) | 0.558 (0.505 to 0.605) |
| Insect only | 0.515 (0.478 to 0.548) | 0.635 (0.577 to 0.685) |
| Crop minus full | −0.009 (−0.026 to +0.008) | −0.018 (−0.054 to +0.022) |
| Insect only minus full | +0.026 (−0.007 to +0.059) | +0.058 (−0.002 to +0.121) |
| Insect only minus crop | **+0.035 (+0.003 to +0.067)** | **+0.076 (+0.023 to +0.134)** |

An interval that includes 0 means the difference could be noise.

**Why each change has the effect it does:**

- **Removing known duplicates barely matters** because that check found 78 test images with a copy, and 324 have one.
- **Removing all copies matters most** because most of the original score came from recognizing other versions of a photo (10.4).
- **Cropping changes nothing measurable.** The insect box already covers about half of a typical native image, and the median native crop needs no upscaling, so a crop is often close to the whole frame. The score also comes mainly from color and texture statistics rather than fine shape, which is a likely reason that framing matters little.
- **Masking lowers the original-split score but raises the fair score.** It lowers 0.910 to 0.880 and 1.000 to 0.905 on the original split, and raises 0.490 to 0.515 and 0.577 to 0.635 on the fair split. That is consistent with the background helping to match copies of a known photo, but doing little for a new one.

**Per class, fair 5-fold F1:**

| Class | Distinct photos | Images | Images per photo | Full | Crop | Insect only |
|---|---|---|---|---|---|---|
| Yellow Mite | 23 | 392 | 17.0 | 0.21 | 0.15 | 0.14 |
| Pod Borer | 28 | 411 | 14.7 | 0.26 | 0.21 | 0.45 |
| Field Cricket | 42 | 426 | 10.1 | 0.28 | 0.29 | 0.36 |
| Jute Hairy | 62 | 441 | 7.1 | 0.29 | 0.32 | 0.38 |
| Jute Semilooper | 22 | 398 | 18.1 | 0.34 | 0.31 | 0.47 |
| Cutworm | 91 | 437 | 4.8 | 0.35 | 0.33 | 0.43 |
| Black Hairy | 35 | 356 | 10.2 | 0.45 | 0.46 | 0.51 |
| Termite odontotermes (Rambur) | 21 | 361 | 17.2 | 0.48 | 0.55 | 0.38 |
| Beet Armyworm | 193 | 259 | 1.3 | 0.49 | 0.44 | 0.35 |
| Mealybug | 120 | 525 | 4.4 | 0.53 | 0.52 | 0.64 |
| Jute Aphid | 225 | 578 | 2.6 | 0.56 | 0.55 | 0.60 |
| Jute Stem Weevil | 317 | 771 | 2.4 | 0.62 | 0.62 | 0.72 |
| Leaf Beetle | 37 | 343 | 9.3 | 0.64 | 0.58 | 0.63 |
| Termite | 22 | 337 | 15.3 | 0.64 | 0.64 | 0.51 |
| Jute Stem Girdler | 23 | 311 | 13.5 | 0.69 | 0.64 | 0.61 |
| Scopula Emissaria | 46 | 350 | 7.6 | 0.70 | 0.74 | 0.70 |
| Jute Red Mite | 204 | 530 | 2.6 | 0.80 | 0.83 | 0.89 |

The 7,226 images in this table are really 1,511 photos. Classes with about 20 photos have noisy scores. We did not test why masking helps some classes and hurts others.

### 10.4 Why the original scores were so high

**Copies (main cause).** The original test images, scored by whether their photo was in training (`results/fair_copy_probe.csv`, 17-class model unless noted):

| Test images | Images | Distinct photos | Accuracy |
|---|---|---|---|
| Photo in training, all | 324 | 134 | **0.978** |
| Photo not in training, all | 55 | 41 | **0.509** |
| Photo in training, native | 257 | 82 | 1.000 |
| Photo not in training, native | 17 | 3 | 0.118 |
| Photo in training, resized | 67 | 53 | 0.896 |
| Photo not in training, resized | 38 | 38 | 0.684 |
| 12-class model, photo in training | 238 | 74 | 1.000 |
| 12-class model, photo not in training | 27 | 4 | 1.000 |

- The 379 test images are only 175 distinct photos.
- A native photo appears as about 12 images on average (5,823 images from 474 photos), and a resized photo as about 1.3 (1,413 from 1,058). The dataset's `Aug_` files are flips, rotations, crops and color shifts of the same photographs.
- The native groups with no copy in training are too small to read on their own. The 17 images are 3 photos and the 27 are 4 photos. Checked by eye against their nearest training images, the Scopula and Jute Aphid photos are clearly different photos; the Black Hairy and termite ones can't be ruled out as copies. In the 17-class model, one Scopula photo with 15 copies accounts for all 15 errors.
- The test files themselves never trained the model. The issue is that other versions of those photos did.
- On the t-SNE maps, native images form tight strings and clumps, which are copies of one photo. Resized images are scattered (`tsne_by_source.png`).

**Background (contributes).** Color and texture only, on images where an insect was found. Macro-F1:

| Classes | Split | Whole image | Insect only | Background only |
|---|---|---|---|---|
| 17 | Original | 0.898 | 0.878 | 0.828 |
| 17 | Fair 5-fold | 0.489 | 0.510 | 0.380 |
| 12 native | Original | 1.000 | 0.903 | **0.961** |
| 12 native | Fair 5-fold | 0.543 | 0.631 | 0.448 |

With no insect at all, the original native set still scores 0.961. On the fair split, the insect beats the background (0.631 against 0.448), which is the right way round for a model looking at the pest. The background still scores far above chance. This could mean some pests really do appear in different settings, or that each class's photos came from similar sources. We can't separate the two.

**Image source (contributes).** Train on one source and test on the other, fair 5-fold, macro-F1. Each row covers only the classes with enough of both sources, so rows are not directly comparable:

| Trained on → tested on | Classes | Full image | Crop |
|---|---|---|---|
| Native → native | 16 | 0.504 | 0.520 |
| Native → resized | 14 | **0.163** | 0.165 |
| Resized → native | 7 | **0.181** | 0.200 |
| Resized → resized | 8 | 0.573 | 0.571 |

Models barely transfer between the two sources, and cropping doesn't fix that. A model trained on native images only is learning what native images look like, not just what the pest looks like.

### 10.5 Why resized images score lower

- **Not the padding.** Accuracy was 0.647 on padded images and 0.610 on unpadded ones. Only 257 of the 1,308 train+val resized images were padded and 6 were blurred.
- **Fewer copies to match.** On the original split, 94% of native test images (257 of 274) have a copy in training, against 64% of resized ones (67 of 105). The gap between native and resized macro-F1 shrinks from 0.254 to 0.141 on the fair split (0.965 and 0.711, against 0.510 and 0.369).
- **Resized images are more varied.** A native image's nearest neighbor in color and texture space is the same class 99.8% of the time, against 50.6% for resized images, and native neighbors are about 5 times closer.
- **The sources differ** (10.4). Training on resized images only gave 0.713 against 0.654 on 7 classes, a rough comparison.
- **Some images are poor.** Resized images had about twice the no-insect rate in the spot check, for example a photo of a mushroom.

The remaining gap can't be fully explained, because the two sources cover different classes.

### 10.6 Full rerun on the fair split

The whole pipeline was repeated separately for each image set: feature-settings study, PCA sweep, 132-run grid, class weights, tuning, then one score on the fair test set. Every choice used grouped CV on the training side only. The settings study found no setting better than the current ones beyond noise on any of the three image sets, so settings did not change. Insect-only images use the same HOG as crops (only HSV and LBP are limited to the insect), so their HOG results are identical. PCA chose 50 HOG components on all three image sets.

| Classes | Image set | Model chosen by CV | CV | Test accuracy | Test macro-F1 | Native | Resized | All test copies |
|---|---|---|---|---|---|---|---|---|
| 17 | Full | Random Forest, HSV+LBP | 0.402 | 0.568 | **0.438** | 0.469 | 0.357 | 0.434 |
| 17 | Crop | Random Forest, HSV+LBP | 0.405 | 0.574 | **0.465** | 0.492 | 0.384 | 0.450 |
| 17 | Insect only | Random Forest, HOG+HSV+LBP (PCA 50) | 0.502 | 0.568 | **0.479** | 0.568 | 0.381 | 0.547 |
| 12 native | Full | Random Forest, HOG+HSV+LBP (PCA 50) | 0.513 | 0.470 | **0.477** | 0.477 | none | 0.457 |
| 12 native | Crop | Random Forest, HOG+HSV+LBP (PCA 50) | 0.526 | 0.515 | **0.517** | 0.517 | none | 0.478 |
| 12 native | Insect only | Random Forest, HOG+HSV+LBP (PCA 50) | 0.586 | 0.621 | **0.613** | 0.613 | none | 0.656 |

- Random Forest was the top model by CV in every case. The other two tuned models per case are in each `final_results.csv`.
- Tuned scores are within 0.07 of the fixed-model scores in 10.3, a small gap for test sets of 303 and 66 photos. In one case a model CV did not choose scored higher on test (insect-only RBF SVM, 0.517). We kept the CV choice.
- **The 5-fold score in 10.3 is the better estimate** of general performance. CV here is lower than that score because it trains on less data and counts every copy, while the 5-fold score counts each photo once.
- HOG adds little on full images and crops. Random Forest with HOG alone gets 0.187 and 0.199 in CV on 17 classes, and HSV+LBP won without it.
- The 12-class models scored 0.118 / 0.146 / 0.327 (full / crop / insect only) on the 50 resized test photos of the same classes.
- PCA on HOG, averaged over the four feature mixes that include HOG: Random Forest gains about 0.02 on every image set (0.01 for crop on 12 classes), KNN gains the most (+0.05 to +0.13), and RBF SVM loses 0.02 to 0.03 on full images and crops.

**Does tuning each image set separately help?** Each tuned model was also scored across all 5 fair folds, each photo once, on the same resampled photos as the fixed model in 10.3:

| Classes | Image set | Fixed model (10.3) | Tuned model chosen by CV | Tuned minus fixed (95% interval) |
|---|---|---|---|---|
| 17 | Full | 0.490 | 0.472 | −0.018 (−0.037 to +0.002) |
| 17 | Crop | 0.481 | 0.465 | −0.016 (−0.033 to 0.000) |
| 17 | Insect only | 0.515 | 0.493 | −0.022 (−0.042 to −0.002) |
| 12 native | Full | 0.577 | 0.561 | −0.015 (−0.051 to +0.020) |
| 12 native | Crop | 0.558 | 0.565 | +0.006 (−0.016 to +0.030) |
| 12 native | Insect only | 0.635 | 0.620 | −0.014 (−0.034 to +0.003) |

- **Separate tuning did not raise the score.** It was lower in five of six cases, though only one of those gaps is clearly different from zero. The tuned settings were chosen by CV on folds 1 to 4, so if anything these tuned scores lean slightly high.
- A likely reason is that tuning CV counts every copy of a photo, while this score counts each photo once, so tuning optimizes a slightly different target. The differences between the top configurations in CV were also small.
- Insect-only stays ahead with tuned models: over full +0.021 (−0.014 to +0.057) on 17 classes and +0.059 (−0.003 to +0.120) on 12, over crop +0.028 (−0.002 to +0.060) and +0.056 (+0.001 to +0.110). As with the fixed model, the lead over full is within noise.
- SVMs gain the most from insect-only images. Tuned linear SVM on 17 classes scores 0.343 / 0.363 / 0.454 (full / crop / insect only). On insect-only images, tuned RBF SVM (0.504) is level with tuned Random Forest (0.493).

### 10.7 Clustering on the fair split

Do the pests form chunks without being told the labels? To compare image sets fairly, this uses the same features for all three: HOG + HSV + LBP with PCA 50 on HOG, one image per distinct photo (1,511 photos). "Nearest other photo" is the closest image from a different photo. Ranges are the middle 95% of 200 random 80% samples of the photos.

| Measure | Full | Crop | Insect only | Chance |
|---|---|---|---|---|
| Nearest other photo is the same species | 0.357 | 0.359 | **0.518** | 0.116 |
| Nearest other photo is the same insect group | 0.448 | 0.463 | **0.632** | 0.238 |
| K-means vs. species (ARI) | 0.045 (0.018 to 0.055) | 0.034 (0.010 to 0.039) | 0.120 (0.036 to 0.155) | 0 |
| K-means vs. insect group (ARI) | 0.031 (0.000 to 0.036) | 0.018 (0.000 to 0.028) | 0.003 (0.000 to 0.116) | 0 |
| Classes whose closest class is in their insect group (of 17) | 4 | 2 | 8 | not computed |

Nearest other photo, same insect group, by group:

| Insect group | Photos | Full | Crop | Insect only | Chance | Insect only: same species |
|---|---|---|---|---|---|---|
| Mite | 227 | 0.471 | 0.502 | **0.828** | 0.150 | 0.815 |
| Beetle | 377 | 0.366 | 0.403 | **0.695** | 0.249 | 0.645 |
| Moth/caterpillar | 477 | 0.522 | 0.537 | 0.608 | 0.315 | 0.367 |
| Termite | 43 | 0.535 | 0.535 | 0.605 | 0.028 | 0.488 |
| Aphid/mealybug | 345 | 0.429 | 0.423 | 0.525 | 0.228 | 0.435 |
| Cricket | 42 | 0.286 | 0.214 | 0.190 | 0.027 | 0.190 |

- **Chunks appear mainly on insect-only images.** Mites and beetles improve the most, and the t-SNE maps show mites and aphids/mealybugs pulling together only on this set. Crickets get worse, on only 42 photos.
- **The chunks are mostly species, not wider insect groups.** Most same-group matches among mites and beetles are the same species. Mites and beetles do not merge across species in the class tree.
- **The gain comes from limiting color and texture to the insect.** HSV + LBP alone gives the same insect-only result (0.518 same species). HOG is the same for crop and insect-only.
- **K-means is a weak and unstable measure here.** Insect-only beats crop against species by +0.068 (+0.010 to +0.126), and its lead over full is +0.057 (−0.003 to +0.116). Against insect groups, no image set can be told apart from another.
- **Class tree (`dendrograms_one_per_photo.png`).** The three beetles never join each other first on any image set. The two termite classes pair directly on all three. Jute Stem Girdler pairs with Field Cricket on full and crop only. Scopula Emissaria pairs with Jute Semilooper on insect-only only. On insect-only, most caterpillar classes sit on one branch (Jute Hairy, Beet Armyworm, Cutworm, Pod Borer), with Jute Aphid inside it.
- **Copies change the picture.** With every copy included (7,226 images), the maps break into many small tight blobs, though insect-only still leads (same species 0.432 vs. 0.310 and 0.315). This is why one image per photo is the fair view.
- **Caution.** Some small tight blobs on the full-image and crop maps may be copies the detector missed. Part of the mite chunk may also come from how very small insects get outlined, which we did not test.

Each image set's own best features with every copy (`exp_clustering.py`), not like for like, gave species ARI 0.033 / 0.054 / 0.090 (full / crop / insect only) and insect-group ARI 0.027 / 0.037 / 0.068.

### 10.8 Checks we ran

- No photo family crosses a fold or the train/test boundary (asserted in code and rechecked: 0).
- Each image set has 132 grid rows, and each final model was scored once on test.
- With no environment variables set, the 17-class Random Forest reproduces its stored CV score of 0.930 and the fixed model reproduces 0.910, 0.864 and 1.000. The original `features/` and `results/` files were not changed by this rerun; only new files were added.
- All 27 charts in the three fair figure folders and the 3 charts in `figures/cluster_compare/` were inspected.
- The fixed-model rows in `fair_tuned_5fold.csv` reproduce `fair_bootstrap.csv` exactly, intervals included. The cluster comparison gave identical numbers on a second run. The insect-only settings study's HOG scores equal the crop study's exactly, as they should.

### 10.9 Limitations

- **Copy detection is incomplete.** It finds only 54% of heavily combined edits, and pairs with 8 to 14 points include real copies, so the fair score is probably still slightly high.
- **Few distinct photos.** The smallest classes have 21 to 23. In the single test set, some classes have 1 to 9 photos, which is why a class can show 1.00 or 0.00.
- **The intervals cover which photos were drawn, not retraining.** They would be wider if retraining were also included.
- **Source and class are still tied.** Beet Armyworm is all resized, and Yellow Mite and one termite class are all native.
- **Masks and boxes are imperfect.** 137 images had no detection, and the background and insect-only probes depend on outline quality.
- **The background probe can't separate scene from photo source.**
- **The 12 classes were chosen on the original split.** That keeps the comparison like for like, but the class choice itself came from the original data.

### 10.10 Files and figures

Results: `results/fair_matrix.csv`, `fair_probes.csv`, `fair_copy_probe.csv`, `fair_bootstrap.csv`, `fair_per_class.csv`, `fair_tuned_5fold.csv`, `fair_cluster_compare.csv`, `fair_cluster_differences.csv`, `fair_cluster_by_group.csv`, `fair_cluster_merges.csv`, `photo_family_calibration.csv`, `photo_family_match_calibration.csv`, and `results_{full,crop,insect_only}_fair/`.

| File | Shows |
|---|---|
| `figures/fair/crop_check_native.png` | Insect crops and masks on native training images |
| `figures/fair/crop_check_resized.png` | Same, on resized training images |
| `figures/fair/family_check.png` | Pairs just above the copy threshold |
| `figures/fair/family_check_below.png` | Pairs just below the copy threshold |
| `figures/final_{full,crop,insect_only}_fair/grid_17class_cv.png` | Every model × feature mix, 17 classes |
| `.../grid_17class_cv_resized.png` | Same grid, resized images only |
| `.../grid_native12_cv.png` | Same grid, 12 native classes |
| `.../pca_sweep.png` | HOG score vs. number of PCA components |
| `.../cm_final_17class.png`, `cm_final_native12.png` | Confusion matrices on the fair test set; some classes have very few photos |
| `.../dendrogram.png` | Which pests look alike |
| `.../tsne_by_source.png`, `tsne_by_species.png` | t-SNE maps; each image set uses its own best features and every copy |
| `figures/cluster_compare/tsne_groups_one_per_photo.png` | t-SNE maps by insect group, same features on full, crop and insect-only, one image per photo |
| `figures/cluster_compare/tsne_groups_every_copy.png` | Same, with every copy of each photo |
| `figures/cluster_compare/dendrograms_one_per_photo.png` | Which pests look alike, the three image sets side by side on one scale |

---

## Limitations (sections 1 to 9; see 10.9 for the fair split)
- **Most test images have a copy of their photo in training** (section 10), so test scores in sections 1 to 9 are too high as a measure of general performance.
- The 250/15 class cutoff was set after looking at the image counts.
- Grouped CV only separates duplicates we detected, so CV scores on native images are too high. Test and cross-source results are the ones to trust.
- Feature settings and model selection both used the same CV folds, which adds slight optimism to CV numbers. Test was not used for either.
- The test set is small: 379 images (265 for 12 classes, 105 resized for 17 classes). Per-class recall rests on only 19 to 24 images.
- Image source and class are still tied together. For example, Beet Armyworm is only resized images.
- Insect groups in the clustering check come from our own mapping of each pest to its group.
- `figures/feature_visualization.png` was made with the old HOG and HSV settings.

## Figure list (`figures/final/`)

| File | Shows |
|---|---|
| `grid_17class_cv.png` | Every model × feature mix, 17 classes |
| `grid_17class_cv_resized.png` | Same grid, resized images only |
| `grid_17class_val.png` | Same grid, supplied val set |
| `grid_native12_cv.png` | Same grid, 12 native classes |
| `pca_sweep.png` | HOG score vs. number of PCA components |
| `cm_final_17class.png` | Confusion matrix, best 17-class model on test |
| `cm_final_native12.png` | Confusion matrix, best 12-class model on test |
| `dendrogram.png` | Which pests look alike, hierarchical clustering |
| `tsne_by_source.png` | t-SNE map by native vs. resized |
| `tsne_by_species.png` | t-SNE map, one panel per species |
