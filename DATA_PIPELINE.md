# Data & Training Pipeline

End-to-end reference for how data flows from the raw LARS dataset through
preprocessing, label cleaning, augmentation, model training, evaluation, and
explainability.

---

## Overview

```
Raw LARS v1.0.0
    │
    ▼
2_DataPreprocessing/datasplit.py   → Data/lars_processed/{train, valid, test, test_unused}/
    │
    ├──▶ [optional] 2_DataPreprocessing/rfdetr_preprocessing.py   (offline augmentation, RF-DETR only)
    │
    ▼
2_DataPreprocessing/cv_relabel.py  → Data/lars_relabeled/{train, valid, test}/_annotations.coco.json
    │                                + provenance.json   (annotation JSONs only, no images)
    ▼  (manual cp into lars_processed/)
Data/lars_processed/   ← active annotations replaced with cv-cleaned versions
    │
    ├──▶ 3_Model/train_rfdetr.py            → runs/rfdetr/baseline/
    ├──▶ 3_Model/optuna_search_rfdetr.py    → runs/rfdetr/optuna_relabeled_lb/
    │
    ├──▶ 3_Model/train_fasterrcnn.py        → runs/fasterrcnn/baseline/
    ├──▶ 3_Model/optuna_search_fasterrcnn.py → runs/fasterrcnn/optuna/
    │
    ├──▶ 3_Model/train_yolo.py              → Data/lars_yolo/ + runs/yolo/baseline/
    ├──▶ 3_Model/optuna_search_yolo.py      → runs/yolo/optuna/
    │
    ▼
3_Model/export_predictions_*.py  → <run_dir>/predictions_test.json
    │
    ▼
3_Model/evaluate_all_models.ipynb   →  per-model metrics, comparison figures
3_Model/eval.py                     →  headless eval, appends row to runs/model_results.csv
    │
    ▼
5_XAI/XAI.ipynb                     →  D-RISE, Grad-CAM++, EigenCAM, RF-DETR x-attn
```

---

## 1. Raw Dataset — `Data/lars_v1.0.0_*`

| Source path | Contents |
|---|---|
| `lars_v1.0.0_annotations/train/panoptic_annotations.json` | COCO panoptic (bbox + masks), 2,605 images |
| `lars_v1.0.0_annotations/train/image_annotations.json` | Scene-level labels (scene_type, lighting, reflections, waves) |
| `lars_v1.0.0_annotations/val/` | Same format, 198 images |
| `lars_v1.0.0_annotations/test/` | Image-level labels only — **no bbox annotations** |
| `lars_v1.0.0_images/{train,val,test}/images/` | Raw JPEGs |

**11 segmentation categories**: 3 stuff (Water, Sky, Static Obstacle) + 8 thing classes
used for detection (Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float, Other).

---

## 2. Data Split — `2_DataPreprocessing/datasplit.py`

**Input:** Raw annotations + images
**Output:** `Data/lars_processed/{train, valid, test, test_unused}/`

### What it does
1. Reads the original LARS **train** split (2,605 images) and performs a **scene-level
   stratified 80/20 split** — images from the same sequence (`davimar_seq_01_*`)
   always stay together to prevent data leakage.
2. Stratification key: concatenation of dominant `scene_type | lighting | reflections | waves`
   per scene. Strata with only one scene are merged into a `"rare"` bucket to avoid
   `sklearn` errors.
3. Extracts **thing-class bboxes only** (drops stuff classes) from the panoptic JSON.
4. Copies images and writes `_annotations.coco.json` (COCO detection format) per split.

### Output splits

| Split | Source | Images | Has bbox labels |
|---|---|---|---|
| `train` | 80% of LARS train | 2,102 | ✓ |
| `test` | 20% of LARS train | 503 | ✓ |
| `valid` | LARS val (all) | 198 | ✓ |
| `test_unused` | LARS test (all) | 1,203 | ✗ (image-level only) |

> **Seed:** `random_state=4`

`test` and `train` both come from the original LARS train split and have panoptic
bbox labels. `test_unused` (original LARS test) has no bbox labels and is **not used**
by any model training or evaluation script.

---

## 3. Label Cleaning — `2_DataPreprocessing/cv_relabel.py`

**Input:** `Data/lars_processed/{train, valid, test}/_annotations.coco.json`
**Output:** `Data/lars_relabeled/{train, valid, test}/_annotations.coco.json` + `provenance.json`
**Intermediate:** `runs/cv_relabel/fold_{0-3}/`

### What it does
Runs **4-fold cross-validation** to find and fix noisy annotations:

1. **Pools all three splits** (train + valid + test, ~2,803 images total) and divides
   them into 4 scene-level folds (same scene-grouping logic as `datasplit.py`).
2. **Per fold k:**
   - Trains `RFDETRBase` on folds 0–3 (excluding k) using hardcoded hyperparameters
     (see below).
   - Runs inference at `threshold=0.0` on held-out fold k.
   - Applies four cleaning rules:
     - **Ghost / phantom boxes** — GT box with area ≥ 2,000 px² that has no
       matching prediction above IoU 0.5 and best matching confidence < 0.05 → **removed**.
     - **Merged / oversized boxes** — GT box (matched or not) that contains ≥ 2 extra
       high-confidence (≥ 0.5) prediction centres → **removed**.
     - **Missing labels** — Unmatched prediction with confidence > 0.55 that does not
       largely overlap an existing GT box → **added** as new annotation.
     - **Resize** — High-conf FP that covers ≥ 50% of an existing GT box (but doesn't
       qualify as a new label) → existing GT box **expanded** to the prediction extent.
3. Caches cleaned annotations per fold to `runs/cv_relabel/fold_k/cleaned_anns.json`.
4. Reconstructs the original train/valid/test membership and writes relabeled JSONs.
5. Writes `provenance.json` logging every removed / resized / added annotation.

### `relabel_action` field semantics (added to each annotation)

| Value | Meaning | Extra fields |
|---|---|---|
| `"added"` | New annotation inserted by the CV model (was a high-conf FP) | `relabel_conf` |
| `"kept"` | Original annotation that survived, on an image where at least one other change was made | — |
| `"resized"` | Original annotation expanded to model's predicted extent | `relabel_conf`, `relabel_orig_bbox` |
| *(absent)* | Original annotation on an image the model made **no changes to** | — |
| *(gone)* | Removed ghost/merged box — simply absent from output | — |

### Applying cleaned labels
`lars_relabeled/` contains **annotation JSONs only** — no images are copied.
To activate them, copy over `lars_processed`:
```bash
cp Data/lars_relabeled/train/_annotations.coco.json Data/lars_processed/train/_annotations.coco.json
cp Data/lars_relabeled/valid/_annotations.coco.json Data/lars_processed/valid/_annotations.coco.json
cp Data/lars_relabeled/test/_annotations.coco.json  Data/lars_processed/test/_annotations.coco.json
```
Backups for train/valid are kept as `_annotations.coco.json.pre-relabel` in `lars_processed/`.

### Current verified annotation counts

| File | Images | Anns | Cleaned (`relabel_action` present) |
|---|---|---|---|
| `lars_processed/train/_annotations.coco.json` | 2,102 | 8,025 | 4,211 (522 added, 3,689 kept) |
| `lars_processed/train/_annotations.coco.json.pre-relabel` | 2,102 | 7,761 | 0 — datasplit baseline |
| `lars_processed/valid/_annotations.coco.json` | 198 | 1,119 | 552 (49 added, 503 kept) |
| `lars_processed/valid/_annotations.coco.json.pre-relabel` | 198 | 1,107 | 0 — datasplit baseline |
| `lars_processed/test/_annotations.coco.json`  | 503 | 1,605 | 751 (106 added, 645 kept) |

### Net changes from cv_relabeling (from `provenance.json`)

| Split | Modified images | Removed | Added | Net |
|---|---|---|---|---|
| train | 422 | 258 | 522 | +264 |
| valid | 46  | 37  | 49  | +12  |
| test  | 101 | 61  | 106 | +45  |

> **Current state:** Active annotations in `lars_processed/` for all three splits are
> the cv-relabeled versions.

---

## 4. Offline Augmentation — `2_DataPreprocessing/rfdetr_preprocessing.py`

**Input:** `Data/lars_processed/train/`
**Output:** Augmented JPEGs written into `Data/lars_processed/train/images/`;
`_annotations.coco.json` updated in-place (original backed up as
`_annotations.coco.original.json`).

### Purpose
RF-DETR-specific **offline** augmentation. Generates N copies of every training
image (default: 1 copy → 2× dataset). Augmented filenames carry the `_aug` suffix.
Used standalone when you want a fixed augmented training set; the Optuna RF-DETR
search uses its own per-trial augmentation policies and does not need this script.

### Augmentation pipeline (fixed)
| Transform | Parameters | Motivation |
|---|---|---|
| HorizontalFlip | p=0.5 | Boats face both directions |
| RandomBrightnessContrast | ±0.25 / ±0.25, p=0.7 | Dawn, dusk, overcast |
| HueSaturationValue | hue±5, sat±40, val±30, p=0.6 | Colour temperature shift |
| GaussianBlur | kernel 3–7 px, p=0.25 | Sea haze, spray |
| CLAHE | clip 2.0, p=0.25 | Glare and reflections |

Boxes below 50 px² or less than 10% visible after transforms are dropped.

### Undo
```bash
python rfdetr_preprocessing.py --undo
```

---

## 5. Model Training

### 5a. RF-DETR — `3_Model/train_rfdetr.py`

**Input:** `Data/lars_processed/` (COCO JSON, images flat inside split dirs)
**Output:** `runs/rfdetr/baseline/`

- Model: `RFDETRBase` (default) or `RFDETRLarge` (from `rfdetr` library)
- Separate LR for encoder (backbone) vs. decoder
- Logs to `training.log` and `metrics.csv`

Defaults mirror the best Optuna trial by val mAP@50
(`rfdetr/optuna_relabeled_lb/trial_000`):

| Param | Default | Notes |
|---|---|---|
| epochs | 100 | |
| batch_size | 24 | per-step |
| grad_accum_steps | 4 | effective batch = 96 |
| lr | 4.39e-4 | |
| lr_encoder | 5.30e-5 | = lr × 0.121 |
| resolution | 672 | |
| weight_decay | 4.49e-4 | |
| grad_clip_max_norm | 0.259 | |
| aug_copies | 3 | when `--aug-policy` ≠ none |
| early_stopping_patience | 10 | |

### 5b. Faster R-CNN — `3_Model/train_fasterrcnn.py`

**Input:** `Data/lars_processed/` (COCO JSON)
**Output:** `runs/fasterrcnn/baseline/`

Two model variants:

| Variant | Backbone | Detection heads | Notes |
|---|---|---|---|
| `base` | ResNet-50-FPN-v2 | **COCO pretrained** (head replaced for 8+1 classes) | Fast convergence; canonical |
| `large` | ResNet-101-FPN | **Randomly initialised** | Needs many more epochs; not used for final results |

- SGD with momentum, StepLR scheduler
- Evaluation via `supervision.metrics.MeanAveragePrecision`
- Tuned anchors come from `3_Model/fasterrcnn_anchor_analysis.ipynb`
  (coverage 66.6% vs. 45.6% with default anchors)

Defaults mirror the best Optuna trial by val mAP@50
(`fasterrcnn/optuna/trial_006`):

| Param | Default |
|---|---|
| epochs | 25 |
| batch_size | 8 |
| lr | 3.37e-3 |
| lr_backbone | 7.85e-4 (= lr × 0.233) |
| momentum | 0.9 |
| weight_decay | 2.01e-5 |
| step_size / gamma | 20 / 0.05 |
| early_stopping_patience | 7 |
| anchor_sizes | (8, 24, 56, 96, 112, 176, 288, 320, 624) |
| aspect_ratios | (0.5, 0.75, 1.25) |

### 5c. YOLOv8 — `3_Model/train_yolo.py`

**Input:** `Data/lars_processed/` (COCO JSON)
**Output:** `Data/lars_yolo/` (YOLO-format labels, symlinked images) + `runs/yolo/baseline/`

On first run, `prepare_yolo_dataset()` converts COCO annotations to YOLO txt format
and writes `Data/lars_yolo/data.yaml`. Images are **symlinked** (not copied).

Category mapping (COCO IDs → 0-indexed YOLO):
`11→0, 12→1, 13→2, 14→3, 15→4, 16→5, 17→6, 19→7`

- Training delegated to `ultralytics.YOLO.train()` (built-in AMP, mixed precision)
- Ultralytics `results.csv` converted to project-standard `metrics.csv`

Defaults:

| Param | Default |
|---|---|
| epochs | 100 |
| batch_size | 8 |
| imgsz | 800 |
| lr0 / lrf | 1e-3 / 0.01 |
| momentum | 0.937 |
| weight_decay | 5e-4 |
| warmup_epochs | 3 |
| box / cls | 7.5 / 0.5 |
| patience | 15 |

---

## 6. Hyperparameter Search — Optuna

All three search scripts share the same structure:
- TPE sampler (multivariate, seed=4) stored in SQLite (`optuna_study.db`)
- Resumable: pass `--study-name` to continue an existing study
- `--dry-run` to sample without training; `--smoke` for a 2-epoch sanity check
- Best trial checkpoints and per-trial `metrics.csv` saved per `trial_NNN/`
- Summary written to `trials_summary.csv` at the end

### 6a. RF-DETR — `3_Model/optuna_search_rfdetr.py`

**Active study dir:** `runs/rfdetr/optuna_relabeled_lb/`

Training HPs sampled:
- `lr` ∈ [1e-5, 5e-4] (log)
- `lr_enc_ratio` ∈ [0.05, 0.25] (log)
- `weight_decay` ∈ [1e-5, 5e-4] (log)
- `grad_clip_max_norm` ∈ [0.05, 0.50] (log)
- `resolution` ∈ {560, 616, 672, 728, 784}
- `batch_size` ∈ {8, 12, 16, 24}
- `model_variant` ∈ {base, large}

Augmentation:
- `aug_copies` ∈ [0, 3]
- `aug_policy` sampled from a fixed `AUG_POLICIES` catalogue

### 6b. Faster R-CNN — `3_Model/optuna_search_fasterrcnn.py`

**Active study dir:** `runs/fasterrcnn/optuna/`

Training HPs sampled:
- `lr` ∈ [5e-4, 1e-2] (log)
- `lr_backbone_ratio` ∈ [0.05, 0.25] (log)
- `weight_decay` ∈ [1e-5, 1e-3] (log)
- `momentum` ∈ {0.85, 0.9, 0.95}
- `batch_size` ∈ {8, 16}
- `step_size` ∈ {10, 15, 20}, `gamma` ∈ {0.05, 0.1, 0.2}

Augmentation: `aug_policy` sampled from `AUG_POLICIES` (online albumentations applied in
`AugLARSDataset.__getitem__`, no disk writes). `model_variant` is **fixed to `base`**.

### 6c. YOLOv8 — `3_Model/optuna_search_yolo.py`

**Active study dir:** `runs/yolo/optuna/`

Training HPs sampled:
- `model_variant`, `imgsz`, `batch_size` (categorical)
- `lr0` ∈ [1e-4, 1e-2] (log), `lrf` ∈ [1e-3, 0.1] (log)
- `momentum` ∈ [0.70, 0.99], `weight_decay` ∈ [1e-5, 1e-3] (log)
- `warmup_epochs` ∈ [1, 5], `box` ∈ [4.0, 12.0], `cls` ∈ [0.3, 3.0]

Augmentation HPs (passed directly to `model.train()`):
- `hsv_h/s/v`, `degrees`, `translate`, `scale`, `fliplr`, `mosaic`, `mixup`

The standalone manual run `runs/yolo/exp7_yolov8m_ep100_img1024_b4/` (yolov8m,
100 epochs, imgsz=1024, batch=4) is the canonical YOLO checkpoint for evaluation,
not an Optuna trial.

---

## 7. Prediction Export — `3_Model/export_predictions_*.py`

One script per detector. Each loads the trained checkpoint, runs inference at a
permissive threshold (evaluation re-filters), and writes COCO results JSON.

```bash
python 3_Model/export_predictions_rfdetr.py \
  --checkpoint runs/rfdetr/optuna_relabeled_lb/trial_004/checkpoint_best_total.pth \
  --split test --resolution 728

python 3_Model/export_predictions_fasterrcnn.py \
  --checkpoint runs/fasterrcnn/optuna/trial_002/checkpoint_best.pth \
  --split test --model base

python 3_Model/export_predictions_yolo.py \
  --weights runs/yolo/exp7_yolov8m_ep100_img1024_b4/weights/best.pt \
  --split test
```

Output: `<checkpoint_dir>/predictions_<split>.json` (COCO results format).

---

## 8. Evaluation

### Canonical: `3_Model/evaluate_all_models.ipynb`
Compares all three detectors side-by-side on the test set. Loads each checkpoint,
runs inference, computes:
- mAP@50, mAP@75, mAP@50:95 (via `supervision.metrics.MeanAveragePrecision`)
- Class-agnostic mAP (object vs. background)
- Per-class precision / recall / F1 at confidence threshold sweep
- Confusion matrices (counts + row-normalised, all-models combined)

Saves the final report-grade figures to `report/images/`:
`confusion_matrix_rownorm_all_models.png`, `fp_fn_crops_*.png`,
`size_distribution_*.png`, `wrong_class_*.png`.

### Headless: `3_Model/eval.py`
Same metric logic, but command-line driven for quick re-evaluation:
```bash
python 3_Model/eval.py <predictions.json> <model_id>
```
Appends a row to `runs/model_results.csv` with mAP@50, mAP@50:95, mAP@75,
mAP@50_agnostic, P/R/F1 at best-F1 threshold, and per-class AP@50.

### Per-trial figures: `3_Model/evaluation.ipynb`
Single-model deep dive (FP/FN crops, location grid, threshold sweep, scene-condition
breakdown). Outputs land alongside the trial checkpoint
(e.g. `runs/fasterrcnn/optuna/trial_002/*.png`).

### Training curves: `3_Model/plot_training_loss.py`
Generates loss curves for all three detectors from each run's `metrics.csv`
and writes PNGs into `runs/training_curves/`. For Faster R-CNN (no per-epoch
val-loss column) val mAP@50:95 is plotted on a twin axis instead.

---

## 9. Results (test set, `runs/model_results.csv`)

| Model (canonical run) | mAP@50:95 | mAP@50 | F1 (best thresh) | FPS |
|---|---:|---:|---:|---:|
| **RF-DETR** (`rfdetr/optuna_relabeled_lb/trial_004`) | **0.3409** | **0.5857** | 0.738 (@0.40) | 27.2 |
| Faster R-CNN (`fasterrcnn/optuna/trial_002`) | 0.2353 | 0.4364 | 0.645 (@0.70) | 20.3 |
| YOLOv8m (`yolo/exp7_yolov8m_ep100_img1024_b4`) | 0.2339 | 0.4221 | 0.650 (@0.30) | **49.8** |

- **RF-DETR** is the most accurate by every overall metric.
- **YOLOv8m** is roughly 2× the throughput at comparable mAP@50:95 to Faster R-CNN.
- Per-class AP@50 columns and additional trials (e.g. tuned-anchor Faster R-CNN,
  `rfdetr_rfs` random-feature-search) are kept in `runs/model_results.csv` for
  reference.

Checkpoints for the three canonical runs are surfaced as symlinks under
`model_showcase/` (`fasterrcnn_trial002.pth`, `rfdetr_trial004.pth`, `yolo_exp7.pt`).

---

## 10. Explainability — `5_XAI/XAI.ipynb`

Visualisations of where each detector is "looking" when it predicts a boat, buoy,
swimmer, etc. Four pipelines, all driven from the same notebook:

| Method | Models | What it shows |
|---|---|---|
| **D-RISE** | Faster R-CNN, RF-DETR, YOLO | Black-box saliency by random-mask occlusion. Score = match between prediction on masked vs. original image. |
| **Grad-CAM++** | Faster R-CNN | Gradient-based saliency on `roi_heads.box_head[3]` RoI features. |
| **EigenCAM** | YOLO | Gradient-free PCA on layer-15 features, masked to the predicted box. |
| **RF-DETR cross-attention** | RF-DETR | Scatter of the deformable attention's sampling locations from the last decoder layer (`core.transformer.decoder.layers[-1].cross_attn`). |

Outputs are written to `5_XAI/plots/`; a curated subset is referenced from
`report/images/report.tex`.

---

## 11. Inference / Demo

`model_showcase/demo_load_models.ipynb` loads all three canonical checkpoints
through symlinks and runs inference on a sample image, demonstrating how to use
each detector from a single notebook. `model_showcase/` is intentionally untracked
(checkpoints are large binaries handled outside git).
