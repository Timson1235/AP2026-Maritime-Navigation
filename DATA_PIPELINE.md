# Data & Training Pipeline

End-to-end reference for how data flows from the raw LARS dataset through
preprocessing, label cleaning, augmentation, model training, and evaluation.

---

## Overview

```
Raw LARS v1.0.0
    │
    ▼
datasplit.py           → Data/lars_processed/  (train / valid / test / test_unused)
    │
    ├──▶ [optional] rfdetr_preprocessing.py  (offline augmentation — RF-DETR only)
    │
    ▼
cv_relabel.py          → Data/lars_relabeled/  (cleaned annotations only, no images)
    │
    ▼  (manual cp)
Data/lars_processed/   ← active annotations replaced with cleaned versions
    │
    ├──▶ train_rfdetr.py        → runs/rfdetr/baseline/
    ├──▶ optuna_search_rfdetr.py → runs/rfdetr/optuna*/
    │
    ├──▶ train_fasterrcnn.py    → runs/fasterrcnn/baseline/
    ├──▶ optuna_search_fasterrcnn.py → runs/fasterrcnn/optuna/
    │
    ├──▶ train_yolo.py          → Data/lars_yolo/ (labels) + runs/yolo/baseline/
    ├──▶ optuna_search_yolo.py  → runs/yolo/optuna/
    │
    ▼
export_predictions_*.py  → runs/<model>/*/predictions_test.json
    │
    ▼
evaluation.ipynb         → confusion matrix, mAP, FP/FN analysis, scene analysis
```

---

## 1. Raw Dataset — `Data/lars_v1.0.0_*`

| Source path | Contents |
|---|---|
| `lars_v1.0.0_annotations/train/panoptic_annotations.json` | COCO panoptic (bbox + masks) for 2,605 images |
| `lars_v1.0.0_annotations/train/image_annotations.json` | Scene-level labels (scene_type, lighting, reflections, waves) |
| `lars_v1.0.0_annotations/val/` | Same format, 198 images |
| `lars_v1.0.0_annotations/test/` | Image-level labels only — **no bbox annotations** |
| `lars_v1.0.0_images/{train,val,test}/images/` | Raw JPEGs |

**11 segmentation categories**: 3 stuff (Water, Sky, Static Obstacle) + 8 thing classes
used for detection: Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float, Other.

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
| `train` | 80% of LARS train | ~2,102 | ✓ |
| `test` | 20% of LARS train | ~503 | ✓ |
| `valid` | LARS val (all) | 198 | ✓ |
| `test_unused` | LARS test (all) | 1,203 | ✗ (image-level only) |

> **Seed:** `random_state=4`

### Notable detail
`test` and `train` both come from the original LARS train split and have panoptic
bbox labels. `test_unused` (original LARS test) has no bbox labels and is **not used**
by any model training or evaluation script.

---

## 3. Label Cleaning — `4_Model/cv_relabel.py`

**Input:** `Data/lars_processed/{train, valid, test}/_annotations.coco.json`  
**Output:** `Data/lars_relabeled/{train, valid, test}/_annotations.coco.json` + `provenance.json`  
**Intermediate:** `runs/cv_relabel/fold_{0-3}/`

### What it does
Runs **4-fold cross-validation** to find and fix noisy annotations:

1. **Pools all three splits** (train + valid + test, ~2,803 images total) and divides
   them into 4 scene-level folds (same scene-grouping logic as `datasplit.py`).
2. **Per fold k:**
   - Trains `RFDETRBase` on folds 0–3 (excluding k) using hardcoded Trial-004
     hyperparameters (see below).
   - Runs inference at `threshold=0.0` on held-out fold k.
   - Applies three cleaning rules:
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

### Hardcoded hyperparameters (TRIAL004)
```python
lr=4.39e-4, lr_encoder=5.30e-5, weight_decay=4.49e-4,
resolution=672, batch_size=4, grad_accum_steps=4
```

### Applying cleaned labels
`lars_relabeled/` contains **annotation JSONs only** — no images are copied.
To activate them, copy over `lars_processed`:
```bash
cp Data/lars_relabeled/train/_annotations.coco.json Data/lars_processed/train/_annotations.coco.json
cp Data/lars_relabeled/valid/_annotations.coco.json Data/lars_processed/valid/_annotations.coco.json
```
Backups are kept as `_annotations.coco.json.pre-relabel` in `lars_processed/`.

> **Current state:** The active annotations in `lars_processed/` are the cv-relabeled
> versions (applied before the `optuna_relabeled_lb` search).

---

## 4. Offline Augmentation — `2_DataPreprocessing/rfdetr_preprocessing.py`

**Input:** `Data/lars_processed/train/`  
**Output:** Augmented JPEGs written into `Data/lars_processed/train/images/`;
`_annotations.coco.json` updated in-place (original backed up as
`_annotations.coco.original.json`).

### Purpose
RF-DETR-specific **offline** augmentation. Generates N copies of every training
image (default: 1 copy → 2× dataset). Augmented filenames carry the `_aug` suffix.

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

> ⚠️ **Conflict risk** — See Section 7 (Issues).

---

## 5. Model Training

### 5a. RF-DETR — `4_Model/train_rfdetr.py`

**Input:** `Data/lars_processed/` (COCO JSON, images flat inside split dirs)  
**Output:** `runs/rfdetr/baseline/`

- Model: `RFDETRBase` or `RFDETRLarge` (from `rfdetr` library)
- Default: 100 epochs, lr=1e-4, lr_encoder=1e-5, resolution=728, batch=4
- Separate LR for encoder (backbone) vs. decoder
- Logs to `training.log` and `metrics.csv`

### 5b. Faster R-CNN — `4_Model/train_fasterrcnn.py`

**Input:** `Data/lars_processed/` (COCO JSON)  
**Output:** `runs/fasterrcnn/baseline/`

Two model variants:
| Variant | Backbone | Detection heads | Notes |
|---|---|---|---|
| `base` | ResNet-50-FPN-v2 | **COCO pretrained** (head replaced for 8+1 classes) | Fast convergence |
| `large` | ResNet-101-FPN | **Randomly initialised** | Needs many more epochs |

- SGD with momentum, StepLR scheduler
- Evaluation via `supervision.metrics.MeanAveragePrecision`
- Default: 50 epochs, lr=5e-3, lr_backbone=5e-4, batch=4, patience=10

### 5c. YOLOv8 — `4_Model/train_yolo.py`

**Input:** `Data/lars_processed/` (COCO JSON)  
**Output:** `Data/lars_yolo/` (YOLO-format labels, symlinked images) + `runs/yolo/baseline/`

On first run, `prepare_yolo_dataset()` converts COCO annotations to YOLO txt format
and writes `Data/lars_yolo/data.yaml`. Images are **symlinked** (not copied).

Category mapping (COCO IDs → 0-indexed YOLO):
`11→0, 12→1, 13→2, 14→3, 15→4, 16→5, 17→6, 19→7`

- Training delegated to `ultralytics.YOLO.train()` (built-in AMP, mixed precision)
- Ultralytics `results.csv` converted to project-standard `metrics.csv`
- Default: 100 epochs, imgsz=800, batch=8, lr0=1e-3, patience=15

---

## 6. Hyperparameter Search — Optuna

All three search scripts follow the same structure:
- TPE sampler (multivariate, seed=4) stored in SQLite (`optuna_study.db`)
- Resumable: pass `--study-name` to continue an existing study
- `--dry-run` to sample without training; `--smoke` for a 2-epoch sanity check
- Best trial checkpoints and per-trial `metrics.csv` saved per `trial_NNN/`
- Summary written to `trials_summary.csv` at the end

### 6a. RF-DETR — `4_Model/optuna_search_rfdetr.py`

**Default study dir:** `runs/rfdetr/optuna/`  
**Epochs/trial:** 40 (≈12 min/trial on A40)

**Training HPs sampled:**
`lr` [1e-5, 5e-4], `lr_enc_ratio` [0.05, 0.25], `weight_decay` [1e-5, 5e-4],
`grad_clip_max_norm` [0.05, 0.50], `resolution` [560, 616, 672, 728, 784],
`batch_size` [8, 12, 16, 24], `model_variant` [base, large]

**Augmentation strategy: offline (per trial)**
Each trial writes N augmented image copies into `lars_processed/train/images/`
and updates `_annotations.coco.json`, trains, then **undoes** augmentation before
the next trial. `aug_copies` is sampled as int [0, 3].

**Best completed study:** `runs/rfdetr/optuna_relabeled_lb/`
- Trial 004: mAP@.5:.95 = **0.3790** (best)
- 10 trials run, dates: 2026-05-04 → 2026-05-05

### 6b. Faster R-CNN — `4_Model/optuna_search_fasterrcnn.py`

**Default study dir:** `runs/fasterrcnn/optuna/`  
**Epochs/trial:** 25 (≈25 min/trial on A40 at batch=8)

**Training HPs sampled:**
`lr` [5e-4, 1e-2], `lr_backbone_ratio` [0.05, 0.25], `weight_decay` [1e-5, 1e-3],
`momentum` [0.85, 0.9, 0.95], `batch_size` [8, 16], `step_size` [10, 15, 20],
`gamma` [0.05, 0.1, 0.2]

`model_variant` is **fixed to `base`** (COCO pretrained head) — the large variant
has random detection heads and cannot converge in 25 epochs.

**Augmentation strategy: online (per batch, albumentations)**  
The full maritime augmentation pipeline (flip, perspective, brightness/contrast,
HSV, CLAHE, blur, noise, fog, rain, sunflare, dropout) is applied live in
`AugLARSDataset.__getitem__`. No disk writes. `aug_enabled` is sampled as bool.

### 6c. YOLOv8 — `4_Model/optuna_search_yolo.py`

**Default study dir:** `runs/yolo/optuna/`  
**Epochs/trial:** 50 (≈50 min/trial on A40 at imgsz=800)

**Training HPs sampled:**
`lr0` [1e-4, 1e-2], `lrf` [1e-3, 0.1], `momentum` [0.70, 0.99],
`weight_decay` [1e-5, 1e-3], `warmup_epochs` [1, 5], `box` [4.0, 12.0],
`cls` [0.3, 3.0], `imgsz` [640, 800, 1024], `batch_size` [8, 16, 24],
`model_variant` [n, s, m]

**Augmentation strategy: YOLOv8 built-in**  
Parameters passed directly to `model.train()`:
`hsv_h/s/v`, `degrees`, `translate`, `scale`, `fliplr`, `mosaic`, `mixup`

---

## 7. Prediction Export

### RF-DETR — `4_Model/export_predictions_rfdetr.py`
```bash
python export_predictions_rfdetr.py \
  --checkpoint ../runs/rfdetr/optuna_relabeled_lb/trial_004/checkpoint_best_total.pth \
  --split test --resolution 728
```
- Loads `RFDETRBase` with the specified checkpoint
- Runs inference at threshold=0.0 (evaluation notebook re-filters)
- Output: `<checkpoint_dir>/predictions_<split>.json` (COCO results format)

### Faster R-CNN — `4_Model/export_predictions_fasterrcnn.py`
```bash
python export_predictions_fasterrcnn.py \
  --checkpoint ../runs/fasterrcnn/baseline/checkpoint_best.pth \
  --split test --model base
```
- Output: `<checkpoint_dir>/predictions_<split>.json`

---

## 8. Evaluation — `4_Model/evaluation.ipynb`

**Input:**
- `Data/lars_processed/test/_annotations.coco.json` (GT)
- `runs/fasterrcnn/baseline/predictions_test.json` (currently configured)
- `Data/lars_processed/test/images/`

**Metrics computed:**
- mAP@50, mAP@75, mAP@50:95 (via `supervision.metrics.MeanAveragePrecision`)
- Class-agnostic mAP (object vs. background)
- mAP excluding unreliable classes (currently `Float`)
- Per-class precision / recall / F1 at configurable confidence threshold
- Confusion matrix (counts + row-normalised)

**Analyses:**
- Wrong-class prediction crops
- FP / FN crop samples
- Object size distribution (log-scale, TP/FN/FP breakdown)
- Spatial 3×3 grid error heatmaps
- Scene condition breakdown (scene_type, lighting, reflections, waves, special flags)
  — requires `Data/lars_v1` image-level annotation JSONs

**Outputs saved to `SAVE_DIR` (currently `runs/fasterrcnn/baseline/`):**
`confusion_matrix.png`, `threshold_sweep.png`, `wrong_class.png`, `fp_fn_crops.png`,
`size_distribution.png`, `location_grid.png`, `scene_analysis.png`, `scene_special.png`

---

## 9. Results Summary

| Model | Study | Best mAP@.5:.95 | Notes |
|---|---|---|---|
| RF-DETR | `optuna_relabeled_lb` trial_004 | **0.3790** | Best overall |
| RF-DETR | baseline | ~0.27 (trial_004 from first optuna) | |
| Faster R-CNN | baseline (40 epochs) | **0.2539** | Default HPs, base model |
| Faster R-CNN | optuna (aborted) | 0.1686 best | Only 25-epoch budget; study deleted |
| YOLOv8 | — | not run yet | Scripts ready |

---

## 10. Issues & Suspicious Findings

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the full list.

### ⚠️ cv_relabel uses Trial-004 from the *first* optuna study, not the best one

`cv_relabel.py` hardcodes `TRIAL004` (lr=4.39e-4, resolution=672) which matches
**trial 4 of the original `optuna` study** (mAP=0.2678 — the 2nd best in that study).
The actual best trial in that study was **trial 10** (mAP=0.3282, lr=1.09e-4, res=728).
The cv_relabeling was therefore run with sub-optimal hyperparameters. The better
`optuna_relabeled_lb` study came *after* cv_relabeling.

### ⚠️ Offline augmentation conflict between `rfdetr_preprocessing.py` and `optuna_search_rfdetr.py`

Both scripts modify `lars_processed/train/_annotations.coco.json` and use the
**same backup filename** (`_annotations.coco.original.json`).

- If `rfdetr_preprocessing.py` is run first and then the optuna search starts,
  `apply_augmentation()` in the search script will raise a `RuntimeError` because
  it detects existing `_aug` entries in the annotation file.
- If `rfdetr_preprocessing.py` has been `--undo`-ed first, the backup file is deleted,
  which is fine — but the original annotations must be clean before starting the search.

**Rule:** never run `rfdetr_preprocessing.py` and `optuna_search_rfdetr.py` on the
same dataset state at the same time.

### ℹ️ Annotation file naming — verified state (2026-05-09)

Two backup naming conventions exist across scripts:

| Script | Backup filename |
|---|---|
| `rfdetr_preprocessing.py` | `_annotations.coco.original.json` |
| `optuna_search_rfdetr.py` | `_annotations.coco.original.json` |
| Manual (apply relabeled) | `_annotations.coco.json.pre-relabel` |

**Actual pipeline timeline (all verified against file contents):**

1. `datasplit.py` → writes `lars_processed/{train,valid,test}/_annotations.coco.json`
   (no `relabel_action` fields, no `_aug` images)
2. `cv_relabel.py` → writes `lars_relabeled/{train,valid,test}/_annotations.coco.json`
   and `lars_relabeled/provenance.json` (all annotations tagged with `relabel_action` where relevant)
3. Manual `cp` (train + valid only) → active train/valid files replaced; originals saved as
   `_annotations.coco.json.pre-relabel`. **Test split was never copied — see warning below.**
4. `optuna_search_rfdetr.py` — writes/deletes `_annotations.coco.original.json` transiently
   within each trial (undo always runs). No permanent `.original.json` exists on disk.

**`relabel_action` field semantics** (defined in `cv_relabel.py` docstring):

| Value | Meaning | Extra fields |
|---|---|---|
| `"added"` | New annotation inserted by the CV model (was a high-conf FP) | `relabel_conf` |
| `"kept"` | Original annotation that survived, but on an image where at least one other change was made | — |
| `"resized"` | Original annotation expanded to model's predicted extent | `relabel_conf`, `relabel_orig_bbox` |
| *(absent)* | Original annotation on an image the model made **no changes to** | — |
| *(gone)* | Removed ghost/merged box — simply absent from output | — |

**Current verified annotation counts:**

| File | Images | Anns | `relabel_action` present |
|---|---|---|---|
| `lars_processed/train/_annotations.coco.json` (active) | 2,102 | 8,025 | 4,211 (522 added, 3,689 kept) |
| `lars_processed/train/_annotations.coco.json.pre-relabel` | 2,102 | 7,761 | 0 — datasplit baseline |
| `lars_relabeled/train/_annotations.coco.json` | 2,102 | 8,025 | 4,211 — identical to active |
| `lars_processed/valid/_annotations.coco.json` (active) | 198 | 1,119 | 552 (49 added, 503 kept) |
| `lars_processed/valid/_annotations.coco.json.pre-relabel` | 198 | 1,107 | 0 — datasplit baseline |
| `lars_relabeled/valid/_annotations.coco.json` | 198 | 1,119 | 552 — identical to active |
| `lars_processed/test/_annotations.coco.json` (active) | 503 | 1,560 | **0 — still original, uncleaned** |
| `lars_relabeled/test/_annotations.coco.json` | 503 | 1,605 | 751 (106 added, 645 kept) |

**Net changes from cv_relabeling (from `provenance.json`):**

| Split | Modified images | Removed | Added | Net |
|---|---|---|---|---|
| train | 422 | 258 | 522 | +264 |
| valid | 46 | 37 | 49 | +12 |
| test | 101 | 61 | 106 | +45 |

No "resized" actions were triggered in any split. Offline augmentation was never
permanently applied — no `_annotations.coco.original.json` exists on disk.

> ~~⚠️ The test split's cleaned annotations were never applied.~~ **Fixed 2026-05-09.**
> `lars_processed/test/_annotations.coco.json` now holds the cv-relabeled version (1,605 anns).
> The CLAUDE.md apply-step instructions have been updated to include all three splits.

### ⚠️ Faster R-CNN `large` variant has cold-start detection heads

`build_model("large", ...)` in `train_fasterrcnn.py` loads **ImageNet pretrained
backbone only**; the RPN and RoI heads are randomly initialised. The `base` variant
loads a fully **COCO pretrained** model (backbone + RPN + RoI heads) and only replaces
the final predictor. The optuna search originally offered `["base", "large"]` as a
choice — the `large` trials consistently failed to converge within 25 epochs (mAP <
0.01), which caused the study to be deleted and `model_variant` was subsequently
fixed to `base`.

### ⚠️ Faster R-CNN optuna batch=24 caused repeated OOMs

Three out of the first five trials sampled `batch_size=24`, each crashing with CUDA
OOM on the 44 GB A40. This was because the previous trial's model (especially `large`)
had not yet released GPU memory. `batch_size=24` was removed from the search space
(now `[8, 16]`) but this only takes effect for new studies — it did not retroactively
prevent the crashes in the deleted study.

### ⚠️ YOLO label cache is not invalidated on annotation change

`prepare_yolo_dataset()` in `train_yolo.py` checks only whether `Data/lars_yolo/data.yaml`
exists; if it does, conversion is skipped entirely. If `lars_processed/_annotations.coco.json`
is later updated (e.g., after another round of cv_relabeling), the YOLO txt labels
will be **stale**. Delete `Data/lars_yolo/` manually to force regeneration.
