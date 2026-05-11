# Known Issues & Suspicious Findings

---

### ⚠️ cv_relabel uses Trial-004 from the *first* optuna study, not the best one

`cv_relabel.py` hardcodes `TRIAL004` (lr=4.39e-4, resolution=672) which matches
**trial 4 of the original `optuna` study** (mAP=0.2678 — the 2nd best in that study).
The actual best trial in that study was **trial 10** (mAP=0.3282, lr=1.09e-4, res=728).
The cv_relabeling was therefore run with sub-optimal hyperparameters. The better
`optuna_relabeled_lb` study came *after* cv_relabeling.

---

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

---

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
| `lars_processed/test/_annotations.coco.json` (active) | 503 | 1,605 | 751 (106 added, 645 kept) |
| `lars_relabeled/test/_annotations.coco.json` | 503 | 1,605 | 751 — identical to active |

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

---

### ⚠️ Faster R-CNN `large` variant has cold-start detection heads

`build_model("large", ...)` in `train_fasterrcnn.py` loads **ImageNet pretrained
backbone only**; the RPN and RoI heads are randomly initialised. The `base` variant
loads a fully **COCO pretrained** model (backbone + RPN + RoI heads) and only replaces
the final predictor. The optuna search originally offered `["base", "large"]` as a
choice — the `large` trials consistently failed to converge within 25 epochs (mAP <
0.01), which caused the study to be deleted and `model_variant` was subsequently
fixed to `base`.

---

### ⚠️ Faster R-CNN optuna batch=24 caused repeated OOMs

Three out of the first five trials sampled `batch_size=24`, each crashing with CUDA
OOM on the 44 GB A40. This was because the previous trial's model (especially `large`)
had not yet released GPU memory. `batch_size=24` was removed from the search space
(now `[8, 16]`) but this only takes effect for new studies — it did not retroactively
prevent the crashes in the deleted study.

---

### ⚠️ YOLO label cache is not invalidated on annotation change

`prepare_yolo_dataset()` in `train_yolo.py` checks only whether `Data/lars_yolo/data.yaml`
exists; if it does, conversion is skipped entirely. If `lars_processed/_annotations.coco.json`
is later updated (e.g., after another round of cv_relabeling), the YOLO txt labels
will be **stale**. Delete `Data/lars_yolo/` manually to force regeneration.
