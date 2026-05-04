"""
Optuna hyperparameter + augmentation search for RF-DETR on the LARS maritime dataset.

Each trial:
  1. Samples hyperparameters  (lr, lr_encoder ratio, weight_decay, resolution, …)
  2. Samples augmentation config (pipeline transforms + copies-per-image)
  3. Undoes any previous offline augmentation, applies the new config
  4. Trains RF-DETR for --epochs epochs with early stopping
  5. Reads best val/mAP_50_95 from metrics.csv and reports it to Optuna
  6. After all trials: restores original dataset, saves trials_summary.csv

Dataset context:
  2,102 training images / 7,761 objects — small enough that heavy augmentation
  is strongly beneficial.  Class imbalance is severe (floats: 19, paddle boards:
  124) so aug_copies=2-3 can meaningfully help rare classes.  Resolution is kept
  at ≥560 because lower values hurt small-object recall.

Why 40 epochs by default?
  A full run is ~100 epochs ≈ 30 min.  40 epochs ≈ 12 min, giving room for
  ~10 trials in ~2 hours.  Increase --epochs for final validation of top configs.

Usage
-----
    # 10 TPE trials, 40 epochs each (default)
    python optuna_search.py

    # Random search, more trials, longer budget per trial
    python optuna_search.py --sampler random --n-trials 20 --epochs 60

    # Resume an interrupted study (reads existing DB, skips completed trials)
    python optuna_search.py --study-name rfdetr_lars --n-trials 5

    # Dry-run: print suggested params without training
    python optuna_search.py --dry-run
"""

import argparse
import json
import logging
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler, RandomSampler
import albumentations as A
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).parent
DATA_ROOT = _HERE / "../Data/lars_processed"
TRAIN_DIR = DATA_ROOT / "train"
IMG_DIR   = TRAIN_DIR / "images"
ANN_FILE  = TRAIN_DIR / "_annotations.coco.json"
ANN_BACKUP = TRAIN_DIR / "_annotations.coco.original.json"
AUG_SUFFIX = "_aug"   # must match rfdetr_preprocessing.py


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def build_aug_pipeline(trial: optuna.Trial) -> A.Compose:
    """
    Build an albumentations Compose pipeline from Optuna trial suggestions.

    Context: 2,102 training images, 7,761 annotations with heavy class imbalance
    (boats: 4,957 | buoys: 1,274 | floats: 19 | paddle boards: 124).
    Heavy augmentation is warranted to prevent overfitting on the small dataset
    and to improve coverage of rare classes and challenging conditions.

    Transforms are organised into five groups:
      1. Geometric        — always applied, tunable probability
      2. Colour           — always applied, tunable intensity
      3. Contrast tools   — optional (CLAHE, gamma, sharpening, equalise)
      4. Blur / noise     — optional (gaussian, motion blur, ISO noise, compression)
      5. Maritime weather — optional (shadow, fog, rain, sun flare)
      6. Occlusion        — optional (coarse dropout for spray / partial cover)
    """
    transforms: list = []

    # ── 1. Geometric ─────────────────────────────────────────────────────────
    flip_p = trial.suggest_float("aug_flip_p", 0.3, 0.7)
    transforms.append(A.HorizontalFlip(p=flip_p))

    # Subtle perspective shift — simulates camera pitch/roll on a moving vessel
    if trial.suggest_categorical("aug_use_perspective", [True, False]):
        persp_scale = trial.suggest_float("aug_perspective_scale", 0.02, 0.08)
        transforms.append(A.Perspective(scale=(0.01, persp_scale), keep_size=True, p=0.35))

    # ── 2. Colour ────────────────────────────────────────────────────────────
    brightness_limit = trial.suggest_float("aug_brightness_limit", 0.10, 0.45)
    contrast_limit   = trial.suggest_float("aug_contrast_limit",   0.10, 0.45)
    brightness_p     = trial.suggest_float("aug_brightness_p",     0.40, 0.90)
    transforms.append(A.RandomBrightnessContrast(
        brightness_limit=brightness_limit,
        contrast_limit=contrast_limit,
        p=brightness_p,
    ))

    hsv_hue = trial.suggest_int  ("aug_hsv_hue_limit", 2,  20)
    hsv_sat = trial.suggest_int  ("aug_hsv_sat_limit", 20, 70)
    hsv_val = trial.suggest_int  ("aug_hsv_val_limit", 15, 50)
    hsv_p   = trial.suggest_float("aug_hsv_p",         0.30, 0.85)
    transforms.append(A.HueSaturationValue(
        hue_shift_limit=hsv_hue,
        sat_shift_limit=hsv_sat,
        val_shift_limit=hsv_val,
        p=hsv_p,
    ))

    # Gamma — simulates exposure variation (overcast dawn vs bright noon)
    if trial.suggest_categorical("aug_use_gamma", [True, False]):
        gamma_p = trial.suggest_float("aug_gamma_p", 0.20, 0.60)
        transforms.append(A.RandomGamma(gamma_limit=(70, 130), p=gamma_p))

    # ── 3. Contrast tools ────────────────────────────────────────────────────

    # CLAHE — local contrast enhancement, combats glare and reflections on water
    clahe_p = trial.suggest_float("aug_clahe_p", 0.0, 0.55)
    if clahe_p > 0:
        transforms.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=clahe_p))

    # Histogram equalisation — alternative global contrast boost
    if trial.suggest_categorical("aug_use_equalize", [True, False]):
        transforms.append(A.Equalize(p=0.25))

    # Sharpening — counter soft focus / spray haze
    if trial.suggest_categorical("aug_use_sharpen", [True, False]):
        transforms.append(A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.30))

    # Downscale + upsample — simulates lower-resolution sensor or distant targets
    if trial.suggest_categorical("aug_use_downscale", [True, False]):
        transforms.append(A.Downscale(scale_range=(0.6, 0.9), p=0.20))

    # ── 4. Blur / noise ──────────────────────────────────────────────────────

    # Blur type — sea haze, spray, camera motion from vessel movement
    blur_type = trial.suggest_categorical("aug_blur_type", ["none", "gaussian", "motion"])
    if blur_type != "none":
        blur_p = trial.suggest_float("aug_blur_p", 0.10, 0.55)
        if blur_type == "motion":
            transforms.append(A.MotionBlur(blur_limit=(3, 11), p=blur_p))
        else:
            transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=blur_p))

    # ISO / sensor noise — low-light, high-ISO night operations
    if trial.suggest_categorical("aug_use_iso_noise", [True, False]):
        transforms.append(A.ISONoise(
            color_shift=(0.01, 0.05),
            intensity=(0.10, 0.40),
            p=0.30,
        ))

    # JPEG compression artifacts — common in onboard streaming video
    if trial.suggest_categorical("aug_use_compression", [True, False]):
        transforms.append(A.ImageCompression(quality_range=(50, 90), p=0.25))

    # ── 5. Maritime weather ──────────────────────────────────────────────────

    # Shadow — partial cloud cover or superstructure shadow across the deck
    if trial.suggest_categorical("aug_use_shadow", [True, False]):
        transforms.append(A.RandomShadow(p=0.35))

    # Fog — morning sea mist, heavy spray, reduced visibility
    if trial.suggest_categorical("aug_use_fog", [True, False]):
        fog_coef = trial.suggest_float("aug_fog_coef_max", 0.10, 0.35)
        transforms.append(A.RandomFog(fog_coef_range=(0.05, fog_coef), p=0.30))

    # Rain — sea spray, precipitation  (drop_length kept short for spray effect)
    if trial.suggest_categorical("aug_use_rain", [True, False]):
        transforms.append(A.RandomRain(
            slant_range=(-10, 10),
            drop_length=8,
            drop_width=1,
            brightness_coefficient=0.9,
            p=0.25,
        ))

    # Sun flare — direct sun reflection on water surface (extremely common)
    if trial.suggest_categorical("aug_use_sunflare", [True, False]):
        transforms.append(A.RandomSunFlare(
            flare_roi=(0.0, 0.0, 1.0, 0.5),   # upper half only (sky/horizon)
            src_radius=100,
            p=0.20,
        ))

    # ── 6. Occlusion ─────────────────────────────────────────────────────────

    # Coarse dropout — wave crests / spray partially occlude distant targets
    if trial.suggest_categorical("aug_use_dropout", [True, False]):
        n_holes = trial.suggest_int("aug_dropout_holes", 1, 10)
        transforms.append(A.CoarseDropout(
            num_holes_range=(1, n_holes),
            hole_height_range=(16, 48),
            hole_width_range=(16, 48),
            fill=0,
            p=0.30,
        ))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_area=50.0,
            min_visibility=0.1,
        ),
    )


def safe_undo() -> None:
    """Remove augmented images and restore the backup annotation file.
    Does not sys.exit — safe to call inside a long-running process."""
    if not ANN_BACKUP.exists():
        return  # nothing to undo

    with open(ANN_BACKUP) as f:
        orig_ann = json.load(f)

    deleted = 0
    for jpg in IMG_DIR.glob("*.jpg"):
        if AUG_SUFFIX in jpg.name:
            jpg.unlink()
            deleted += 1

    shutil.copy2(ANN_BACKUP, ANN_FILE)
    ANN_BACKUP.unlink()

    print(f"[undo] Removed {deleted} augmented images; annotations restored "
          f"({len(orig_ann['images'])} images).")


def apply_augmentation(pipeline: A.Compose, copies: int, seed: int = 4) -> tuple[int, int]:
    """
    Write offline augmented copies of every training image using *pipeline*.

    Returns (n_new_images, n_new_annotations).
    Raises RuntimeError if augmented data already exists (call safe_undo first).
    """
    random.seed(seed)
    np.random.seed(seed)

    with open(ANN_FILE) as f:
        ann = json.load(f)

    if any(AUG_SUFFIX in img["file_name"] for img in ann["images"]):
        raise RuntimeError(
            "Augmented data already present in annotation file. "
            "Call safe_undo() before re-augmenting."
        )

    anns_by_img: dict[int, list] = defaultdict(list)
    for a in ann["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    next_img_id = max(img["id"] for img in ann["images"]) + 1
    next_ann_id = max((a["id"] for a in ann["annotations"]), default=0) + 1

    new_images: list[dict] = []
    new_anns:   list[dict] = []
    skipped = 0

    for img_entry in tqdm(ann["images"], desc="Augmenting", leave=False):
        img_path = IMG_DIR / Path(img_entry["file_name"]).name
        if not img_path.exists():
            skipped += 1
            continue

        img_arr  = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img_arr.shape[:2]
        img_anns = anns_by_img.get(img_entry["id"], [])
        # Clip to image bounds — guards against sub-pixel float drift in annotations
        bboxes, cat_ids = [], []
        for a in img_anns:
            x, y, bw, bh = a["bbox"]
            x  = max(0.0, x);  y  = max(0.0, y)
            bw = min(bw, img_w - x);  bh = min(bh, img_h - y)
            if bw > 0 and bh > 0:
                bboxes.append([x, y, bw, bh])
                cat_ids.append(a["category_id"])

        for copy_idx in range(copies):
            result     = pipeline(image=img_arr, bboxes=bboxes, category_ids=cat_ids)
            aug_arr    = result["image"]
            aug_bboxes = result["bboxes"]
            aug_cats   = result["category_ids"]

            stem      = Path(img_entry["file_name"]).stem
            aug_fname = f"images/{stem}{AUG_SUFFIX}{copy_idx}.jpg"
            aug_path  = IMG_DIR / f"{stem}{AUG_SUFFIX}{copy_idx}.jpg"

            Image.fromarray(aug_arr).save(aug_path, quality=95)

            new_images.append({
                "id":        next_img_id,
                "width":     img_entry["width"],
                "height":    img_entry["height"],
                "file_name": aug_fname,
            })

            for bbox, cat_id in zip(aug_bboxes, aug_cats):
                x, y, w, h = bbox
                new_anns.append({
                    "id":          next_ann_id,
                    "image_id":    next_img_id,
                    "category_id": int(cat_id),
                    "bbox":        [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                    "area":        round(w * h, 1),
                    "iscrowd":     0,
                })
                next_ann_id += 1

            next_img_id += 1

    # Back up original annotations (once)
    if not ANN_BACKUP.exists():
        shutil.copy2(ANN_FILE, ANN_BACKUP)

    ann["images"]      = ann["images"]      + new_images
    ann["annotations"] = ann["annotations"] + new_anns

    with open(ANN_FILE, "w") as f:
        json.dump(ann, f)

    if skipped:
        print(f"[aug] Warning: {skipped} images not found on disk and were skipped.")

    return len(new_images), len(new_anns)


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def read_best_map(metrics_csv: Path) -> float | None:
    """Return best val/mAP_50_95 from a metrics.csv, or None on failure."""
    if not metrics_csv.exists():
        return None
    try:
        df     = pd.read_csv(metrics_csv)
        val_df = df.dropna(subset=["val/mAP_50_95"])
        if val_df.empty:
            return None
        return float(val_df["val/mAP_50_95"].max())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(
    trial:   optuna.Trial,
    args:    argparse.Namespace,
    logger:  logging.Logger,
) -> float:

    trial_dir = Path(args.study_dir) / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ── Sample training hyperparameters ──────────────────────────────────────
    lr       = trial.suggest_float("lr",           1e-5, 5e-4, log=True)
    # Express lr_encoder as a fraction of lr (keeps ratio meaningful)
    lr_ratio = trial.suggest_float("lr_enc_ratio", 0.05, 0.25, log=True)
    lr_encoder = lr * lr_ratio
    trial.set_user_attr("lr_encoder", round(lr_encoder, 8))

    weight_decay       = trial.suggest_float("weight_decay",       1e-5, 5e-4, log=True)
    grad_clip_max_norm = trial.suggest_float("grad_clip_max_norm", 0.05, 0.50, log=True)
    # 560 is the minimum — lower resolutions hurt small object detection
    # (buoys, swimmers, paddle boards at range) and the model is already large
    res_choices  = [560] if args.smoke else [560, 616, 672, 728, 784]
    resolution   = trial.suggest_categorical("resolution", res_choices)
    batch_size   = trial.suggest_categorical("batch_size",  [16] if args.smoke else [8,12,16,24])

    # ── Sample augmentation ───────────────────────────────────────────────────
    # 2,102 training images is small — up to 3 copies (4× dataset) is reasonable
    aug_copies = 0 if args.smoke else trial.suggest_int("aug_copies", 0, 3)

    logger.info("")
    logger.info("=" * 64)
    logger.info(f"Trial {trial.number:3d}  →  {trial_dir.name}")
    logger.info(f"  lr={lr:.2e}  lr_encoder={lr_encoder:.2e}  wd={weight_decay:.2e}")
    logger.info(f"  resolution={resolution}  batch={batch_size}  "
                f"grad_clip={grad_clip_max_norm:.3f}")
    logger.info(f"  aug_copies={aug_copies}")
    logger.info("=" * 64)

    t_trial_start = time.time()

    if args.dry_run:
        logger.info("[DRY RUN] Skipping augmentation and training.")
        import random as _r
        return _r.uniform(0.1, 0.5)   # dummy value

    # ── Reset augmentation from previous trial ────────────────────────────────
    if ANN_BACKUP.exists():
        logger.info("Undoing previous augmentation …")
        safe_undo()

    # ── Apply new augmentation ────────────────────────────────────────────────
    t_aug_start = time.time()
    if aug_copies > 0:
        pipeline = build_aug_pipeline(trial)
        logger.info(f"Applying augmentation ({aug_copies} cop{'y' if aug_copies == 1 else 'ies'} per image) …")
        n_imgs, n_anns = apply_augmentation(pipeline, copies=aug_copies, seed=4)
        aug_elapsed = time.time() - t_aug_start
        logger.info(f"  +{n_imgs:,} images  +{n_anns:,} annotations  ({aug_elapsed:.0f}s)")
        trial.set_user_attr("aug_time_s", round(aug_elapsed, 1))
    else:
        logger.info("No augmentation for this trial (aug_copies=0).")
        trial.set_user_attr("aug_time_s", 0.0)

    # ── Train ─────────────────────────────────────────────────────────────────
    from rfdetr import RFDETRBase

    model = RFDETRBase(resolution=resolution)

    # Only save a checkpoint at the very end to save time / disk
    t_train_start = time.time()
    try:
        model.train(
            dataset_dir              = str(DATA_ROOT),
            epochs                   = args.epochs,
            batch_size               = batch_size,
            grad_accum_steps         = 4,
            lr                       = lr,
            lr_encoder               = lr_encoder,
            resolution               = resolution,
            weight_decay             = weight_decay,
            grad_clip_max_norm       = grad_clip_max_norm,
            checkpoint_interval      = args.epochs,
            output_dir               = str(trial_dir),
            early_stopping           = True,
            early_stopping_patience  = 8,
            early_stopping_min_delta = 0.001,
            early_stopping_use_ema   = False,
        )
    except Exception as exc:
        logger.error(f"Trial {trial.number} training raised an exception: {exc}")
        raise  # Optuna's catch=(Exception,) will mark this trial as FAIL

    train_elapsed = time.time() - t_train_start
    trial_elapsed = time.time() - t_trial_start
    trial.set_user_attr("train_time_s", round(train_elapsed, 1))
    trial.set_user_attr("trial_time_s", round(trial_elapsed, 1))

    th, tr = divmod(int(train_elapsed), 3600)
    tm, ts = divmod(tr, 60)
    logger.info(f"  Training time : {th:02d}:{tm:02d}:{ts:02d}  ({train_elapsed/60:.1f} min)")

    # ── Read result ───────────────────────────────────────────────────────────
    best_map = read_best_map(trial_dir / "metrics.csv")
    if best_map is None:
        logger.warning(f"Trial {trial.number}: no valid mAP found in metrics.csv.")
        raise optuna.exceptions.TrialPruned()

    logger.info(f"Trial {trial.number} best val/mAP_50_95 = {best_map:.4f}  "
                f"(trial total: {trial_elapsed/60:.1f} min)")
    return best_map


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HP + augmentation search for RF-DETR on the LARS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-trials",   type=int,   default=10,
                   help="Number of Optuna trials to run")
    p.add_argument("--epochs",     type=int,   default=40,
                   help="Training epochs per trial (40 ≈ 12 min; raise for accuracy)")
    p.add_argument("--study-name", type=str,   default="rfdetr_lars",
                   help="Optuna study name — used as key in the SQLite DB. "
                        "Pass the same name to resume a study.")
    p.add_argument("--study-dir",  type=str,
                   default=str(_HERE / "../runs/optuna"),
                   help="Root directory for per-trial checkpoints and logs")
    p.add_argument("--data-root",  type=str,   default=None,
                   help="Override dataset root (default: Data/lars_processed). "
                        "Pass e.g. Data/lars_relabeled to use cleaned labels.")
    p.add_argument("--sampler",    choices=["tpe", "random"], default="tpe",
                   help="Optuna sampler: tpe (Bayesian, recommended) or random")
    p.add_argument("--timeout",    type=int,   default=None,
                   help="Hard wall-clock limit in seconds (study stops after this)")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print sampled hyperparameters without actually training")
    p.add_argument("--smoke",      action="store_true",
                   help="Smoke-test mode: fixes resolution=560 and batch_size=4 for fast runs")
    return p.parse_args()


def setup_logging(study_dir: Path) -> logging.Logger:
    study_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s  %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(study_dir / "search.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger = logging.getLogger("optuna_search")
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def print_summary(
    study: optuna.Study,
    logger: logging.Logger,
    study_dir: Path,
    total_elapsed: float,
) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    if not completed:
        logger.info("No completed trials to summarise.")
        return

    logger.info("")
    logger.info("=" * 64)
    logger.info("SEARCH COMPLETE — results ranked by val/mAP_50_95")
    logger.info("=" * 64)

    rows = sorted(
        [
            (
                t.number,
                t.value,
                t.params,
                t.user_attrs.get("train_time_s", float("nan")),
                t.user_attrs.get("aug_time_s",   float("nan")),
                t.user_attrs.get("trial_time_s", float("nan")),
            )
            for t in completed
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    for rank, (num, val, params, train_s, aug_s, total_s) in enumerate(rows, 1):
        logger.info(
            f"  #{rank:2d}  trial={num:3d}  mAP={val:.4f}"
            f"  train={train_s/60:.1f}min  aug={aug_s/60:.1f}min  total={total_s/60:.1f}min"
        )
        for k, v in params.items():
            logger.info(f"         {k} = {v}")

    best = study.best_trial
    logger.info("")
    logger.info(f"Best trial : #{best.number}  mAP={best.value:.4f}")
    logger.info(f"  lr_encoder (derived) = {best.user_attrs.get('lr_encoder', 'N/A')}")
    avg_train = sum(
        t.user_attrs.get("train_time_s", 0) for t in completed
    ) / len(completed)
    logger.info(f"  Avg training time/trial : {avg_train/60:.1f} min")
    logger.info(f"  Completed / failed      : {len(completed)} / {len(failed)}")

    sh, sr = divmod(int(total_elapsed), 3600)
    sm, ss = divmod(sr, 60)
    logger.info(f"  Total search wall time  : {sh:02d}:{sm:02d}:{ss:02d}  ({total_elapsed/60:.1f} min)")
    logger.info("=" * 64)

    # Save full Optuna dataframe (includes user_attrs as columns)
    df = study.trials_dataframe()
    out = study_dir / "trials_summary.csv"
    df.to_csv(out, index=False)
    logger.info(f"Trials summary → {out}")


def main() -> None:
    args      = parse_args()
    study_dir = Path(args.study_dir)

    if args.data_root is not None:
        global DATA_ROOT, TRAIN_DIR, IMG_DIR, ANN_FILE, ANN_BACKUP
        DATA_ROOT  = (Path(args.data_root) if Path(args.data_root).is_absolute()
                      else (_HERE / ".." / args.data_root).resolve())
        TRAIN_DIR  = DATA_ROOT / "train"
        IMG_DIR    = TRAIN_DIR / "images"
        ANN_FILE   = TRAIN_DIR / "_annotations.coco.json"
        ANN_BACKUP = TRAIN_DIR / "_annotations.coco.original.json"

    logger    = setup_logging(study_dir)

    # Suppress noisy Optuna per-trial INFO lines; we handle logging ourselves
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Reproducibility
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4)
        torch.backends.cudnn.benchmark = True

    sampler = (
        TPESampler(seed=4, multivariate=True)
        if args.sampler == "tpe"
        else RandomSampler(seed=4)
    )

    storage  = f"sqlite:///{study_dir}/optuna_study.db"
    study    = optuna.create_study(
        study_name     = args.study_name,
        direction      = "maximize",
        sampler        = sampler,
        storage        = storage,
        load_if_exists = True,   # resume if study already exists in DB
    )

    existing = len([t for t in study.trials if
                    t.state in (optuna.trial.TrialState.COMPLETE,
                                optuna.trial.TrialState.RUNNING)])
    logger.info("=" * 64)
    logger.info("RF-DETR  —  Optuna Hyperparameter + Augmentation Search")
    logger.info("=" * 64)
    logger.info(f"Study       : {args.study_name}")
    logger.info(f"Storage     : {storage}")
    logger.info(f"Sampler     : {args.sampler.upper()}")
    logger.info(f"Trials      : {args.n_trials}  ({existing} already in DB)")
    logger.info(f"Epochs/trial: {args.epochs}")
    logger.info(f"Output root : {study_dir}")
    if args.dry_run:
        logger.info("DRY RUN — no training will occur")
    logger.info("=" * 64)

    t_search_start = time.time()
    try:
        study.optimize(
            lambda trial: objective(trial, args, logger),
            n_trials = args.n_trials,
            timeout  = args.timeout,
            catch    = (Exception,),   # mark as FAIL, not crash
        )
    except KeyboardInterrupt:
        logger.info("Search interrupted by user.")
    finally:
        # Always restore the clean dataset
        if ANN_BACKUP.exists():
            logger.info("Restoring original dataset (undoing last augmentation) …")
            safe_undo()

    total_elapsed = time.time() - t_search_start
    print_summary(study, logger, study_dir, total_elapsed)
    logger.info("Done.")


if __name__ == "__main__":
    main()
