"""
RF-DETR fine-tuning script for the LARS maritime dataset.

Usage:
    python train_rfdetr.py
    python train_rfdetr.py --output-dir ../runs/rfdetr_v2 --epochs 80
    python train_rfdetr.py --resume                  # resume from latest checkpoint

    # Train the large model on relabeled data:
    python train_rfdetr.py --model large --data-root ../Data/lars_processed

    # MacBook Pro (MPS auto-detected by PyTorch Lightning):
    python train_rfdetr.py --mac

    # Force CPU (slow, for testing):
    python train_rfdetr.py --device cpu

Logs training progress to both console and <output-dir>/training.log.
Checkpoints and metrics.csv are saved to <output-dir>.
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from augmentations import AUG_POLICIES, get_maritime_augmentations
from optuna_search_rfdetr import (
    ANN_FILE, ANN_BACKUP, AUG_SUFFIX,
    apply_augmentation, safe_undo,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).parent
DATA_ROOT = _HERE / "../Data/lars_processed"

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # Mirror the Optuna best trial: rfdetr_relabeled_lb / trial_004
    # (val/mAP_50_95 = 0.379). trial_004 used aug_copies=2 with a heavy
    # multi-knob aug pipeline; closest single policy is
    # sensor_noise_and_occlusion, so that is the --aug-policy default.
    epochs               = 100,
    batch_size           = 4,
    grad_accum_steps     = 4,       # effective batch = 4 x 4 = 16 (trial_004)
    lr                   = 1.21e-4, # trial_004 best HP
    lr_encoder           = 2.81e-5, # trial_004 best HP
    resolution           = 784,     # base: multiple of 56 (784=56x14); large: multiple of 32 (e.g. 768)
    weight_decay         = 1.81e-5, # trial_004 best HP
    grad_clip_max_norm   = 0.083,   # trial_004 best HP
    aug_copies           = 2,       # trial_004 best HP (used when --aug-policy != none)
    checkpoint_interval  = 5,
    early_stopping_patience  = 10,
    early_stopping_min_delta = 0.001,
)


# ---------------------------------------------------------------------------
# Offline class-balanced oversampling  (LVIS-style Repeat Factor Sampling)
# ---------------------------------------------------------------------------
def apply_oversampling(t: float = 0.1, max_r: float = 10.0,
                       seed: int = 4) -> tuple[int, int]:
    """
    Mutate ``ANN_FILE`` in-place to duplicate image entries for rare-class
    images, COCO-side only — no disk image copies (each duplicate is a new
    images[] entry pointing at the same on-disk file_name).

      f_c = (# train images containing category c) / N
      r_c = max(1, sqrt(t / f_c))     (capped at max_r)
      r_i = max over c in image i of r_c
      extras = floor(r_i) - 1 + Bernoulli(r_i - floor(r_i))

    Stochastic rounding ensures Animal/Paddle/Swimmer (r_c ~ 1.4–1.8) also
    get a fractional boost rather than being silently floored to 0.

    Backs the original up to ``ANN_BACKUP`` if no backup exists yet.
    safe_undo() restores from that backup.

    Returns (n_added_images, n_added_annotations).
    """
    # Always read the current (possibly already-augmented) JSON.
    with open(ANN_FILE) as f:
        ann = json.load(f)
    # Preserve a backup of the *true* original — only write it once.
    if not ANN_BACKUP.exists():
        ANN_BACKUP.write_text(json.dumps(ann))

    imgs        = ann["images"]
    anns_by_img: dict[int, list] = defaultdict(list)
    for a in ann["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    N = len(imgs)
    cat_img_count: dict[int, int] = defaultdict(int)
    per_img_cats:  list[set[int]] = []
    for img in imgs:
        cats = {a["category_id"] for a in anns_by_img.get(img["id"], [])}
        per_img_cats.append(cats)
        for c in cats:
            cat_img_count[c] += 1

    r_c = {c: min(max_r, max(1.0, (t / (n / N)) ** 0.5))
           for c, n in cat_img_count.items()}

    next_img_id = max(img["id"] for img in imgs) + 1
    next_ann_id = max((a["id"] for a in ann["annotations"]), default=0) + 1

    rng = random.Random(seed)
    new_images: list[dict] = []
    new_anns:   list[dict] = []
    for img, cats in zip(imgs, per_img_cats):
        r_i   = max((r_c[c] for c in cats), default=1.0)
        floor = int(r_i)
        extra = floor - 1 + (1 if rng.random() < (r_i - floor) else 0)
        if extra <= 0:
            continue
        for _ in range(extra):
            new_img = {**img, "id": next_img_id}
            new_images.append(new_img)
            for a in anns_by_img.get(img["id"], []):
                new_anns.append({**a, "id": next_ann_id, "image_id": next_img_id})
                next_ann_id += 1
            next_img_id += 1

    ann["images"].extend(new_images)
    ann["annotations"].extend(new_anns)
    with open(ANN_FILE, "w") as f:
        json.dump(ann, f)

    return len(new_images), len(new_anns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train RF-DETR on the LARS maritime dataset")
    p.add_argument("--output-dir", default=str(_HERE / "../runs/rfdetr/baseline"),
                   help="Directory for checkpoints, metrics.csv, and training.log")
    p.add_argument("--epochs",     type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int,   default=DEFAULTS["batch_size"],
                   help="Per-GPU batch size (effective = batch-size × grad-accum-steps)")
    p.add_argument("--lr",         type=float, default=DEFAULTS["lr"],
                   help="Decoder/head learning rate")
    p.add_argument("--lr-encoder", type=float, default=DEFAULTS["lr_encoder"],
                   help="Backbone learning rate (keep ≤ 1/10 of --lr)")
    p.add_argument("--resolution", type=int,   default=DEFAULTS["resolution"],
                   choices=[448, 504, 560, 616, 672, 728, 768, 784, 800],
                   help="Input resolution. Base model: multiple of 56. Large model: multiple of 32 (e.g. 768, 800)")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from the latest checkpoint in --output-dir")
    p.add_argument("--no-early-stopping", action="store_true",
                   help="Disable early stopping and always train for --epochs epochs")
    p.add_argument("--mac", action="store_true",
                   help="Optimised defaults for MacBook Pro: resolution=560, batch-size=2, "
                        "num-workers=0. MPS is auto-detected by PyTorch Lightning.")
    p.add_argument("--device", default=None,
                   help="Force a specific device string, e.g. 'cpu'. "
                        "Leave unset to let PyTorch Lightning auto-select (CUDA / MPS / CPU).")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader worker processes (default: 2; use 0 on Mac to avoid spawn issues)")
    p.add_argument("--model", choices=["base", "large"], default="large",
                   help="RF-DETR variant: 'base' or 'large' (default, higher accuracy)")
    p.add_argument("--data-root", default=None,
                   help="Override dataset root (default: Data/lars_processed). "
                        "Pass e.g. ../Data/lars_processed to use a different split.")
    p.add_argument("--aug-policy", choices=list(AUG_POLICIES),
                   default="sensor_noise_and_occlusion",
                   help="Offline augmentation policy applied via aug_copies "
                        "image duplicates (default mirrors trial_004's heavy mix).")
    p.add_argument("--aug-copies", type=int, default=DEFAULTS["aug_copies"],
                   help="Augmented copies per training image (0 disables augmentation).")
    p.add_argument("--oversample", choices=["off", "rfs"], default="off",
                   help="Class-balanced oversampling via JSON duplicate entries. "
                        "rfs = LVIS Repeat Factor Sampling (t=0.1; Float ~3.3x, common 1x).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger = logging.getLogger("rfdetr_train")
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)

    global DATA_ROOT
    if args.data_root is not None:
        DATA_ROOT = (Path(args.data_root) if Path(args.data_root).is_absolute()
                     else (_HERE / ".." / args.data_root).resolve())

    logger     = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("RF-DETR  —  LARS Maritime Dataset Fine-tuning")
    logger.info("=" * 60)

    # Apply --mac preset before anything else
    if args.mac:
        if args.resolution == DEFAULTS["resolution"]:
            args.resolution = 560
        if args.batch_size == DEFAULTS["batch_size"]:
            args.batch_size = 2
        if args.num_workers is None:
            args.num_workers = 0

    # Fill num_workers default for non-Mac runs
    if args.num_workers is None:
        args.num_workers = 2

    # Reproducibility
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4)
        logger.info(f"Device        : CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Device        : MPS (Apple Silicon)")
    else:
        logger.info("Device        : CPU (training will be slow)")

    # Validate dataset
    if not DATA_ROOT.exists():
        logger.error(f"Dataset not found at {DATA_ROOT.resolve()}")
        logger.error("Run  2_DataPreprocessing/datasplit.py  first.")
        sys.exit(1)

    logger.info(f"Model variant : RF-DETR{args.model.capitalize()}")
    logger.info(f"Dataset root  : {DATA_ROOT.resolve()}")
    logger.info(f"Output dir    : {output_dir.resolve()}")

    # Log hyperparameters
    eff_batch = args.batch_size * DEFAULTS["grad_accum_steps"]
    logger.info("Hyperparameters:")
    logger.info(f"  num_workers         = {args.num_workers}")
    logger.info(f"  epochs              = {args.epochs}")
    logger.info(f"  batch_size          = {args.batch_size}")
    logger.info(f"  grad_accum_steps    = {DEFAULTS['grad_accum_steps']}")
    logger.info(f"  effective_batch     = {eff_batch}")
    logger.info(f"  lr                  = {args.lr}")
    logger.info(f"  lr_encoder          = {args.lr_encoder}")
    logger.info(f"  resolution          = {args.resolution}")
    logger.info(f"  weight_decay        = {DEFAULTS['weight_decay']}")
    logger.info(f"  grad_clip_max_norm  = {DEFAULTS['grad_clip_max_norm']}")
    logger.info(f"  checkpoint_interval = {DEFAULTS['checkpoint_interval']}")
    logger.info(f"  early_stopping      = {not args.no_early_stopping}"
                + (f" (patience={DEFAULTS['early_stopping_patience']}, "
                   f"min_delta={DEFAULTS['early_stopping_min_delta']})"
                   if not args.no_early_stopping else ""))

    # Resolve resume checkpoint
    resume_ckpt = None
    if args.resume:
        best = output_dir / "checkpoint_best_total.pth"
        if best.exists():
            resume_ckpt = best
        else:
            candidates = sorted(output_dir.glob("checkpoint*.pth"))
            resume_ckpt = candidates[-1] if candidates else None

        if resume_ckpt:
            logger.info(f"Resuming from : {resume_ckpt.name}")
        else:
            logger.warning("--resume set but no checkpoint found; starting from COCO pretrained weights")

    # Lazy import — fails fast on missing deps before training starts
    if args.model == "large":
        from rfdetr import RFDETRLarge
        model = RFDETRLarge(resolution=args.resolution)
    else:
        from rfdetr import RFDETRBase
        model = RFDETRBase(resolution=args.resolution)

    # ── Offline dataset ops: oversample + augmentation (mutate JSON) ─────────
    # RFDETR's trainer owns the DataLoader internally — no place to inject a
    # WeightedRandomSampler — so both ops happen by editing _annotations.coco.json
    # before training and restoring afterwards via safe_undo.
    if ANN_BACKUP.exists():
        logger.info("Found stale augmented JSON — undoing first …")
        safe_undo()

    try:
        # Augmentation must run before oversampling — otherwise duplicate JSON
        # entries (from oversampling) would race for the same aug filename and
        # overwrite each other.
        if args.aug_policy != "none" and args.aug_copies > 0:
            pipeline = get_maritime_augmentations(args.aug_policy)
            t_aug = time.time()
            logger.info(f"Augmentation    : policy={args.aug_policy}  copies={args.aug_copies} (offline) …")
            n_img, n_ann = apply_augmentation(pipeline, copies=args.aug_copies, seed=4)
            logger.info(f"  +{n_img:,} images  +{n_ann:,} annotations  ({time.time()-t_aug:.0f}s)")
        else:
            logger.info(f"Augmentation    : off  (aug_policy={args.aug_policy}, aug_copies={args.aug_copies})")

        if args.oversample == "rfs":
            n_img, n_ann = apply_oversampling(t=0.1)
            logger.info(f"Oversample (rfs): +{n_img} JSON entries, +{n_ann} annotations")
        else:
            logger.info("Oversample      : off")

        logger.info("Starting training …")
        t_train_start = time.time()
        train_kwargs = dict(
            dataset_dir              = str(DATA_ROOT),
            epochs                   = args.epochs,
            batch_size               = args.batch_size,
            grad_accum_steps         = DEFAULTS["grad_accum_steps"],
            lr                       = args.lr,
            lr_encoder               = args.lr_encoder,
            resolution               = args.resolution,
            weight_decay             = DEFAULTS["weight_decay"],
            grad_clip_max_norm       = DEFAULTS["grad_clip_max_norm"],
            checkpoint_interval      = DEFAULTS["checkpoint_interval"],
            output_dir               = str(output_dir),
            resume                   = str(resume_ckpt) if resume_ckpt else None,
            early_stopping           = not args.no_early_stopping,
            early_stopping_patience  = DEFAULTS["early_stopping_patience"],
            early_stopping_min_delta = DEFAULTS["early_stopping_min_delta"],
            early_stopping_use_ema   = False,
            num_workers              = args.num_workers,
        )
        if args.device:
            train_kwargs["device"] = args.device

        model.train(**train_kwargs)
        t_train_elapsed = time.time() - t_train_start
    finally:
        # Always restore the original JSON + delete augmented files, even on failure.
        if ANN_BACKUP.exists():
            logger.info("Restoring original annotations …")
            safe_undo()

    # ── Post-training summary ──────────────────────────────────────────────
    h, rem = divmod(int(t_train_elapsed), 3600)
    m, s   = divmod(rem, 60)
    logger.info("Training complete.")
    logger.info(f"Training time     : {h:02d}:{m:02d}:{s:02d}  ({t_train_elapsed/60:.1f} min)")
    logger.info(f"Artefacts saved to: {output_dir}")

    for ckpt in sorted(output_dir.glob("checkpoint_best*.pth")):
        logger.info(f"  {ckpt.name}")

    metrics_csv = output_dir / "metrics.csv"
    if metrics_csv.exists():
        try:
            import pandas as pd
            df     = pd.read_csv(metrics_csv)
            val_df = df.dropna(subset=["val/mAP_50_95"])
            if not val_df.empty:
                best = val_df.loc[val_df["val/mAP_50_95"].idxmax()]
                logger.info(
                    f"Best epoch {int(best['epoch'])}: "
                    f"mAP@.50:.95={best['val/mAP_50_95']:.4f}  "
                    f"mAP@.50={best['val/mAP_50']:.4f}  "
                    f"F1={best['val/F1']:.4f}"
                )
        except Exception as exc:
            logger.warning(f"Could not parse metrics.csv: {exc}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
