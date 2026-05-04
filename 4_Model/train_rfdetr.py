"""
RF-DETR fine-tuning script for the LARS maritime dataset.

Usage:
    python train_rfdetr.py
    python train_rfdetr.py --output-dir ../runs/rfdetr_v2 --epochs 80
    python train_rfdetr.py --resume                  # resume from latest checkpoint

Logs training progress to both console and <output-dir>/training.log.
Checkpoints and metrics.csv are saved to <output-dir>.
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).parent
DATA_ROOT = _HERE / "../Data/lars_processed"

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    epochs               = 100,
    batch_size           = 4,
    grad_accum_steps     = 4,       # effective batch = batch_size × grad_accum_steps = 16
    lr                   = 1e-4,
    lr_encoder           = 1e-5,    # backbone — 10× lower to preserve pretrained features
    resolution           = 728,
    weight_decay         = 1e-4,
    grad_clip_max_norm   = 0.1,
    checkpoint_interval  = 5,
    early_stopping_patience  = 10,
    early_stopping_min_delta = 0.001,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train RF-DETR on the LARS maritime dataset")
    p.add_argument("--output-dir", default=str(_HERE / "../runs/rfdetr"),
                   help="Directory for checkpoints, metrics.csv, and training.log")
    p.add_argument("--epochs",     type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int,   default=DEFAULTS["batch_size"],
                   help="Per-GPU batch size (effective = batch-size × grad-accum-steps)")
    p.add_argument("--lr",         type=float, default=DEFAULTS["lr"],
                   help="Decoder/head learning rate")
    p.add_argument("--lr-encoder", type=float, default=DEFAULTS["lr_encoder"],
                   help="Backbone learning rate (keep ≤ 1/10 of --lr)")
    p.add_argument("--resolution", type=int,   default=DEFAULTS["resolution"],
                   choices=[448, 504, 560, 616, 672, 728, 784],
                   help="Input resolution (must be a multiple of 56)")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from the latest checkpoint in --output-dir")
    p.add_argument("--no-early-stopping", action="store_true",
                   help="Disable early stopping and always train for --epochs epochs")
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
    logger     = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("RF-DETR  —  LARS Maritime Dataset Fine-tuning")
    logger.info("=" * 60)

    # Reproducibility
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4)
        logger.info(f"Device        : CUDA — {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Device        : CPU (training will be slow)")

    # Validate dataset
    if not DATA_ROOT.exists():
        logger.error(f"Dataset not found at {DATA_ROOT.resolve()}")
        logger.error("Run  2_DataPreprocessing/datasplit.py  first.")
        sys.exit(1)

    logger.info(f"Dataset root  : {DATA_ROOT.resolve()}")
    logger.info(f"Output dir    : {output_dir.resolve()}")

    # Log hyperparameters
    eff_batch = args.batch_size * DEFAULTS["grad_accum_steps"]
    logger.info("Hyperparameters:")
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
    from rfdetr import RFDETRBase

    model = RFDETRBase(resolution=args.resolution)

    logger.info("Starting training …")
    t_train_start = time.time()
    model.train(
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
    )
    t_train_elapsed = time.time() - t_train_start

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
