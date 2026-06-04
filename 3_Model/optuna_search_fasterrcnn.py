"""
optuna_search_fasterrcnn.py

Optuna hyperparameter + augmentation search for Faster R-CNN on the LARS maritime dataset.

Each trial:
  1. Samples hyperparameters  (lr, lr_backbone_ratio, weight_decay, momentum,
                                batch_size, step_size, gamma, model_variant)
  2. Samples an online augmentation pipeline (albumentations, applied per epoch)
  3. Trains Faster R-CNN for --epochs epochs with early stopping
  4. Reads best val/mAP_50 from metrics.csv and reports it to Optuna
  5. After all trials: saves trials_summary.csv

Augmentation is applied online (every epoch sees different random transforms),
so no offline image copies or undo logic is needed.

Why 25 epochs by default?
  Most mAP gain for Faster R-CNN happens in the first 15–20 epochs on this
  dataset (~4 min/epoch on an A40).  25 epochs with patience=5 gives Optuna
  a reliable signal in ≈100 min per trial while keeping the search tractable.

Usage
-----
    # 10 TPE trials, 25 epochs each (default)
    python optuna_search_fasterrcnn.py

    # Random search, more trials
    python optuna_search_fasterrcnn.py --sampler random --n-trials 20 --epochs 35

    # Resume an interrupted study
    python optuna_search_fasterrcnn.py --study-name frcnn_lars --n-trials 5

    # Dry-run: print suggested params without training
    python optuna_search_fasterrcnn.py --dry-run
"""

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from PIL import Image
import albumentations as A
import optuna
from optuna.samplers import TPESampler, RandomSampler

from train_fasterrcnn import (
    build_model, evaluate, collate_fn, compute_repeat_factors,
)
from augmentations import AUG_POLICIES, get_maritime_augmentations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).parent
DATA_ROOT = (_HERE / "../Data/lars_processed").resolve()

EARLY_STOPPING_PATIENCE  = 7      # bumped from 5; val mAP noise envelope ~±0.015
EARLY_STOPPING_MIN_DELTA = 0.005  # bumped from 0.001; below this is noise
NUM_WORKERS              = 4

# ---------------------------------------------------------------------------
# Hardcoded anchor geometry  (from fasterrcnn_anchor_analysis.ipynb)
# ---------------------------------------------------------------------------
# K-means on training-set GT box sizes (with IoU distance) → 9 cluster sizes
# distributed across the 5 FPN levels P2–P6. Tailored to LARS maritime objects
# (distant small craft up through close-up large vessels). Optuna leaves these
# fixed and only searches LR / weight decay / augmentation.
#
# torchvision's RPN requires the same num_anchors_per_location at every level,
# so we keep two sizes per level (the analysis suggested only one for P6 —
# we add 512 alongside 624 so all levels have 2×3 = 6 anchors/location).
# This changes num_anchors_per_location from the pretrained default 3 to 6,
# which triggers an RPN head rebuild in build_model().
REC_SIZES = (
    (8,   24),    # P2: small distant objects
    (56,  96),    # P3
    (112, 176),   # P4
    (288, 320),   # P5
    (512, 624),   # P6: large close-up ships (padded from analysis (624,))
)
REC_RATIOS = (0.5, 0.75, 1.25)


# ---------------------------------------------------------------------------
# Augmented dataset
# ---------------------------------------------------------------------------

class AugLARSDataset(Dataset):
    """
    COCO-format detection dataset with optional albumentations online augmentation.
    Augmentation is only applied during training (pass pipeline=None for val).
    """

    def __init__(self, root: Path, split: str, pipeline: A.Compose | None = None):
        self.img_dir  = root / split / "images"
        self.pipeline = pipeline

        ann_file = root / split / "_annotations.coco.json"
        with open(ann_file) as f:
            ann = json.load(f)

        self.categories  = ann["categories"]
        self.cat_id_map  = {c["id"]: i + 1 for i, c in enumerate(self.categories)}
        self.num_classes = len(self.categories) + 1

        self.imgs = ann["images"]
        self.anns_by_img: dict[int, list] = {}
        for a in ann["annotations"]:
            self.anns_by_img.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        meta    = self.imgs[idx]
        img_arr = np.array(Image.open(self.img_dir / meta["file_name"]).convert("RGB"))
        W, H    = meta["width"], meta["height"]

        # Collect valid GT boxes in COCO xywh format
        bboxes, cat_ids = [], []
        for a in self.anns_by_img.get(meta["id"], []):
            x, y, w, h = a["bbox"]
            x  = max(0.0, x);      y  = max(0.0, y)
            w  = min(w, W - x);    h  = min(h, H - y)
            if w > 0 and h > 0:
                bboxes.append([x, y, w, h])
                cat_ids.append(self.cat_id_map[a["category_id"]])

        # Apply augmentation
        if self.pipeline is not None and bboxes:
            result  = self.pipeline(image=img_arr, bboxes=bboxes, category_ids=cat_ids)
            img_arr = result["image"]
            bboxes  = list(result["bboxes"])
            cat_ids = list(result["category_ids"])

        tensor = TF.to_tensor(Image.fromarray(img_arr))

        # Convert xywh → xyxy
        boxes_xyxy, labels = [], []
        for (x, y, w, h), cid in zip(bboxes, cat_ids):
            x2, y2 = x + w, y + h
            if x2 > x and y2 > y:
                boxes_xyxy.append([x, y, x2, y2])
                labels.append(cid)

        target = {
            "boxes":    (torch.as_tensor(boxes_xyxy, dtype=torch.float32) if boxes_xyxy
                         else torch.zeros((0, 4), dtype=torch.float32)),
            "labels":   (torch.as_tensor(labels, dtype=torch.int64) if labels
                         else torch.zeros(0, dtype=torch.int64)),
            "image_id": torch.tensor([meta["id"]]),
        }
        return tensor, target


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def read_best_map(metrics_csv: Path) -> float | None:
    """Return best val/mAP_50 from a metrics.csv, or None on failure."""
    if not metrics_csv.exists():
        return None
    try:
        df     = pd.read_csv(metrics_csv)
        val_df = df.dropna(subset=["val/mAP_50"])
        if val_df.empty:
            return None
        return float(val_df["val/mAP_50"].max())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(
    trial:  optuna.Trial,
    args:   argparse.Namespace,
    logger: logging.Logger,
) -> float:

    trial_dir = Path(args.study_dir) / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ── Sample training hyperparameters ──────────────────────────────────────
    # Always use the base (ResNet-50-FPN-v2) variant — it loads a fully
    # COCO-pretrained model (backbone + RPN + RoI heads) and only replaces
    # the final predictor, so it converges reliably within 25 epochs.
    # The large (ResNet-101) variant has randomly initialised detection heads
    # and needs far more epochs to reach a meaningful mAP signal.
    lr               = trial.suggest_float("lr",           5e-4, 1e-2, log=True)
    lr_backbone_ratio= trial.suggest_float("lr_backbone_ratio", 0.05, 0.25, log=True)
    lr_backbone      = lr * lr_backbone_ratio
    trial.set_user_attr("lr_backbone", round(lr_backbone, 8))

    weight_decay  = trial.suggest_float("weight_decay",  1e-5, 1e-3, log=True)
    momentum      = trial.suggest_categorical("momentum", [0.85, 0.9, 0.95])
    batch_size    = trial.suggest_categorical("batch_size", [8] if args.smoke else [8, 16])
    model_variant = "base"
    step_size     = trial.suggest_categorical("step_size", [10, 15, 20])
    gamma         = trial.suggest_categorical("gamma",     [0.05, 0.1, 0.2])

    # ── Sample augmentation ───────────────────────────────────────────────────
    aug_policy = ("none" if args.smoke
                  else trial.suggest_categorical("aug_policy", list(AUG_POLICIES)))

    logger.info("")
    logger.info("=" * 64)
    logger.info(f"Trial {trial.number:3d}  →  {trial_dir.name}")
    logger.info(f"  lr={lr:.2e}  lr_backbone={lr_backbone:.2e}  wd={weight_decay:.2e}")
    logger.info(f"  momentum={momentum}  batch={batch_size}  model=base")
    logger.info(f"  step_size={step_size}  gamma={gamma}  aug_policy={aug_policy}  oversample={args.oversample}")
    logger.info("=" * 64)

    t_trial_start = time.time()

    if args.dry_run:
        logger.info("[DRY RUN] Skipping training.")
        import random as _r
        return _r.uniform(0.1, 0.5)

    # ── Build augmentation pipeline ───────────────────────────────────────────
    aug_pipeline = get_maritime_augmentations(aug_policy) if aug_policy != "none" else None

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Datasets ─────────────────────────────────────────────────────────────
    data_root = Path(args.data_root).resolve() if args.data_root else DATA_ROOT
    train_ds  = AugLARSDataset(data_root, "train", pipeline=aug_pipeline)
    val_ds    = AugLARSDataset(data_root, "valid",  pipeline=None)

    if args.oversample == "rfs":
        weights = compute_repeat_factors(train_ds, t=0.1)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_ds), replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )
        trial.set_user_attr("oversample", "rfs")
        trial.set_user_attr("oversample_max_r", round(max(weights), 2))
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )
        trial.set_user_attr("oversample", "off")
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    # Anchors are hardcoded from the maritime-aware anchor analysis (see
    # REC_SIZES / REC_RATIOS at module top). Optuna does not search them.
    model = build_model(
        model_variant,
        train_ds.num_classes,
        anchor_sizes  = REC_SIZES,
        aspect_ratios = REC_RATIOS,
    )
    model.to(device)

    # ── Optimizer ────────────────────────────────────────────────────────────
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.SGD(
        [{"params": backbone_params, "lr": lr_backbone},
         {"params": head_params,     "lr": lr}],
        momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma,
    )

    # ── Metrics CSV ───────────────────────────────────────────────────────────
    csv_fields   = ["epoch", "train/loss", "val/mAP_50_95", "val/mAP_50", "val/mAP_75"]
    metrics_path = trial_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    best_map   = 0.0
    no_improve = 0
    t_train    = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for imgs, targets in train_loader:
                imgs    = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(imgs, targets)
                loss      = sum(loss_dict.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

            val_metrics = evaluate(model, val_loader, device)
            map50       = val_metrics["val/mAP_50"]
            map5095     = val_metrics["val/mAP_50_95"]
            scheduler.step()

            logger.info(
                f"  Epoch {epoch:2d}/{args.epochs}  "
                f"loss={avg_loss:.4f}  "
                f"mAP@.5={map50:.4f}  "
                f"mAP@.5:.95={map5095:.4f}"
            )

            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow({
                    "epoch":         epoch,
                    "train/loss":    round(avg_loss, 6),
                    "val/mAP_50_95": round(map5095, 6),
                    "val/mAP_50":    round(map50, 6),
                    "val/mAP_75":    round(val_metrics["val/mAP_75"], 6),
                })

            # Optimise mAP@50 (early stop + best-checkpoint + Optuna objective).
            if map50 > best_map + EARLY_STOPPING_MIN_DELTA:
                best_map   = map50
                no_improve = 0
                torch.save(model.state_dict(), trial_dir / "checkpoint_best.pth")
            else:
                no_improve += 1

            if no_improve >= EARLY_STOPPING_PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch}.")
                break

    except Exception as exc:
        logger.error(f"Trial {trial.number} raised an exception: {exc}")
        raise

    train_elapsed = time.time() - t_train
    trial_elapsed = time.time() - t_trial_start
    trial.set_user_attr("train_time_s", round(train_elapsed, 1))
    trial.set_user_attr("trial_time_s", round(trial_elapsed, 1))

    th, tr = divmod(int(train_elapsed), 3600)
    tm, ts = divmod(tr, 60)
    logger.info(f"  Training time : {th:02d}:{tm:02d}:{ts:02d}  ({train_elapsed/60:.1f} min)")

    best_map = read_best_map(metrics_path) or best_map
    logger.info(f"Trial {trial.number} best val/mAP_50 = {best_map:.4f}  "
                f"(total: {trial_elapsed/60:.1f} min)")
    return best_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HP + augmentation search for Faster R-CNN on the LARS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-trials",   type=int, default=10,
                   help="Number of Optuna trials to run")
    p.add_argument("--epochs",     type=int, default=25,
                   help="Training epochs per trial (25 ≈ 100 min on A40; raise for accuracy)")
    p.add_argument("--study-name", type=str, default="frcnn_lars",
                   help="Optuna study name — pass the same name to resume a study")
    p.add_argument("--study-dir",  type=str,
                   default=str(_HERE / "../runs/fasterrcnn/optuna"),
                   help="Root directory for per-trial checkpoints and logs")
    p.add_argument("--data-root",  type=str, default=None,
                   help="Override dataset root (default: Data/lars_processed)")
    p.add_argument("--sampler",    choices=["tpe", "random"], default="tpe",
                   help="Optuna sampler: tpe (Bayesian, recommended) or random")
    p.add_argument("--timeout",    type=int, default=None,
                   help="Hard wall-clock limit in seconds")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print sampled hyperparameters without training")
    p.add_argument("--smoke",      action="store_true",
                   help="Smoke-test: fixes batch_size=8, base model, no aug, 2 epochs")
    p.add_argument("--oversample", choices=["off", "rfs"], default="off",
                   help="Class-balanced oversampling (study-wide, not searched). "
                        "rfs = LVIS Repeat Factor Sampling with t=0.1 "
                        "(Float ~3.3x, common classes 1x).")
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
    logger = logging.getLogger("optuna_frcnn")
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
    logger.info("SEARCH COMPLETE — results ranked by val/mAP_50")
    logger.info("=" * 64)

    rows = sorted(
        [
            (
                t.number,
                t.value,
                t.params,
                t.user_attrs.get("train_time_s", float("nan")),
                t.user_attrs.get("trial_time_s", float("nan")),
            )
            for t in completed
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    for rank, (num, val, params, train_s, total_s) in enumerate(rows, 1):
        logger.info(
            f"  #{rank:2d}  trial={num:3d}  mAP={val:.4f}"
            f"  train={train_s/60:.1f}min  total={total_s/60:.1f}min"
        )
        for k, v in params.items():
            logger.info(f"         {k} = {v}")

    best = study.best_trial
    logger.info("")
    logger.info(f"Best trial : #{best.number}  mAP={best.value:.4f}")
    logger.info(f"  lr_backbone (derived) = {best.user_attrs.get('lr_backbone', 'N/A')}")

    avg_train = sum(t.user_attrs.get("train_time_s", 0) for t in completed) / len(completed)
    logger.info(f"  Avg training time/trial : {avg_train/60:.1f} min")
    logger.info(f"  Completed / failed      : {len(completed)} / {len(failed)}")

    sh, sr = divmod(int(total_elapsed), 3600)
    sm, ss = divmod(sr, 60)
    logger.info(f"  Total search wall time  : {sh:02d}:{sm:02d}:{ss:02d}  ({total_elapsed/60:.1f} min)")
    logger.info("=" * 64)

    df  = study.trials_dataframe()
    out = study_dir / "trials_summary.csv"
    df.to_csv(out, index=False)
    logger.info(f"Trials summary → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args      = parse_args()
    study_dir = Path(args.study_dir)

    if args.smoke:
        args.epochs = 2

    logger = setup_logging(study_dir)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4)
        torch.backends.cudnn.benchmark = True
        logger.info(f"Device : CUDA — {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Device : CPU  (training will be very slow)")

    sampler = (
        TPESampler(seed=4, multivariate=True)
        if args.sampler == "tpe"
        else RandomSampler(seed=4)
    )

    storage = f"sqlite:///{study_dir}/optuna_study.db"
    study   = optuna.create_study(
        study_name     = args.study_name,
        direction      = "maximize",
        sampler        = sampler,
        storage        = storage,
        load_if_exists = True,
    )

    existing = len([t for t in study.trials if
                    t.state in (optuna.trial.TrialState.COMPLETE,
                                optuna.trial.TrialState.RUNNING)])
    logger.info("=" * 64)
    logger.info("Faster R-CNN  —  Optuna Hyperparameter + Augmentation Search")
    logger.info("=" * 64)
    logger.info(f"Study       : {args.study_name}")
    logger.info(f"Storage     : {storage}")
    logger.info(f"Sampler     : {args.sampler.upper()}")
    logger.info(f"Trials      : {args.n_trials}  ({existing} already in DB)")
    logger.info(f"Epochs/trial: {args.epochs}")
    logger.info(f"Output root : {study_dir}")
    logger.info(f"Oversample  : {args.oversample}")
    if args.dry_run:
        logger.info("DRY RUN — no training will occur")
    logger.info("=" * 64)

    t_search_start = time.time()
    try:
        study.optimize(
            lambda trial: objective(trial, args, logger),
            n_trials = args.n_trials,
            timeout  = args.timeout,
            catch    = (Exception,),
        )
    except KeyboardInterrupt:
        logger.info("Search interrupted by user.")

    print_summary(study, logger, study_dir, time.time() - t_search_start)
    logger.info("Done.")


if __name__ == "__main__":
    main()
