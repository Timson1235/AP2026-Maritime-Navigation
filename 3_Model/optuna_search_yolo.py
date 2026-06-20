"""
optuna_search_yolo.py

Optuna hyperparameter search for YOLOv8 on the LARS maritime dataset.

Each trial:
  1. Samples training hyperparameters (lr0, lrf, momentum, weight_decay,
                                       warmup_epochs, box, cls,
                                       imgsz, batch_size, model_variant)
  2. Samples YOLOv8 built-in augmentation parameters (hsv_h/s/v, degrees,
     translate, scale, fliplr, mosaic, mixup)
  3. Trains YOLOv8 for --epochs epochs with early stopping (patience=10)
  4. Reads best val/mAP_50_95 from Ultralytics results.csv and reports to Optuna
  5. After all trials: saves trials_summary.csv

Optimizer is fixed to AdamW (--optimizer) and pretrained/deterministic are on,
mirroring the project's YOLOv8m training setup.

Why 50 epochs by default?
  YOLOv8s trains ~1 min/epoch on an A40 at imgsz=800.  50 epochs ≈ 50 min
  per trial gives Optuna a reliable signal for 10 trials in ~8 h.
  Use --epochs 25 for a faster search with less accuracy.

Usage
-----
    # 10 TPE trials, 50 epochs each (default)
    python optuna_search_yolo.py

    # Random search, more trials
    python optuna_search_yolo.py --sampler random --n-trials 20 --epochs 30

    # Resume an interrupted study
    python optuna_search_yolo.py --study-name yolo_lars --n-trials 5

    # Dry-run: print suggested params without training
    python optuna_search_yolo.py --dry-run

    # Smoke-test: 2 epochs, nano model, fixed config
    python optuna_search_yolo.py --smoke
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler, RandomSampler

from train_yolo import prepare_yolo_dataset, read_best_map

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).parent
DATA_ROOT = (_HERE / "../Data/lars_processed").resolve()
YOLO_ROOT = (_HERE / "../Data/lars_yolo").resolve()

EARLY_STOPPING_PATIENCE = 10


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(
    trial:     optuna.Trial,
    args:      argparse.Namespace,
    logger:    logging.Logger,
    data_yaml: Path,
) -> float:
    from ultralytics import YOLO

    trial_dir = Path(args.study_dir) / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ── Training hyperparameters ─────────────────────────────────────────────
    model_variant = trial.suggest_categorical("model_variant",
                        ["n"] if args.smoke else ["n", "s", "m"])
    imgsz         = trial.suggest_categorical("imgsz",
                        [640] if args.smoke else [640, 800, 1024])
    batch_size    = trial.suggest_categorical("batch_size",
                        [16] if args.smoke else [8, 16, 24])

    lr0           = trial.suggest_float("lr0",          1e-4, 1e-2, log=True)
    lrf           = trial.suggest_float("lrf",          1e-3, 0.1,  log=True)
    momentum      = trial.suggest_float("momentum",     0.70, 0.99)
    weight_decay  = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    warmup_epochs = trial.suggest_int  ("warmup_epochs", 1, 5)
    box           = trial.suggest_float("box",          4.0, 12.0)
    cls           = trial.suggest_float("cls",          0.3, 3.0)

    # ── Augmentation hyperparameters (YOLOv8 built-in) ──────────────────────
    hsv_h    = trial.suggest_float("hsv_h",    0.0,  0.10)   # hue shift
    hsv_s    = trial.suggest_float("hsv_s",    0.3,  0.90)   # saturation
    hsv_v    = trial.suggest_float("hsv_v",    0.2,  0.70)   # value/brightness
    degrees  = trial.suggest_float("degrees",  0.0, 10.0)    # rotation (vessel roll)
    translate= trial.suggest_float("translate",0.0,  0.20)   # translation
    scale    = trial.suggest_float("scale",    0.2,  0.90)   # scale jitter
    fliplr   = trial.suggest_float("fliplr",   0.3,  0.70)   # horizontal flip prob
    mosaic   = trial.suggest_float("mosaic",   0.0,  1.00)   # mosaic aug prob
    mixup    = trial.suggest_float("mixup",    0.0,  0.30)   # mixup aug prob

    logger.info("")
    logger.info("=" * 64)
    logger.info(f"Trial {trial.number:3d}  →  {trial_dir.name}")
    logger.info(f"  model=yolov8{model_variant}  imgsz={imgsz}  batch={batch_size}")
    logger.info(f"  lr0={lr0:.2e}  lrf={lrf:.3f}  momentum={momentum:.3f}  wd={weight_decay:.2e}")
    logger.info(f"  warmup={warmup_epochs}  box={box:.2f}  cls={cls:.2f}")
    logger.info(f"  hsv=({hsv_h:.3f},{hsv_s:.3f},{hsv_v:.3f})  deg={degrees:.1f}  "
                f"scale={scale:.2f}  fliplr={fliplr:.2f}  mosaic={mosaic:.2f}  mixup={mixup:.3f}")
    logger.info("=" * 64)

    t_trial_start = time.time()

    if args.dry_run:
        logger.info("[DRY RUN] Skipping training.")
        return random.uniform(0.1, 0.5)

    model = YOLO(f"yolov8{model_variant}.pt")

    try:
        model.train(
            data          = str(data_yaml),
            epochs        = args.epochs,
            batch         = batch_size,
            imgsz         = imgsz,
            optimizer     = args.optimizer,
            lr0           = lr0,
            lrf           = lrf,
            momentum      = momentum,
            weight_decay  = weight_decay,
            warmup_epochs = warmup_epochs,
            box           = box,
            cls           = cls,
            hsv_h         = hsv_h,
            hsv_s         = hsv_s,
            hsv_v         = hsv_v,
            degrees       = degrees,
            translate     = translate,
            scale         = scale,
            fliplr        = fliplr,
            mosaic        = mosaic,
            mixup         = mixup,
            patience      = EARLY_STOPPING_PATIENCE,
            seed          = 4,
            workers       = 0,
            device        = args.device,
            deterministic = True,
            pretrained    = True,
            project       = str(trial_dir.parent),
            name          = trial_dir.name,
            exist_ok      = True,
            plots         = False,   # skip per-trial plots to save time
            verbose       = False,   # suppress Ultralytics per-epoch stdout
        )
    except Exception as exc:
        logger.error(f"Trial {trial.number} raised an exception: {exc}")
        raise

    train_elapsed = time.time() - t_trial_start
    trial_elapsed = train_elapsed
    trial.set_user_attr("train_time_s", round(train_elapsed, 1))
    trial.set_user_attr("trial_time_s", round(trial_elapsed, 1))

    th, tr = divmod(int(train_elapsed), 3600)
    tm, ts = divmod(tr, 60)
    logger.info(f"  Training time : {th:02d}:{tm:02d}:{ts:02d}  ({train_elapsed/60:.1f} min)")

    ul_csv   = trial_dir / "results.csv"
    best_map = read_best_map(ul_csv)
    if best_map is None:
        logger.warning(f"Trial {trial.number}: no valid mAP found in results.csv.")
        raise optuna.exceptions.TrialPruned()

    logger.info(f"Trial {trial.number} best val/mAP_50_95 = {best_map:.4f}  "
                f"(total: {trial_elapsed/60:.1f} min)")
    return best_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HP search for YOLOv8 on the LARS dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-trials",   type=int, default=10,
                   help="Number of Optuna trials to run")
    p.add_argument("--epochs",     type=int, default=50,
                   help="Training epochs per trial (50 ≈ 50 min/trial on A40 at imgsz=800)")
    p.add_argument("--study-name", type=str, default="yolo_lars",
                   help="Optuna study name — pass the same name to resume a study")
    p.add_argument("--study-dir",  type=str,
                   default=str(_HERE / "../runs/yolo/optuna"),
                   help="Root directory for per-trial checkpoints and logs")
    p.add_argument("--data-root",  type=str, default=None,
                   help="Override COCO dataset root (default: Data/lars_processed)")
    p.add_argument("--yolo-data-dir", type=str, default=None,
                   help="Override YOLO-format dataset dir (default: Data/lars_yolo)")
    p.add_argument("--optimizer",  type=str, default="AdamW",
                   help="Optimizer fixed across trials (e.g. AdamW, SGD, auto)")
    p.add_argument("--device",     type=str, default="0",
                   help="CUDA device index (e.g. 0), '0,1' for multi-GPU, or 'cpu'")
    p.add_argument("--sampler",    choices=["tpe", "random"], default="tpe",
                   help="Optuna sampler: tpe (Bayesian, recommended) or random")
    p.add_argument("--timeout",    type=int, default=None,
                   help="Hard wall-clock limit in seconds")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print sampled hyperparameters without training")
    p.add_argument("--smoke",      action="store_true",
                   help="Smoke-test: fixes nano model, imgsz=640, batch=16, 2 epochs")
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
    logger = logging.getLogger("optuna_yolo")
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

    coco_root = Path(args.data_root).resolve()    if args.data_root    else DATA_ROOT
    yolo_root = Path(args.yolo_data_dir).resolve() if args.yolo_data_dir else YOLO_ROOT

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

    # Prepare YOLO dataset once before any trials start
    logger.info(f"Preparing YOLO dataset from {coco_root} …")
    data_yaml = prepare_yolo_dataset(coco_root, yolo_root)
    logger.info(f"Dataset YAML : {data_yaml}")

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
    logger.info("YOLOv8  —  Optuna Hyperparameter + Augmentation Search")
    logger.info("=" * 64)
    logger.info(f"Study       : {args.study_name}")
    logger.info(f"Storage     : {storage}")
    logger.info(f"Sampler     : {args.sampler.upper()}")
    logger.info(f"Optimizer   : {args.optimizer}  |  device: {args.device}")
    logger.info(f"Trials      : {args.n_trials}  ({existing} already in DB)")
    logger.info(f"Epochs/trial: {args.epochs}")
    logger.info(f"Output root : {study_dir}")
    if args.dry_run:
        logger.info("DRY RUN — no training will occur")
    logger.info("=" * 64)

    t_search_start = time.time()
    try:
        study.optimize(
            lambda trial: objective(trial, args, logger, data_yaml),
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
