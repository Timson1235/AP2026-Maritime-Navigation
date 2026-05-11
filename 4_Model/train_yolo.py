"""
train_yolo.py

Fine-tune YOLOv8 on the LARS maritime dataset.

Model choices (--model):
  n  — YOLOv8 Nano   (3.2 M params, fastest)
  s  — YOLOv8 Small  (11 M params) [default]
  m  — YOLOv8 Medium (25 M params, best accuracy / speed trade-off)
  l  — YOLOv8 Large  (43 M params)

On first run the COCO annotations are converted to YOLO txt format and
cached in Data/lars_yolo/ (images are symlinked, not copied).
Pretrained weights are downloaded from the Ultralytics CDN on first run
and cached in ~/.cache/ultralytics/.

Usage
-----
    python train_yolo.py
    python train_yolo.py --model m --epochs 100 --imgsz 800
    python train_yolo.py --output-dir ../runs/yolo_m --batch 8
    python train_yolo.py --resume --output-dir ../runs/yolo_s
    python train_yolo.py --data-root ../Data/lars_relabeled
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
import torch
import yaml

_HERE     = Path(__file__).parent
DATA_ROOT = (_HERE / "../Data/lars_processed").resolve()
YOLO_ROOT = (_HERE / "../Data/lars_yolo").resolve()

DEFAULTS = dict(
    epochs                   = 100,
    batch_size               = 8,
    imgsz                    = 800,
    lr0                      = 1e-3,
    lrf                      = 0.01,    # final lr = lr0 * lrf
    momentum                 = 0.937,
    weight_decay             = 5e-4,
    warmup_epochs            = 3,
    box                      = 7.5,
    cls                      = 0.5,
    patience                 = 15,      # early stopping patience (0 = off)
    workers                  = 4,
    checkpoint_interval      = -1,      # -1 = only best + last
)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_yolo_dataset(coco_root: Path, yolo_root: Path) -> Path:
    """
    Convert COCO-format LARS dataset to YOLO txt format.

    Writes label .txt files to coco_root/{split}/labels/ (alongside the images
    directory) so Ultralytics can find them regardless of symlink resolution.
    The data.yaml is stored in yolo_root/ and points at coco_root.

    Idempotent: skips conversion if data.yaml already exists.
    """
    data_yaml = yolo_root / "data.yaml"
    if data_yaml.exists():
        return data_yaml

    yolo_root.mkdir(parents=True, exist_ok=True)

    ann_file = coco_root / "train" / "_annotations.coco.json"
    with open(ann_file) as f:
        master = json.load(f)

    categories = master["categories"]
    cat_id_map = {c["id"]: i for i, c in enumerate(categories)}
    names      = [c["name"] for c in categories]

    for split in ("train", "valid"):
        split_ann = coco_root / split / "_annotations.coco.json"
        if not split_ann.exists():
            continue

        # Labels live next to the images dir so Ultralytics finds them
        # whether or not it resolves symlinks on the images path.
        label_dir = coco_root / split / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)

        with open(split_ann) as f:
            ann = json.load(f)

        anns_by_img: dict[int, list] = {}
        for a in ann["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        for img in ann["images"]:
            W, H  = img["width"], img["height"]
            stem  = Path(img["file_name"]).stem
            lines: list[str] = []
            for a in anns_by_img.get(img["id"], []):
                x, y, w, h = a["bbox"]
                x = max(0.0, x);  y = max(0.0, y)
                w = min(w, W - x); h = min(h, H - y)
                if w <= 0 or h <= 0:
                    continue
                cls_id = cat_id_map[a["category_id"]]
                xc = (x + w / 2) / W
                yc = (y + h / 2) / H
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w/W:.6f} {h/H:.6f}")
            (label_dir / f"{stem}.txt").write_text("\n".join(lines))

    cfg = {
        "path":  str(coco_root.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(names),
        "names": names,
    }
    with open(data_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    return data_yaml


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

_ULTRLYTICS_MAP_COL = "metrics/mAP50-95(B)"
_ULTRLYTICS_MAP50   = "metrics/mAP50(B)"
_ULTRLYTICS_BOX_L   = "train/box_loss"
_ULTRLYTICS_CLS_L   = "train/cls_loss"
_ULTRLYTICS_DFL_L   = "train/dfl_loss"


def convert_metrics_csv(ultralytics_csv: Path, out_csv: Path) -> None:
    """Re-export Ultralytics results.csv to our standard metrics.csv format."""
    import pandas as pd
    df = pd.read_csv(ultralytics_csv)
    df.columns = df.columns.str.strip()

    rows = []
    for _, r in df.iterrows():
        total_loss = (
            r.get(_ULTRLYTICS_BOX_L, 0)
            + r.get(_ULTRLYTICS_CLS_L, 0)
            + r.get(_ULTRLYTICS_DFL_L, 0)
        )
        rows.append({
            "epoch":         int(r["epoch"]),
            "train/loss":    round(float(total_loss), 6),
            "val/mAP_50_95": round(float(r.get(_ULTRLYTICS_MAP_COL, float("nan"))), 6),
            "val/mAP_50":    round(float(r.get(_ULTRLYTICS_MAP50,   float("nan"))), 6),
            "val/mAP_75":    "",   # not available per-epoch from Ultralytics
        })

    fields = ["epoch", "train/loss", "val/mAP_50_95", "val/mAP_50", "val/mAP_75"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_best_map(results_csv: Path) -> float | None:
    """Return best val/mAP_50_95 from Ultralytics results.csv, or None."""
    if not results_csv.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        col = _ULTRLYTICS_MAP_COL
        if col not in df.columns:
            return None
        val = df[col].dropna()
        return float(val.max()) if not val.empty else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on the LARS maritime dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      choices=["n", "s", "m", "l"], default="s",
                   help="YOLOv8 variant: n=Nano  s=Small  m=Medium  l=Large")
    p.add_argument("--output-dir", default=str(_HERE / "../runs/yolo/baseline"),
                   help="Directory for weights, metrics.csv, and training.log")
    p.add_argument("--data-root",  default=None,
                   help="Override COCO dataset root (default: Data/lars_processed)")
    p.add_argument("--yolo-data-dir", default=None,
                   help="Where to write/read YOLO-format labels (default: Data/lars_yolo)")
    p.add_argument("--epochs",     type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch",      type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--imgsz",      type=int,   default=DEFAULTS["imgsz"])
    p.add_argument("--lr0",        type=float, default=DEFAULTS["lr0"])
    p.add_argument("--lrf",        type=float, default=DEFAULTS["lrf"])
    p.add_argument("--patience",   type=int,   default=DEFAULTS["patience"],
                   help="Early stopping patience (0 = disabled)")
    p.add_argument("--workers",    type=int,   default=DEFAULTS["workers"])
    p.add_argument("--resume",     action="store_true",
                   help="Resume from last.pt in --output-dir")
    p.add_argument("--no-plots",   action="store_true",
                   help="Skip saving Ultralytics training plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(output_dir / "training.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger("yolo_train")
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from ultralytics import YOLO

    args       = parse_args()
    output_dir = Path(args.output_dir).resolve()
    coco_root  = Path(args.data_root).resolve() if args.data_root else DATA_ROOT
    yolo_root  = Path(args.yolo_data_dir).resolve() if args.yolo_data_dir else YOLO_ROOT

    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("YOLOv8  —  LARS Maritime Dataset Fine-tuning")
    logger.info("=" * 60)

    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4)
        logger.info(f"Device       : CUDA — {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Device       : CPU  (training will be very slow)")

    # Dataset
    logger.info(f"Preparing YOLO dataset from {coco_root} …")
    data_yaml = prepare_yolo_dataset(coco_root, yolo_root)
    logger.info(f"Dataset YAML : {data_yaml}")

    # Model
    if args.resume:
        last_ckpt = output_dir / "weights" / "last.pt"
        if not last_ckpt.exists():
            logger.error(f"--resume set but {last_ckpt} not found.")
            sys.exit(1)
        model = YOLO(str(last_ckpt))
        logger.info(f"Resuming from {last_ckpt.name}")
    else:
        model = YOLO(f"yolov8{args.model}.pt")
        logger.info(f"Model        : YOLOv8{args.model.upper()}")

    logger.info(f"Output dir   : {output_dir}")
    logger.info(f"Epochs       : {args.epochs}  |  Batch: {args.batch}  |  imgsz: {args.imgsz}")
    logger.info(f"lr0={args.lr0}  lrf={args.lrf}  patience={args.patience}")
    logger.info("Starting training …")

    t_start = time.time()
    model.train(
        data             = str(data_yaml),
        epochs           = args.epochs,
        batch            = args.batch,
        imgsz            = args.imgsz,
        lr0              = args.lr0,
        lrf              = args.lrf,
        momentum         = DEFAULTS["momentum"],
        weight_decay     = DEFAULTS["weight_decay"],
        warmup_epochs    = DEFAULTS["warmup_epochs"],
        box              = DEFAULTS["box"],
        cls              = DEFAULTS["cls"],
        patience         = args.patience,
        workers          = args.workers,
        seed             = 4,
        project          = str(output_dir.parent),
        name             = output_dir.name,
        exist_ok         = True,
        resume           = args.resume,
        plots            = not args.no_plots,
        verbose          = True,
    )
    total_time = time.time() - t_start

    # Convert Ultralytics results.csv → our metrics.csv
    ul_csv  = output_dir / "results.csv"
    our_csv = output_dir / "metrics.csv"
    if ul_csv.exists():
        convert_metrics_csv(ul_csv, our_csv)
        best_map = read_best_map(ul_csv) or 0.0
    else:
        logger.warning("results.csv not found — metrics.csv not written.")
        best_map = 0.0

    h, rem = divmod(int(total_time), 3600)
    m, s   = divmod(rem, 60)
    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info(f"Training time   : {h:02d}:{m:02d}:{s:02d}  ({total_time/60:.1f} min)")
    logger.info(f"Best val mAP    : {best_map:.4f}")
    logger.info(f"Artefacts saved : {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
