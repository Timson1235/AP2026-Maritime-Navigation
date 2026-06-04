"""
train_fasterrcnn.py

Fine-tune Faster R-CNN on the LARS maritime dataset.

Model choices (--model):
  base   — FasterRCNN + ResNet-50-FPN v2, COCO pretrained head + backbone  [default]
  large  — FasterRCNN + ResNet-101-FPN, ImageNet-pretrained backbone

Both variants fine-tune the full network. The base model benefits from COCO
pretrained RPN and RoI heads in addition to the backbone, so it converges
faster. The large model only has a pretrained backbone (randomly initialised
detection heads), so it needs more epochs to reach peak performance.

Usage
-----
    python train_fasterrcnn.py
    python train_fasterrcnn.py --model large --epochs 80 --batch-size 4
    python train_fasterrcnn.py --data-root ../Data/lars_processed --output-dir ../runs/frcnn
    python train_fasterrcnn.py --resume --output-dir ../runs/frcnn_v1
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from PIL import Image
import supervision as sv
import albumentations as A

from augmentations import AUG_POLICIES, get_maritime_augmentations

_HERE     = Path(__file__).parent
DATA_ROOT = _HERE / "../Data/lars_processed"

DEFAULTS = dict(
    # Defaults mirror the best Optuna trial by mAP@50: fasterrcnn/optuna/trial_006
    # (val/mAP_50 = 0.4527; val/mAP_50_95 = 0.2529). trial_006 ran with no
    # augmentation (aug_enabled=False), so --aug-policy default is "none".
    # Anchors stay as the flat list (subsampled to one per FPN level) so the
    # pretrained RPN head is preserved.
    epochs                   = 25,      # matches Optuna trial budget
    batch_size               = 8,       # trial_006 best HP
    lr                       = 3.37e-3, # trial_006 best HP
    lr_backbone              = 7.85e-4, # trial_006 (= lr * lr_backbone_ratio=0.233)
    momentum                 = 0.9,     # trial_006 best HP
    weight_decay             = 2.01e-5, # trial_006 best HP
    step_size                = 20,      # trial_006 best HP
    gamma                    = 0.05,    # trial_006 best HP (aggressive LR decay)
    checkpoint_interval      = 5,
    early_stopping_patience  = 7,       # tuned to val-mAP noise envelope
    early_stopping_min_delta = 0.005,
    num_workers              = 4,
    # Tuned anchors from fasterrcnn_anchor_analysis.ipynb (coverage 66.6% vs 45.6% default)
    anchor_sizes             = (8, 24, 56, 96, 112, 176, 288, 320, 624),
    aspect_ratios            = (0.5, 0.75, 1.25),
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LARSDetectionDataset(Dataset):
    """COCO-format detection dataset for LARS.

    Optional online augmentation via an albumentations pipeline (xywh boxes).
    Pass ``pipeline=None`` for validation/test.
    """

    def __init__(self, root: Path, split: str,
                 pipeline: A.Compose | None = None):
        self.img_dir  = root / split / "images"
        self.pipeline = pipeline
        ann_file = root / split / "_annotations.coco.json"
        with open(ann_file) as f:
            ann = json.load(f)

        # Faster R-CNN expects labels in [1, num_classes]; 0 is background.
        self.categories = ann["categories"]
        self.cat_id_map = {c["id"]: i + 1 for i, c in enumerate(self.categories)}
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
            x = max(0.0, x);  y = max(0.0, y)
            w = min(w, W - x); h = min(h, H - y)
            if w > 0 and h > 0:
                bboxes.append([x, y, w, h])
                cat_ids.append(self.cat_id_map[a["category_id"]])

        # Apply augmentation if a pipeline is set
        if self.pipeline is not None and bboxes:
            result  = self.pipeline(image=img_arr, bboxes=bboxes, category_ids=cat_ids)
            img_arr = result["image"]
            bboxes  = list(result["bboxes"])
            cat_ids = list(result["category_ids"])

        tensor = TF.to_tensor(Image.fromarray(img_arr))

        # Convert xywh -> xyxy
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


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(model_name: str, num_classes: int,
                anchor_sizes=None, aspect_ratios=None) -> torch.nn.Module:
    """
    `anchor_sizes` accepts two formats:

      - **Flat**  — e.g. ``(8, 24, 56, 96, 112, 176, 288, 320, 624)``.
        Sizes are subsampled to one-per-FPN-level. Keeps the pretrained RPN
        head weights (``num_anchors_per_location`` stays at the default 3).

      - **Grouped** (one tuple per FPN level) — e.g.
        ``((8, 24), (56, 96), (112, 176), (288, 320), (512, 624))``.
        Used as-is. Short groups are right-padded with their last size so
        ``num_anchors_per_location`` is consistent across levels (required
        by torchvision's RPN). If the resulting count differs from the
        pretrained default (3), the RPN head is rebuilt from scratch.
    """
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if model_name == "large":
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.models import ResNet101_Weights
        backbone = resnet_fpn_backbone(
            "resnet101",
            weights=ResNet101_Weights.IMAGENET1K_V1,
            trainable_layers=3,
        )
        model = FasterRCNN(backbone, num_classes=num_classes)
    else:
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
        )
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if anchor_sizes is not None and aspect_ratios is not None:
        from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
        n_levels       = len(model.rpn.anchor_generator.sizes)
        ratios_tuple   = tuple(aspect_ratios)
        default_n_apl  = model.rpn.anchor_generator.num_anchors_per_location()[0]

        is_grouped = (len(anchor_sizes) > 0
                      and hasattr(anchor_sizes[0], "__iter__"))

        if is_grouped:
            if len(anchor_sizes) != n_levels:
                raise ValueError(
                    f"Grouped anchor_sizes has {len(anchor_sizes)} levels, "
                    f"but FPN has {n_levels}."
                )
            groups = [tuple(g) for g in anchor_sizes]
            max_sizes_per_level = max(len(g) for g in groups)
            sizes_per_level = tuple(
                g + (g[-1],) * (max_sizes_per_level - len(g)) for g in groups
            )
        else:
            # Flat list — subsample evenly to one-per-level.
            sorted_sizes = sorted(anchor_sizes)
            if len(sorted_sizes) > n_levels:
                idxs = [int(round(i * (len(sorted_sizes) - 1) / (n_levels - 1)))
                        for i in range(n_levels)]
                sorted_sizes = [sorted_sizes[i] for i in idxs]
            sizes_per_level = tuple((s,) for s in sorted_sizes)

        model.rpn.anchor_generator = AnchorGenerator(
            sizes=sizes_per_level,
            aspect_ratios=(ratios_tuple,) * n_levels,
        )

        # Rebuild RPN head if anchors/location no longer match the pretrained
        # head's output width (otherwise the head produces too few/many channels).
        new_n_apl = len(sizes_per_level[0]) * len(ratios_tuple)
        if new_n_apl != default_n_apl:
            in_channels = model.backbone.out_channels
            model.rpn.head = RPNHead(in_channels, new_n_apl, conv_depth=2)

    return model


# ---------------------------------------------------------------------------
# Repeat Factor Sampling  (LVIS-style class-balanced oversampling)
# ---------------------------------------------------------------------------

def compute_repeat_factors(ds: Dataset,
                           t: float    = 0.1,
                           max_r: float = 10.0) -> list[float]:
    """
    Per-image repeat factor for class-balanced sampling
    (LVIS Repeat Factor Sampling, adapted to LARS frequencies).

      f_c = (# train images containing category c) / N
      r_c = max(1, sqrt(t / f_c))            (capped at max_r)
      r_i = max over c in image i of r_c

    LVIS uses t=1e-3 because their tail goes <0.01%. LARS's rarest
    class (Float) is at 0.9%, so t=0.1 is calibrated to give it a
    meaningful boost while leaving the majority classes untouched:
      Float        (0.9%)  -> 3.3x
      Animal       (3.2%)  -> 1.8x
      Paddle board (4.0%)  -> 1.6x
      Swimmer      (4.8%)  -> 1.4x
      Row boats    (9.1%)  -> 1.0x (just below)
      Buoy / Other / Boat  -> 1.0x (above t)
    Mean repeat factor ~= 1.09 -> effective epoch size grows ~9%.

    Expects ``ds`` to expose ``imgs`` (list of COCO image dicts) and
    ``anns_by_img`` (dict image_id -> list of annotation dicts).
    """
    from collections import defaultdict
    N = len(ds.imgs)
    cat_img_count: dict[int, int] = defaultdict(int)
    per_img_cats:  list[set[int]] = []

    for img in ds.imgs:
        cats = {a["category_id"] for a in ds.anns_by_img.get(img["id"], [])}
        per_img_cats.append(cats)
        for c in cats:
            cat_img_count[c] += 1

    r_c = {c: min(max_r, max(1.0, (t / (n / N)) ** 0.5))
           for c, n in cat_img_count.items()}

    return [max((r_c[c] for c in cats), default=1.0) for cats in per_img_cats]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_pred, all_gt = [], []

    for imgs, targets in loader:
        imgs  = [img.to(device) for img in imgs]
        preds = model(imgs)

        for target, pred in zip(targets, preds):
            gt_boxes  = target["boxes"].cpu().numpy()
            gt_labels = target["labels"].cpu().numpy() - 1  # → 0-indexed for supervision

            all_gt.append(
                sv.Detections(xyxy=gt_boxes, class_id=gt_labels) if len(gt_boxes)
                else sv.Detections(xyxy=np.zeros((0, 4), dtype=float),
                                   class_id=np.array([], dtype=int))
            )

            pred_boxes  = pred["boxes"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy() - 1
            pred_scores = pred["scores"].cpu().numpy()

            all_pred.append(
                sv.Detections(xyxy=pred_boxes, class_id=pred_labels,
                              confidence=pred_scores) if len(pred_boxes)
                else sv.Detections(xyxy=np.zeros((0, 4), dtype=float),
                                   class_id=np.array([], dtype=int),
                                   confidence=np.array([], dtype=float))
            )

    metric = sv.metrics.MeanAveragePrecision()
    metric.update(all_pred, all_gt)
    res = metric.compute()

    return {
        "val/mAP_50_95": float(res.map50_95),
        "val/mAP_50":    float(res.map50),
        "val/mAP_75":    float(res.map75),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune Faster R-CNN on the LARS maritime dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",  default=str(_HERE / "../runs/fasterrcnn/baseline"),
                   help="Directory for checkpoints, metrics.csv, and training.log")
    p.add_argument("--epochs",      type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch-size",  type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",          type=float, default=DEFAULTS["lr"],
                   help="Head / RPN learning rate")
    p.add_argument("--lr-backbone", type=float, default=DEFAULTS["lr_backbone"],
                   help="Backbone learning rate (keep ≤ 1/10 of --lr)")
    p.add_argument("--model", choices=["base", "large"], default="base",
                   help="'base': ResNet-50-FPN-v2 (COCO pretrained) | "
                        "'large': ResNet-101-FPN (ImageNet backbone)")
    p.add_argument("--data-root",   default=None,
                   help="Override dataset root (default: Data/lars_processed)")
    p.add_argument("--resume",      action="store_true",
                   help="Resume from checkpoint_best.pth in --output-dir")
    p.add_argument("--no-early-stopping", action="store_true",
                   help="Disable early stopping and always train for --epochs epochs")
    p.add_argument("--num-workers",   type=int,   default=DEFAULTS["num_workers"])
    p.add_argument("--momentum",      type=float, default=DEFAULTS["momentum"])
    p.add_argument("--weight-decay",  type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--step-size",     type=int,   default=DEFAULTS["step_size"],
                   help="StepLR: decay LR every N epochs")
    p.add_argument("--gamma",         type=float, default=DEFAULTS["gamma"],
                   help="StepLR: multiply LR by this at each step")
    p.add_argument("--anchor-sizes", type=int, nargs="+",
                   default=list(DEFAULTS["anchor_sizes"]),
                   help="RPN anchor sizes in px distributed across FPN levels. "
                        "Default: LARS-tuned from fasterrcnn_anchor_analysis.ipynb.")
    p.add_argument("--aspect-ratios", type=float, nargs="+",
                   default=list(DEFAULTS["aspect_ratios"]),
                   help="RPN aspect ratios (h/w) shared across all FPN levels. "
                        "Default: LARS-tuned from fasterrcnn_anchor_analysis.ipynb.")
    p.add_argument("--oversample", choices=["off", "rfs"], default="off",
                   help="Class-balanced oversampling. rfs = LVIS Repeat Factor "
                        "Sampling with t=0.1 (Float ~3.3x, common classes 1x).")
    p.add_argument("--aug-policy", choices=list(AUG_POLICIES), default="none",
                   help="Online augmentation policy. Default 'none' mirrors "
                        "trial_006 which had aug_enabled=False.")
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
    logger = logging.getLogger("frcnn_train")
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

    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("Faster R-CNN  —  LARS Maritime Dataset Fine-tuning")
    logger.info("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Device        : CUDA — {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        logger.info("Device        : CPU (training will be slow)")

    # Reproducibility
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(4)

    # Dataset
    if not DATA_ROOT.exists():
        logger.error(f"Dataset not found at {DATA_ROOT.resolve()}")
        sys.exit(1)

    aug_pipeline = (get_maritime_augmentations(args.aug_policy)
                    if args.aug_policy != "none" else None)
    train_ds = LARSDetectionDataset(DATA_ROOT, "train", pipeline=aug_pipeline)
    val_ds   = LARSDetectionDataset(DATA_ROOT, "valid", pipeline=None)
    num_classes = train_ds.num_classes
    logger.info(f"Aug policy    : {args.aug_policy}")

    if args.oversample == "rfs":
        weights = compute_repeat_factors(train_ds, t=0.1)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_ds), replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler,
                                  num_workers=args.num_workers, collate_fn=collate_fn,
                                  pin_memory=device.type == "cuda")
        logger.info(f"Oversample    : rfs  (t=0.1, max_r={max(weights):.2f}, "
                    f"effective_size={sum(weights)/len(weights):.2f}x)")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn,
                                  pin_memory=device.type == "cuda")
        logger.info("Oversample    : off")
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")

    logger.info(f"Model variant : FasterRCNN-{args.model.capitalize()}")
    logger.info(f"Dataset root  : {DATA_ROOT.resolve()}")
    logger.info(f"Train images  : {len(train_ds)}  |  Val images: {len(val_ds)}")
    logger.info(f"Classes       : {num_classes - 1} + background")
    logger.info(f"Output dir    : {output_dir.resolve()}")
    logger.info("Hyperparameters:")
    logger.info(f"  epochs            = {args.epochs}")
    logger.info(f"  batch_size        = {args.batch_size}")
    logger.info(f"  lr                = {args.lr}")
    logger.info(f"  lr_backbone       = {args.lr_backbone}")
    logger.info(f"  weight_decay      = {args.weight_decay}")
    logger.info(f"  step_size / gamma = {args.step_size} / {args.gamma}")
    logger.info(f"  momentum          = {args.momentum}")
    logger.info(f"  early_stopping    = {not args.no_early_stopping}"
                + (f" (patience={DEFAULTS['early_stopping_patience']})"
                   if not args.no_early_stopping else ""))

    # Model
    model = build_model(args.model, num_classes,
                        anchor_sizes=args.anchor_sizes,
                        aspect_ratios=args.aspect_ratios)
    model.to(device)

    if args.resume:
        ckpt_path = output_dir / "checkpoint_best.pth"
        if not ckpt_path.exists():
            candidates = sorted(output_dir.glob("checkpoint_epoch*.pth"))
            ckpt_path  = candidates[-1] if candidates else None
        if ckpt_path and ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            logger.info(f"Resumed from  : {ckpt_path.name}")
        else:
            logger.warning("--resume set but no checkpoint found; starting fresh")

    # Optimizer — separate LR for backbone vs detection heads
    backbone_params = [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params,     "lr": args.lr},
        ],
        momentum     = args.momentum,
        weight_decay = args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    # Metrics CSV
    csv_fields   = ["epoch", "train/loss", "val/mAP_50_95", "val/mAP_50", "val/mAP_75"]
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    best_map   = 0.0
    no_improve = 0
    t_start    = time.time()

    logger.info("Starting training …")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
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

        # ── Validate ───────────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, device)
        map50       = val_metrics["val/mAP_50"]
        map5095     = val_metrics["val/mAP_50_95"]

        scheduler.step()

        elapsed = time.time() - t_start
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  "
            f"mAP@.5={map50:.4f}  "
            f"mAP@.5:.95={map5095:.4f}  "
            f"({elapsed / 60:.1f} min elapsed)"
        )

        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                "epoch":         epoch,
                "train/loss":    round(avg_loss, 6),
                "val/mAP_50_95": round(map5095, 6),
                "val/mAP_50":    round(map50, 6),
                "val/mAP_75":    round(val_metrics["val/mAP_75"], 6),
            })

        if epoch % DEFAULTS["checkpoint_interval"] == 0:
            torch.save(model.state_dict(),
                       output_dir / f"checkpoint_epoch{epoch:03d}.pth")

        # Best-checkpoint + early-stopping use mAP@50.
        if map50 > best_map + DEFAULTS["early_stopping_min_delta"]:
            best_map   = map50
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "checkpoint_best.pth")
            logger.info(f"  → New best mAP@50: {best_map:.4f} — checkpoint saved")
        else:
            no_improve += 1

        if not args.no_early_stopping and no_improve >= DEFAULTS["early_stopping_patience"]:
            logger.info(f"Early stopping triggered (no improvement for {no_improve} epochs).")
            break

    # Summary
    total_time = time.time() - t_start
    h, rem = divmod(int(total_time), 3600)
    m, s   = divmod(rem, 60)
    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info(f"Training time   : {h:02d}:{m:02d}:{s:02d}  ({total_time / 60:.1f} min)")
    logger.info(f"Best val mAP@50 : {best_map:.4f}")
    logger.info(f"Artefacts saved : {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
