"""
cv_relabel.py  —  4-fold cross-validation with model-assisted label cleaning.

Pipeline
--------
1. Load all COCO annotations from train / valid / test splits of lars_processed.
2. Pool all annotated images and split them into 4 scene-level folds (no scene
   appears in more than one fold).
3. For each fold k (0-3):
     a. Write a temporary RF-DETR dataset:
          fold_dir/train/   ← images from folds 0..3 except k  (symlinked, flat)
          fold_dir/valid/   ← images from fold k               (symlinked, flat)
        Images are symlinked flat (no images/ subdir) — rfdetr expects
        dataset_dir/split/filename.jpg.
     b. Train RFDETRBase with Trial-004 hyperparameters.
     c. Run the trained model at threshold=0.0 on fold k images and apply the
        noisy-label detection from noisy_label_eval.ipynb:
          • Ghost / phantom GT boxes  → removed
          • Merged / oversized GT boxes → removed
          • High-conf unmatched predictions → added as new annotations
     d. Cache cleaned annotations to runs/cv_relabel/fold_k/cleaned_anns.json.
4. Write Data/lars_relabeled/{train,valid,test}/_annotations.coco.json,
   preserving original split membership.  provenance.json logs every change.

Usage
-----
  python 4_Model/cv_relabel.py                      # full run
  python 4_Model/cv_relabel.py --resume-fold 2      # skip folds 0-1 from cache
  python 4_Model/cv_relabel.py --relabel-only       # skip training, use existing ckpts
  python 4_Model/cv_relabel.py --keep-tmp           # keep fold dataset dirs
  python 4_Model/cv_relabel.py --epochs 60          # shorter per-fold training
"""

import argparse
import json
import logging
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths  (always absolute — never pass raw strings as path components)
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve().parent
DATA_ROOT = (_HERE.parent / "Data" / "lars_processed").resolve()
RUNS_DIR  = (_HERE.parent / "runs"  / "cv_relabel").resolve()
OUT_ROOT  = (_HERE.parent / "Data"  / "lars_relabeled").resolve()

# ---------------------------------------------------------------------------
# Trial-004 hyperparameters (from hyperparameter search)
# ---------------------------------------------------------------------------
TRIAL004 = dict(
    lr                 = 4.39e-4,
    lr_encoder         = 5.30e-5,
    weight_decay       = 4.49e-4,
    resolution         = 672,
    batch_size         = 4,
    grad_accum_steps   = 4,
    grad_clip_max_norm = 0.1,
    checkpoint_interval= 5,
)

# ---------------------------------------------------------------------------
# Detection thresholds  (match noisy_label_eval.ipynb exactly)
# ---------------------------------------------------------------------------
IOU_THRESH              = 0.5
VIZ_THRESHOLD           = 0.1     # min pred conf to consider at all
FP_CONF_THRESHOLD       = 0.55    # unmatched pred > this → likely missing label
FN_CONF_THRESHOLD       = 0.05    # FN: max conf inside box < this → ghost
MERGE_CONF_THRESHOLD    = 0.5     # min conf to count as "real object inside" for merge
MIN_GHOST_AREA          = 2000    # skip ghost check for GT boxes smaller than this (px²)
RESIZE_COVERAGE_THRESH  = 0.5     # FP covers ≥ this fraction of a GT box → resize, not new label

K_FOLDS  = 4
SEED     = 4


# ===========================================================================
# Logging
# ===========================================================================
def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(output_dir / "cv_relabel.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log = logging.getLogger("cv_relabel")
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


# ===========================================================================
# Data loading
# ===========================================================================
def load_all_splits(log: logging.Logger):
    """
    Load train + valid + test from lars_processed.
    Returns (all_images, anns_by_id, categories).
    Each image dict gets an extra '_split' key so we can restore it later.
    """
    all_images: list[dict]       = []
    anns_by_id: dict[int, list]  = defaultdict(list)
    categories: list[dict]       = []

    for split in ("train", "valid", "test"):
        ann_path = DATA_ROOT / split / "_annotations.coco.json"
        if not ann_path.exists():
            log.warning(f"[load] {split}: annotation file missing, skipping")
            continue
        coco = json.load(open(ann_path))
        if not categories:
            categories = coco["categories"]
        for img in coco["images"]:
            img_copy = {**img, "_split": split}
            all_images.append(img_copy)
        for ann in coco["annotations"]:
            anns_by_id[ann["image_id"]].append(ann)
        log.info(f"[load] {split}: {len(coco['images'])} images, "
                 f"{len(coco['annotations'])} annotations")

    return all_images, anns_by_id, categories


# ===========================================================================
# K-fold split  (scene-level — no scene leaks across folds)
# ===========================================================================
def scene_of(fname: str) -> str:
    """'mastr_0001_00009.jpg' → 'mastr_0001'"""
    return re.sub(r"_\d{5}\.jpg$", "", fname)


def make_folds(images: list[dict], k: int) -> list[list[dict]]:
    scenes: dict[str, list] = defaultdict(list)
    for img in images:
        scenes[scene_of(img["file_name"])].append(img)

    names = sorted(scenes.keys())
    rng   = random.Random(SEED)
    rng.shuffle(names)

    folds: list[list] = [[] for _ in range(k)]
    for i, name in enumerate(names):
        folds[i % k].extend(scenes[name])
    return folds


# ===========================================================================
# Fold dataset construction
# ===========================================================================
def _img_source(img: dict) -> Path:
    """Return the absolute path to the source image file."""
    split = img["_split"]
    return (DATA_ROOT / split / "images" / img["file_name"]).resolve()


def _symlink_flat(images: list[dict], dst_dir: Path) -> None:
    """
    Symlink each image into dst_dir/<file_name> (flat, no images/ subdir).
    rfdetr loads images from dataset_dir/split/filename.jpg, so files must
    live directly inside the split directory.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        src = _img_source(img)
        dst = dst_dir / img["file_name"]
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if src.exists():
            dst.symlink_to(src)
        else:
            pass  # missing source — training will skip gracefully


def _write_coco_json(images: list[dict], anns_by_id: dict,
                     categories: list, path: Path) -> None:
    anns, ann_id = [], 1
    clean_images = []
    for img in images:
        clean_img = {k: v for k, v in img.items() if k != "_split"}
        clean_images.append(clean_img)
        for ann in anns_by_id.get(img["id"], []):
            anns.append({**ann, "id": ann_id})
            ann_id += 1
    with open(path, "w") as f:
        json.dump({"images": clean_images, "annotations": anns,
                   "categories": categories}, f)


def build_fold_dataset(
    k:          int,
    folds:      list[list],
    anns_by_id: dict,
    categories: list,
    fold_dir:   Path,
    log:        logging.Logger,
) -> None:
    """
    Write a temporary RF-DETR dataset:
      fold_dir/train/   — all folds except k (symlinked flat)
      fold_dir/valid/   — fold k              (symlinked flat, used for val during training)
    """
    train_imgs = [img for i, fold in enumerate(folds) if i != k for img in fold]
    valid_imgs = folds[k]

    train_dir = (fold_dir / "train").resolve()
    valid_dir = (fold_dir / "valid").resolve()

    _symlink_flat(train_imgs, train_dir)
    _symlink_flat(valid_imgs, valid_dir)

    _write_coco_json(train_imgs, anns_by_id, categories,
                     train_dir / "_annotations.coco.json")
    _write_coco_json(valid_imgs, anns_by_id, categories,
                     valid_dir / "_annotations.coco.json")

    log.info(f"  [Fold {k}] dataset: {len(train_imgs)} train, "
             f"{len(valid_imgs)} valid → {fold_dir}")


# ===========================================================================
# Model helpers
# ===========================================================================
def train_fold(fold_dir: Path, ckpt_dir: Path, epochs: int,
               log: logging.Logger) -> None:
    from rfdetr import RFDETRBase

    log.info(f"  dataset_dir : {fold_dir}")
    log.info(f"  output_dir  : {ckpt_dir}")
    log.info(f"  epochs      : {epochs}")

    model = RFDETRBase(resolution=TRIAL004["resolution"])
    model.train(
        dataset_dir         = str(fold_dir),
        epochs              = epochs,
        batch_size          = TRIAL004["batch_size"],
        grad_accum_steps    = TRIAL004["grad_accum_steps"],
        lr                  = TRIAL004["lr"],
        lr_encoder          = TRIAL004["lr_encoder"],
        resolution          = TRIAL004["resolution"],
        weight_decay        = TRIAL004["weight_decay"],
        grad_clip_max_norm  = TRIAL004["grad_clip_max_norm"],
        checkpoint_interval = TRIAL004["checkpoint_interval"],
        output_dir          = str(ckpt_dir),
    )


def load_model(ckpt_dir: Path, log: logging.Logger):
    from rfdetr import RFDETRBase

    candidates = sorted(ckpt_dir.glob("checkpoint_best*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
    best = next((p for p in candidates if p.name == "checkpoint_best_total.pth"),
                candidates[-1])
    log.info(f"  checkpoint  : {best.name}")
    return RFDETRBase(pretrain_weights=str(best),
                      resolution=TRIAL004["resolution"])


# ===========================================================================
# Detection helpers  (identical logic to noisy_label_eval.ipynb)
# ===========================================================================
def _iou(a, b) -> float:
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _gt_coverage(pred_box, gt_box) -> float:
    """Fraction of gt_box area covered by pred_box (asymmetric — not IoU)."""
    xi1 = max(pred_box[0], gt_box[0]); yi1 = max(pred_box[1], gt_box[1])
    xi2 = min(pred_box[2], gt_box[2]); yi2 = min(pred_box[3], gt_box[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    return inter / gt_area if gt_area > 0 else 0.0


def _centers_inside(gt_box, vis_boxes, vis_scores,
                    excl_idx=None, min_conf: float = 0.0) -> list[float]:
    result = []
    for i in range(len(vis_boxes)):
        if excl_idx is not None and i == excl_idx:
            continue
        if vis_scores[i] < min_conf:
            continue
        cx = (vis_boxes[i][0] + vis_boxes[i][2]) / 2
        cy = (vis_boxes[i][1] + vis_boxes[i][3]) / 2
        if gt_box[0] <= cx <= gt_box[2] and gt_box[1] <= cy <= gt_box[3]:
            result.append(float(vis_scores[i]))
    return result


# ===========================================================================
# Core relabeling  (per fold)
# ===========================================================================
def relabel_images(
    fold_imgs:  list[dict],
    anns_by_id: dict,
    categories: list,
    model,
    log:        logging.Logger,
) -> tuple[dict, dict, Counter]:
    """
    Run the trained model on every image in fold_imgs and clean their annotations.

    Returns
    -------
    cleaned_anns : {image_id: [ann, ...]}
    provenance   : {image_id: {file_name, split, removed, added}}
    stats        : Counter
    """
    from PIL import Image as PILImage

    class_ids    = [c["id"] for c in categories]
    stats        = Counter()
    cleaned_anns = {}
    provenance   = {}

    for img_entry in fold_imgs:
        gid    = img_entry["id"]
        fname  = img_entry["file_name"]
        split  = img_entry["_split"]

        orig_anns = anns_by_id.get(gid, [])

        gt_boxes = np.array([
            [a["bbox"][0], a["bbox"][1],
             a["bbox"][0] + a["bbox"][2],
             a["bbox"][1] + a["bbox"][3]]
            for a in orig_anns
        ], dtype=float) if orig_anns else np.empty((0, 4))

        # ── Inference ────────────────────────────────────────────────────────
        img_path = _img_source(img_entry)
        if not img_path.exists():
            log.warning(f"  [skip] {fname} ({split}) — source image missing")
            cleaned_anns[gid] = orig_anns
            continue

        pil_img = PILImage.open(img_path).convert("RGB")
        raw     = model.predict(pil_img, threshold=0.0)

        pred_boxes  = raw.xyxy        if len(raw) > 0 else np.empty((0, 4))
        pred_scores = (raw.confidence if raw.confidence is not None and len(raw) > 0
                       else np.ones(len(raw)))
        pred_cids   = raw.class_id    if len(raw) > 0 else np.array([], dtype=int)

        # Visible predictions (above VIZ_THRESHOLD)
        vis_mask   = pred_scores >= VIZ_THRESHOLD
        vis_boxes  = pred_boxes[vis_mask]
        vis_scores = pred_scores[vis_mask]
        vis_cids   = pred_cids[vis_mask]

        # ── Greedy TP matching (highest confidence first) ─────────────────────
        order = np.argsort(-vis_scores) if len(vis_scores) > 0 else np.array([], dtype=int)
        pb_s, ps_s = vis_boxes[order], vis_scores[order]

        matched_gt, matched_pred = {}, {}
        for pi in range(len(pb_s)):
            best_iou, best_gi = IOU_THRESH, -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt:
                    continue
                v = _iou(pb_s[pi], gt_boxes[gi])
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_gi >= 0:
                matched_gt[best_gi]  = pi
                matched_pred[pi]     = best_gi

        # ── Step 3: Split high-conf FPs into resize vs. new label ────────────
        # pb_s is sorted by confidence descending, so the first FP targeting
        # a given GT index is always the highest-confidence one.
        resize_preds    = []   # (pred_box, score, cls_idx, gi) — expand existing GT
        new_label_preds = []   # (pred_box, score, cls_idx)     — add new annotation
        resize_gt_seen  = set()

        for pi in range(len(pb_s)):
            if pi not in matched_pred and ps_s[pi] > FP_CONF_THRESHOLD:
                pred_cls_idx = int(vis_cids[order[pi]])
                b = pb_s[pi]; s = float(ps_s[pi])
                best_cov, best_gi = RESIZE_COVERAGE_THRESH, -1
                for gi in range(len(gt_boxes)):
                    cov = _gt_coverage(b, gt_boxes[gi])
                    if cov > best_cov:
                        best_cov, best_gi = cov, gi
                if best_gi >= 0 and best_gi not in resize_gt_seen:
                    resize_preds.append((b, s, pred_cls_idx, best_gi))
                    resize_gt_seen.add(best_gi)
                else:
                    new_label_preds.append((b, s, pred_cls_idx))

        # ── Step 4: Bad GT boxes ──────────────────────────────────────────────
        bad_gt: dict[int, str] = {}   # gi → "ghost" | "merged"

        for gi in range(len(gt_boxes)):
            gt_box  = gt_boxes[gi]
            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

            if gi in matched_gt:
                # TP: flag if ≥2 extra confident predictions are inside (merged)
                matched_vis_idx = int(order[matched_gt[gi]])
                extra = _centers_inside(
                    gt_box, vis_boxes, vis_scores,
                    excl_idx=matched_vis_idx,
                    min_conf=MERGE_CONF_THRESHOLD,
                )
                if len(extra) >= 2:
                    bad_gt[gi] = "merged"
                    stats["removed_merged"] += 1
            else:
                # FN: scan all raw predictions for best IoU / conf
                best_iou_val, best_conf_val = 0.0, 0.0
                for pi in range(len(pred_boxes)):
                    v = _iou(pred_boxes[pi], gt_box)
                    if v > best_iou_val:
                        best_iou_val  = v
                        best_conf_val = float(pred_scores[pi])

                inside = _centers_inside(
                    gt_box, vis_boxes, vis_scores,
                    min_conf=MERGE_CONF_THRESHOLD,
                )
                if len(inside) >= 2:
                    bad_gt[gi] = "merged"
                    stats["removed_merged"] += 1
                elif (gt_area >= MIN_GHOST_AREA
                      and (best_iou_val < IOU_THRESH
                           or best_conf_val < FN_CONF_THRESHOLD)):
                    bad_gt[gi] = "ghost"
                    stats["removed_ghost"] += 1

        # ── Build cleaned annotation list ─────────────────────────────────────
        resize_by_gi  = {gi: (b, s) for b, s, _, gi in resize_preds}
        image_changed = bool(bad_gt) or bool(resize_preds) or bool(new_label_preds)

        kept, removed_record, resized_record = [], [], []
        for gi, ann in enumerate(orig_anns):
            if gi in bad_gt:
                removed_record.append({
                    **ann,
                    "relabel_action": "removed",
                    "relabel_reason": bad_gt[gi],
                })
            elif gi in resize_by_gi:
                pred_box, score = resize_by_gi[gi]
                x1, y1, x2, y2 = pred_box
                w, h = float(x2 - x1), float(y2 - y1)
                resized_ann = {
                    **ann,
                    "bbox":               [float(x1), float(y1), w, h],
                    "area":               float(w * h),
                    "relabel_action":     "resized",
                    "relabel_conf":       score,
                    "relabel_orig_bbox":  ann["bbox"],
                }
                kept.append(resized_ann)
                resized_record.append(resized_ann)
                stats["resized"] += 1
            else:
                kept_ann = {**ann}
                if image_changed:
                    kept_ann["relabel_action"] = "kept"
                kept.append(kept_ann)
        stats["kept"] += len(kept)

        added, added_record = [], []
        for box_xyxy, score, pred_cls_idx in new_label_preds:
            x1, y1, x2, y2 = box_xyxy
            w, h = float(x2 - x1), float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            cat_id = (class_ids[pred_cls_idx]
                      if pred_cls_idx < len(class_ids)
                      else class_ids[0])
            ann_dict = {
                "id":             -1,   # reassigned in write_relabeled_output
                "image_id":       gid,
                "category_id":    cat_id,
                "bbox":           [float(x1), float(y1), w, h],
                "area":           float(w * h),
                "iscrowd":        0,
                "relabel_action": "added",
                "relabel_conf":   score,
            }
            added.append(ann_dict)
            added_record.append(ann_dict)
        stats["added"] += len(added)

        cleaned_anns[gid] = kept + added

        if image_changed:
            provenance[gid] = {
                "file_name": fname,
                "split":     split,
                "removed":   removed_record,
                "resized":   resized_record,
                "added":     added_record,
            }

    return cleaned_anns, provenance, stats


# ===========================================================================
# Output writing
# ===========================================================================
def write_relabeled_output(
    all_images:     list[dict],
    cleaned_anns:   dict,
    all_provenance: dict,
    categories:     list,
    out_root:       Path,
    log:            logging.Logger,
) -> None:
    """
    Write relabeled COCO JSONs restoring the original train/valid/test split.

    Annotation 'relabel_action' field:
      "kept"    — original box on an image that had at least one change
      "resized" — original box expanded to model's predicted extent
                  (also has 'relabel_conf' and 'relabel_orig_bbox')
      "added"   — new box inserted by the model (also has 'relabel_conf')
      (absent)  — original box on an image the model made no changes to

    provenance.json logs every removed / resized / added annotation per image.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list] = defaultdict(list)
    for img in all_images:
        by_split[img["_split"]].append(img)

    for split, imgs in by_split.items():
        split_dir = (out_root / split).resolve()
        split_dir.mkdir(parents=True, exist_ok=True)

        coco_images, coco_anns = [], []
        ann_id = 1
        for img in imgs:
            clean_img = {k: v for k, v in img.items() if k != "_split"}
            coco_images.append(clean_img)
            for ann in cleaned_anns.get(img["id"], []):
                entry              = {k: v for k, v in ann.items()}
                entry["id"]        = ann_id
                entry["image_id"]  = img["id"]
                coco_anns.append(entry)
                ann_id += 1

        out_path = split_dir / "_annotations.coco.json"
        with open(out_path, "w") as f:
            json.dump({"images": coco_images, "annotations": coco_anns,
                       "categories": categories}, f)
        log.info(f"[output] {split}: {len(coco_images)} images, "
                 f"{len(coco_anns)} annotations → {out_path}")

    # provenance.json
    prov_path = out_root / "provenance.json"
    with open(prov_path, "w") as f:
        json.dump({str(k): v for k, v in all_provenance.items()}, f, indent=2)

    n_changed = len(all_provenance)
    n_removed = sum(len(v["removed"])          for v in all_provenance.values())
    n_resized = sum(len(v.get("resized", [])) for v in all_provenance.values())
    n_added   = sum(len(v["added"])            for v in all_provenance.values())
    log.info(f"[output] provenance.json: {n_changed} images changed "
             f"({n_removed} removed, {n_resized} resized, {n_added} added) → {prov_path}")


# ===========================================================================
# Argument parsing
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="4-fold CV relabeling for LARS maritime dataset")
    p.add_argument("--folds",        type=int,   default=K_FOLDS)
    p.add_argument("--epochs",       type=int,   default=50,
                   help="Training epochs per fold")
    p.add_argument("--resume-fold",  type=int,   default=None,
                   help="Skip folds 0..N-1 (load cache) and start from fold N")
    p.add_argument("--relabel-only", action="store_true",
                   help="Skip training; load existing checkpoints and relabel only")
    p.add_argument("--keep-tmp",     action="store_true",
                   help="Keep temporary fold dataset dirs after training")
    p.add_argument("--output-dir",   default=str(RUNS_DIR),
                   help="Directory for fold checkpoints and cache files")
    p.add_argument("--out-root",     default=str(OUT_ROOT),
                   help="Where to write the final relabeled dataset")
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    args     = parse_args()
    runs_dir = Path(args.output_dir).resolve()
    out_root = Path(args.out_root).resolve()
    log      = setup_logging(runs_dir)

    log.info("=" * 60)
    log.info("CV-Relabel  —  LARS Maritime Dataset")
    log.info("=" * 60)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        log.info(f"Device        : CUDA — {torch.cuda.get_device_name(0)}")
    else:
        log.info("Device        : CPU  (training will be very slow)")

    # ------------------------------------------------------------------
    # 1. Load all splits
    # ------------------------------------------------------------------
    if not DATA_ROOT.exists():
        log.error(f"DATA_ROOT not found: {DATA_ROOT}")
        log.error("Run 2_DataPreprocessing/datasplit.py first.")
        sys.exit(1)

    all_images, anns_by_id, categories = load_all_splits(log)

    log.info(f"Total images  : {len(all_images)}")
    log.info(f"Categories    : {[c['name'] for c in categories]}")
    log.info(f"K folds       : {args.folds}")
    log.info(f"Epochs/fold   : {args.epochs}")
    log.info(f"Hyperparams   : lr={TRIAL004['lr']:.2e}  "
             f"lr_enc={TRIAL004['lr_encoder']:.2e}  "
             f"wd={TRIAL004['weight_decay']:.2e}  "
             f"res={TRIAL004['resolution']}")
    log.info(f"Relabel-only  : {args.relabel_only}")
    log.info(f"Output dir    : {runs_dir}")
    log.info(f"Relabeled out : {out_root}")

    # ------------------------------------------------------------------
    # 2. Build K folds  (scene-level)
    # ------------------------------------------------------------------
    folds = make_folds(all_images, args.folds)
    for i, fold in enumerate(folds):
        split_counts = Counter(img["_split"] for img in fold)
        log.info(f"  Fold {i}: {len(fold)} images  {dict(split_counts)}")

    # ------------------------------------------------------------------
    # 3. Per-fold: dataset → train → relabel
    # ------------------------------------------------------------------
    all_cleaned_anns: dict[int, list] = {}
    all_provenance:   dict[int, dict] = {}

    def fold_cache_path(k: int) -> Path:
        return (runs_dir / f"fold_{k}" / "cleaned_anns.json").resolve()

    for k in range(args.folds):
        fold_run_dir     = (runs_dir / f"fold_{k}").resolve()
        fold_dataset_dir = (fold_run_dir / "dataset").resolve()
        fold_ckpt_dir    = (fold_run_dir / "checkpoint").resolve()

        # ── Resume: load cached results ────────────────────────────────
        if args.resume_fold is not None and k < args.resume_fold:
            cache = fold_cache_path(k)
            if cache.exists():
                log.info(f"\n[Fold {k}] Loading cached annotations from {cache} …")
                raw = json.load(open(cache))
                all_cleaned_anns.update(
                    {int(gid): anns for gid, anns in raw["cleaned_anns"].items()})
                all_provenance.update(
                    {int(gid): p for gid, p in raw.get("provenance", {}).items()})
                continue
            else:
                log.warning(f"[Fold {k}] --resume-fold set but no cache at {cache} "
                            f"— recomputing")

        log.info(f"\n{'='*60}")
        log.info(f"  Fold {k}  ({len(folds[k])} held-out images)")
        log.info(f"{'='*60}")

        # ── Build temporary fold dataset ───────────────────────────────
        log.info(f"[Fold {k}] Building dataset …")
        build_fold_dataset(
            k           = k,
            folds       = folds,
            anns_by_id  = anns_by_id,
            categories  = categories,
            fold_dir    = fold_dataset_dir,
            log         = log,
        )

        # ── Train (unless --relabel-only) ──────────────────────────────
        if not args.relabel_only:
            log.info(f"[Fold {k}] Training …")
            train_fold(fold_dataset_dir, fold_ckpt_dir, args.epochs, log)
        else:
            log.info(f"[Fold {k}] --relabel-only: skipping training")

        # ── Load best checkpoint ───────────────────────────────────────
        log.info(f"[Fold {k}] Loading best checkpoint …")
        model = load_model(fold_ckpt_dir, log)

        # ── Relabel held-out fold k ────────────────────────────────────
        log.info(f"[Fold {k}] Relabeling {len(folds[k])} images …")
        cleaned, provenance, stats = relabel_images(
            fold_imgs  = folds[k],
            anns_by_id = anns_by_id,
            categories = categories,
            model      = model,
            log        = log,
        )
        log.info(f"[Fold {k}] stats: {dict(stats)}")

        # ── Cache to disk ──────────────────────────────────────────────
        cache = fold_cache_path(k)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump({
                "cleaned_anns": {str(gid): anns for gid, anns in cleaned.items()},
                "provenance":   {str(gid): p   for gid, p   in provenance.items()},
            }, f)
        log.info(f"[Fold {k}] cache → {cache}")

        all_cleaned_anns.update(cleaned)
        all_provenance.update(provenance)

        # ── Clean up temporary dataset ─────────────────────────────────
        if not args.keep_tmp and fold_dataset_dir.exists():
            shutil.rmtree(fold_dataset_dir)
            log.info(f"[Fold {k}] removed {fold_dataset_dir}")

    # ------------------------------------------------------------------
    # 4. Pass through any image not covered by relabeling (safety net)
    # ------------------------------------------------------------------
    for img in all_images:
        if img["id"] not in all_cleaned_anns:
            all_cleaned_anns[img["id"]] = anns_by_id.get(img["id"], [])

    # ------------------------------------------------------------------
    # 5. Write relabeled dataset
    # ------------------------------------------------------------------
    write_relabeled_output(
        all_images     = all_images,
        cleaned_anns   = all_cleaned_anns,
        all_provenance = all_provenance,
        categories     = categories,
        out_root       = out_root,
        log            = log,
    )

    # ------------------------------------------------------------------
    # 6. Final summary
    # ------------------------------------------------------------------
    total_removed = sum(len(v["removed"])          for v in all_provenance.values())
    total_resized = sum(len(v.get("resized", [])) for v in all_provenance.values())
    total_added   = sum(len(v["added"])            for v in all_provenance.values())
    total_changed = len(all_provenance)

    log.info("=" * 60)
    log.info("Done.")
    log.info(f"  Images processed    : {len(all_cleaned_anns)}")
    log.info(f"  Images changed      : {total_changed}")
    log.info(f"  Annotations removed : {total_removed}  (ghost + merged)")
    log.info(f"  Annotations resized : {total_resized}  (GT box too small)")
    log.info(f"  Annotations added   : {total_added}  (missing labels)")
    log.info(f"  Relabeled dataset   : {out_root}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
