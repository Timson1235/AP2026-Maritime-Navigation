"""
export_predictions.py

Run RF-DETR inference on a dataset split and write COCO results JSON.

Usage
-----
    python export_predictions.py --checkpoint ../runs/optuna/trial_010/checkpoint_best_total.pth
    python export_predictions.py --checkpoint <path> --split test --threshold 0.2
    python export_predictions.py --checkpoint <path> --split val --out my_preds.json

Output format (COCO detection results):
    [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]

category_id values match the IDs in _annotations.coco.json so the evaluation
notebook can consume the file directly.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from rfdetr import RFDETRBase


def parse_args():
    p = argparse.ArgumentParser(description="Export RF-DETR predictions to COCO results JSON")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth checkpoint file")
    p.add_argument("--data-root", default="../Data/lars_processed",
                   help="Dataset root containing {split}/images and {split}/_annotations.coco.json")
    p.add_argument("--split", default="valid", choices=["train", "valid", "test"],
                   help="Dataset split to run inference on (default: valid)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Confidence threshold — keep predictions above this value "
                        "(default: 0.0, keep all; the evaluation notebook re-filters)")
    p.add_argument("--resolution", type=int, default=728,
                   help="Model input resolution (must match training, default: 728)")
    p.add_argument("--out", default=None,
                   help="Output JSON path. Defaults to <checkpoint_dir>/predictions_<split>.json")
    return p.parse_args()


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data_root  = Path(args.data_root)
    split_dir  = data_root / args.split
    ann_file   = split_dir / "_annotations.coco.json"
    images_dir = split_dir / "images"

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    out_path = Path(args.out) if args.out else checkpoint.parent / f"predictions_{args.split}.json"

    # ── Load annotations ────────────────────────────────────────────────────
    ann        = json.load(open(ann_file))
    img_meta   = {img["id"]: img for img in ann["images"]}
    categories = ann["categories"]
    CLASS_IDS  = [c["id"] for c in categories]  # original COCO category IDs
    N_CLASSES  = len(CLASS_IDS)

    print(f"Split       : {args.split}  ({len(img_meta)} images)")
    print(f"Classes     : {N_CLASSES}  {[c['name'] for c in categories]}")
    print(f"Checkpoint  : {checkpoint}")
    print(f"Output      : {out_path}")

    # ── Load model ──────────────────────────────────────────────────────────
    model = RFDETRBase(
        resolution=args.resolution,
        num_classes=N_CLASSES,
        pretrain_weights=str(checkpoint),
    )

    # ── Run inference ───────────────────────────────────────────────────────
    results = []

    for img_id, meta in tqdm(img_meta.items(), desc="Inference"):
        img_path = images_dir / meta["file_name"]
        pil_img  = Image.open(img_path).convert("RGB")

        dets = model.predict(pil_img, threshold=args.threshold)

        if len(dets) == 0:
            continue

        boxes      = dets.xyxy           # (N, 4) xyxy
        class_idxs = dets.class_id       # (N,)   0-indexed
        scores     = dets.confidence if dets.confidence is not None else np.ones(len(dets))

        for box, cls_idx, score in zip(boxes, class_idxs, scores):
            x1, y1, x2, y2 = box.tolist()
            # RF-DETR outputs 0-indexed classes; map back to original COCO category IDs
            cat_id = CLASS_IDS[int(cls_idx)] if int(cls_idx) < N_CLASSES else int(cls_idx)
            results.append({
                "image_id":    int(img_id),
                "category_id": int(cat_id),
                "bbox":        [x1, y1, x2 - x1, y2 - y1],  # xywh
                "score":       float(score),
            })

    print(f"\nTotal predictions: {len(results)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
