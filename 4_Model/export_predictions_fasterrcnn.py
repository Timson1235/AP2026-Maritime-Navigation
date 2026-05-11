"""
export_predictions_fasterrcnn.py

Run Faster R-CNN inference on a dataset split and write COCO results JSON.

Usage
-----
    python export_predictions_fasterrcnn.py --checkpoint ../runs/fasterrcnn/baseline/checkpoint_best.pth
    python export_predictions_fasterrcnn.py --checkpoint <path> --split test --threshold 0.2
    python export_predictions_fasterrcnn.py --checkpoint <path> --split val --out my_preds.json

Output format (COCO detection results):
    [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]

category_id values match the IDs in _annotations.coco.json so the evaluation
notebook can consume the file directly.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from train_fasterrcnn import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Export Faster R-CNN predictions to COCO results JSON")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pth checkpoint file")
    p.add_argument("--data-root", default="../Data/lars_processed",
                   help="Dataset root containing {split}/images and {split}/_annotations.coco.json")
    p.add_argument("--split", default="valid", choices=["train", "valid", "test"],
                   help="Dataset split to run inference on (default: valid)")
    p.add_argument("--model", choices=["base", "large"], default="base",
                   help="Model variant — must match the checkpoint (default: base)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Confidence threshold — keep predictions above this value "
                        "(default: 0.0, keep all; the evaluation notebook re-filters)")
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
    CLASS_IDS  = [c["id"] for c in categories]   # original COCO category IDs
    N_CLASSES  = len(CLASS_IDS)

    # Faster R-CNN training maps original IDs → 1-indexed; reverse that here.
    # cat_id_map in train_fasterrcnn: {orig_id: i+1} → we need {i+1: orig_id}
    idx_to_cat = {i + 1: c["id"] for i, c in enumerate(categories)}

    print(f"Split       : {args.split}  ({len(img_meta)} images)")
    print(f"Classes     : {N_CLASSES}  {[c['name'] for c in categories]}")
    print(f"Checkpoint  : {checkpoint}")
    print(f"Output      : {out_path}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    num_classes = N_CLASSES + 1   # +1 for background (matches training)
    model = build_model(args.model, num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # ── Run inference ────────────────────────────────────────────────────────
    results = []

    with torch.no_grad():
        for img_id, meta in tqdm(img_meta.items(), desc="Inference"):
            img_path = images_dir / meta["file_name"]
            tensor   = TF.to_tensor(Image.open(img_path).convert("RGB")).to(device)

            preds = model([tensor])[0]

            boxes      = preds["boxes"].cpu().numpy()    # (N, 4) xyxy
            labels     = preds["labels"].cpu().numpy()   # (N,)   1-indexed
            scores     = preds["scores"].cpu().numpy()   # (N,)

            for box, label, score in zip(boxes, labels, scores):
                if score < args.threshold:
                    continue
                x1, y1, x2, y2 = box.tolist()
                # Map 1-indexed label back to original COCO category ID
                cat_id = idx_to_cat.get(int(label), int(label))
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
