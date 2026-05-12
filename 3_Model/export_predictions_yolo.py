"""
export_predictions_yolo.py

Run YOLO inference on a dataset split and write COCO results JSON.

Output format:
[{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Export YOLO predictions to COCO results JSON")
    p.add_argument("--checkpoint", required=True,
                   help="Path to YOLO best.pt file")
    p.add_argument("--data-root", default="../Data/lars_processed",
                   help="Dataset root containing {split}/images and {split}/_annotations.coco.json")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"],
                   help="Dataset split to run inference on")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Confidence threshold. Use 0.0 because evaluation notebook filters later.")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="YOLO inference image size")
    p.add_argument("--out", default=None,
                   help="Output JSON path. Defaults to <checkpoint_dir>/predictions_<split>.json")
    return p.parse_args()


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    ann_file = split_dir / "_annotations.coco.json"
    images_dir = split_dir / "images"

    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    out_path = Path(args.out) if args.out else checkpoint.parent / f"predictions_{args.split}.json"

    ann = json.load(open(ann_file))
    img_meta = {img["id"]: img for img in ann["images"]}
    categories = ann["categories"]
    class_ids = [c["id"] for c in categories]

    print(f"Split      : {args.split} ({len(img_meta)} images)")
    print(f"Classes    : {len(class_ids)} {[c['name'] for c in categories]}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Output     : {out_path}")

    model = YOLO(str(checkpoint))

    results = []

    for img_id, meta in tqdm(img_meta.items(), desc="YOLO inference"):
        img_path = images_dir / meta["file_name"]

        preds = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.threshold,
            verbose=False
        )[0]

        if preds.boxes is None or len(preds.boxes) == 0:
            continue

        boxes = preds.boxes.xyxy.cpu().numpy()
        labels = preds.boxes.cls.cpu().numpy().astype(int)
        scores = preds.boxes.conf.cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.tolist()

            # YOLO class labels are 0-indexed, so map them back to original COCO category IDs.
            cat_id = class_ids[int(label)]

            results.append({
                "image_id": int(img_id),
                "category_id": int(cat_id),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)

    print(f"\nTotal predictions: {len(results)}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()