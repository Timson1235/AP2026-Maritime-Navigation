"""
Reorganises the raw LARS splits into a clean train/val/test structure:

  new train     <- 80% of original train (scene-level split)
  new val       <- 20% of original train (scene-level split)
  test          <- original val  (has panoptic / bbox labels)
  test_unused   <- original test (no bbox labels)

Output is written to Data/lars_processed/. Original data is not modified.
"""

import json
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ANN  = Path("../Data/lars_v1")
RAW_IMG  = Path("../Data/lars_v1-2")
OUT_ROOT = Path("../Data/lars_processed")

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------
for split in ("train", "val", "test", "test_unused"):
    (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load raw annotations
# ---------------------------------------------------------------------------
with open(RAW_ANN / "train" / "image_annotations.json") as f:
    train_img_anns = json.load(f)["annotations"]

with open(RAW_ANN / "train" / "panoptic_annotations.json") as f:
    train_pan = json.load(f)

with open(RAW_ANN / "val" / "image_annotations.json") as f:
    val_img_anns = json.load(f)["annotations"]

with open(RAW_ANN / "val" / "panoptic_annotations.json") as f:
    val_pan = json.load(f)

with open(RAW_ANN / "test" / "image_annotations.json") as f:
    test_img_anns = json.load(f)["annotations"]

# Thing categories only (object detection — no stuff classes)
thing_categories = [c for c in train_pan["categories"] if c["isthing"]]
thing_cat_ids    = {c["id"] for c in thing_categories}
print("Detection categories:", [c["name"] for c in thing_categories])

# ---------------------------------------------------------------------------
# Scene-level 80/20 split of original train
# ---------------------------------------------------------------------------
scenes = defaultdict(list)
for ann in train_img_anns:
    scene = re.sub(r"_\d{5}\.jpg$", "", ann["file_name"])
    scenes[scene].append(ann)

scene_names = sorted(scenes.keys())
random.seed(42)
random.shuffle(scene_names)

split_idx    = int(len(scene_names) * 0.8)
train_scenes = set(scene_names[:split_idx])
val_scenes   = set(scene_names[split_idx:])

train_anns = [ann for scene in train_scenes for ann in scenes[scene]]
val_anns   = [ann for scene in val_scenes   for ann in scenes[scene]]

print(f"Scenes  — train: {len(train_scenes)}, val: {len(val_scenes)}")
print(f"Images  — train: {len(train_anns)}, val: {len(val_anns)}")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def build_coco_detection(img_anns, pan_json, src_img_dir, out_dir):
    """Copy images and write a COCO detection JSON (thing bboxes only)."""
    img_filenames    = {ann["file_name"] for ann in img_anns}
    fname_to_imgentry = {img["file_name"]: img for img in pan_json["images"]}

    coco_images, coco_annotations = [], []
    ann_id = 1

    for pan_ann in pan_json["annotations"]:
        jpg_name = pan_ann["file_name"].replace(".png", ".jpg")
        if jpg_name not in img_filenames:
            continue
        img_entry = fname_to_imgentry.get(jpg_name)
        if img_entry is None:
            continue

        coco_images.append(img_entry)
        for seg in pan_ann["segments_info"]:
            if seg["category_id"] not in thing_cat_ids:
                continue
            coco_annotations.append({
                "id":          ann_id,
                "image_id":    img_entry["id"],
                "category_id": seg["category_id"],
                "bbox":        seg["bbox"],  # [x, y, w, h]
                "area":        seg["area"],
                "iscrowd":     seg["iscrowd"],
            })
            ann_id += 1

    with open(out_dir / "annotations.json", "w") as f:
        json.dump({"images": coco_images,
                   "annotations": coco_annotations,
                   "categories": thing_categories}, f)

    for fn in img_filenames:
        src = src_img_dir / fn
        if src.exists():
            shutil.copy2(src, out_dir / "images" / fn)

    print(f"  {out_dir.name}: {len(img_filenames)} images, {len(coco_annotations)} bboxes")

# ---------------------------------------------------------------------------
# Build splits
# ---------------------------------------------------------------------------
print("\nBuilding processed dataset...")

build_coco_detection(train_anns,  train_pan, RAW_IMG / "train" / "images", OUT_ROOT / "train")
build_coco_detection(val_anns,    train_pan, RAW_IMG / "train" / "images", OUT_ROOT / "val")
build_coco_detection(val_img_anns, val_pan,  RAW_IMG / "val"   / "images", OUT_ROOT / "test")

# test_unused — no bbox labels, just images + image-level annotations
with open(OUT_ROOT / "test_unused" / "image_annotations.json", "w") as f:
    json.dump({"annotations": test_img_anns}, f)
for ann in test_img_anns:
    src = RAW_IMG / "test" / "images" / ann["file_name"]
    if src.exists():
        shutil.copy2(src, OUT_ROOT / "test_unused" / "images" / ann["file_name"])
print(f"  test_unused: {len(test_img_anns)} images (no bbox labels)")

print("\nDone.")
