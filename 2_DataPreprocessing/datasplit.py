"""
Reorganises the raw LARS splits into a clean train/val/test structure:

  new train     <- ~86% of original LARS train (stratified scene-level split)
  new val       <- ~14% of original LARS train (stratified scene-level split)
  test          <- original LARS val  (has panoptic / bbox labels)
  test_unused   <- original LARS test (no bbox labels)

The val/test sizes are intentionally kept comparable (~10% each of the
overall labelled pool), giving a rough 80/10/10 distribution.

Stratification key per scene: scene_type x lighting x primary_thing_category
Scenes (image sequences) are kept intact — no frame from the same sequence
appears in more than one split.

Output is written to Data/lars_processed/. Original data is not modified.
"""

import json
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ANN  = Path("../Data/lars_v1.0.0_annotations")
RAW_IMG  = Path("../Data/lars_v1.0.0_images")
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

# Panoptic segment lookup: jpg filename -> segments_info
pan_segs_lookup = {
    ann["file_name"].replace(".png", ".jpg"): ann["segments_info"]
    for ann in train_pan["annotations"]
}

# ---------------------------------------------------------------------------
# Group images into scenes (sequence-level)
# ---------------------------------------------------------------------------
scenes: dict[str, list] = defaultdict(list)
for ann in train_img_anns:
    scene = re.sub(r"_\d{5}\.jpg$", "", ann["file_name"])
    scenes[scene].append(ann)

# ---------------------------------------------------------------------------
# Per-scene stratification key: scene_type x lighting x primary_thing
# ---------------------------------------------------------------------------
def dominant(values):
    vals = [v for v in values if v is not None]
    return Counter(vals).most_common(1)[0][0] if vals else "unknown"

def primary_thing(scene_anns):
    cat_name = {c["id"]: c["name"] for c in thing_categories}
    counts = Counter(
        cat_name[seg["category_id"]]
        for ann in scene_anns
        for seg in pan_segs_lookup.get(ann["file_name"], [])
        if seg["category_id"] in thing_cat_ids
    )
    return counts.most_common(1)[0][0] if counts else "none"

scene_strata: dict[str, str] = {}
for scene, anns in scenes.items():
    scene_type = dominant(a["labels"]["scene_type"] for a in anns)
    lighting   = dominant(a["labels"]["lighting"]   for a in anns)
    scene_strata[scene] = f"{scene_type}|{lighting}|{primary_thing(anns)}"

# ---------------------------------------------------------------------------
# Stratified ~90/10 scene-level split
# Val target ~14% of LARS train → comparable in size to the fixed test set
# ---------------------------------------------------------------------------
strata_groups: dict[str, list] = defaultdict(list)
for scene, key in scene_strata.items():
    strata_groups[key].append(scene)

random.seed(4)

VAL_RATIO = 0.14  # ~14% of LARS train → val comparable in size to the fixed test set

train_scenes, val_scenes = [], []

for key, group in sorted(strata_groups.items()):
    random.shuffle(group)
    n_val = max(1, round(len(group) * VAL_RATIO)) if len(group) >= 2 else 0
    train_scenes.extend(group[n_val:])
    val_scenes.extend(group[:n_val])

new_train_anns = [ann for s in train_scenes for ann in scenes[s]]
new_val_anns   = [ann for s in val_scenes   for ann in scenes[s]]

total_labelled = len(new_train_anns) + len(new_val_anns) + len(val_img_anns)
print(f"\nScenes  — train: {len(train_scenes)}, val: {len(val_scenes)}")
print(f"Images  — train: {len(new_train_anns)} ({len(new_train_anns)/total_labelled:.1%}), "
      f"val: {len(new_val_anns)} ({len(new_val_anns)/total_labelled:.1%}), "
      f"test: {len(val_img_anns)} ({len(val_img_anns)/total_labelled:.1%})")

# ---------------------------------------------------------------------------
# Helper: write COCO detection JSON and copy images
# ---------------------------------------------------------------------------
def build_coco_detection(img_anns, pan_json, src_img_dir, out_dir):
    """Copy images and write a COCO detection JSON (thing bboxes only)."""
    img_filenames     = {ann["file_name"] for ann in img_anns}
    fname_to_imgentry = {img["file_name"]: img for img in pan_json["images"]}

    coco_images, coco_annotations = [], []
    ann_id = 1

    for pan_ann in pan_json["annotations"]:
        jpg_name  = pan_ann["file_name"].replace(".png", ".jpg")
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
                "bbox":        seg["bbox"],
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

build_coco_detection(new_train_anns, train_pan, RAW_IMG / "train" / "images", OUT_ROOT / "train")
build_coco_detection(new_val_anns,   train_pan, RAW_IMG / "train" / "images", OUT_ROOT / "val")
build_coco_detection(val_img_anns,   val_pan,   RAW_IMG / "val"   / "images", OUT_ROOT / "test")

# test_unused — no bbox labels, just images + image-level annotations
with open(OUT_ROOT / "test_unused" / "image_annotations.json", "w") as f:
    json.dump({"annotations": test_img_anns}, f)
for ann in test_img_anns:
    src = RAW_IMG / "test" / "images" / ann["file_name"]
    if src.exists():
        shutil.copy2(src, OUT_ROOT / "test_unused" / "images" / ann["file_name"])
print(f"  test_unused: {len(test_img_anns)} images (no bbox labels)")

print("\nDone.")
