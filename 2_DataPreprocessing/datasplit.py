"""
Reorganises the raw LARS splits into a clean train/test/valid structure:

  train       <- 80% of original train (scene-level, stratified by label fields)
  test        <- 20% of original train (scene-level, stratified by label fields)
  valid       <- original val  (has panoptic / bbox labels)
  test_unused <- original test (no bbox labels)

Stratification ensures each label field (scene_type, lighting, reflections, waves)
is proportionally represented in both train and test. Splitting is scene-level —
no scene appears in more than one split.

Output is written to Data/lars_processed/. Original data is not modified.
"""

import json
import re
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ANN  = Path("../Data/lars_v1.0.0_annotations")
RAW_IMG  = Path("../Data/lars_v1.0.0_images")
OUT_ROOT = Path("../Data/lars_processed")

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------
for split in ("train", "test", "valid", "test_unused"):
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
# Scene-level stratified 80/20 split of original train
# ---------------------------------------------------------------------------
LABEL_FIELDS = ["scene_type", "lighting", "reflections", "waves"]

# Group image annotations by scene
scenes = defaultdict(list)
for ann in train_img_anns:
    scene = re.sub(r"_\d{5}\.jpg$", "", ann["file_name"])
    scenes[scene].append(ann)

scene_names = sorted(scenes.keys())


def dominant_label(anns, field):
    """Most common value for a label field across a scene's images."""
    values = [a["labels"][field] for a in anns if a["labels"].get(field) is not None]
    return Counter(values).most_common(1)[0][0] if values else "unknown"


# Build a combined stratification key per scene from all four label fields
strat_labels = [
    "|".join(dominant_label(scenes[s], f) for f in LABEL_FIELDS)
    for s in scene_names
]

# Merge strata that have only one scene into a single "rare" bucket so
# sklearn's stratified split does not raise a ValueError
strat_counts = Counter(strat_labels)
strat_labels = [
    k if strat_counts[k] >= 2 else "rare"
    for k in strat_labels
]

train_scenes, test_scenes = train_test_split(
    scene_names,
    test_size=0.2,
    random_state=4,
    stratify=strat_labels,
)

train_scenes = set(train_scenes)
test_scenes  = set(test_scenes)

train_anns = [ann for scene in train_scenes for ann in scenes[scene]]
test_anns  = [ann for scene in test_scenes  for ann in scenes[scene]]

print(f"\nScenes  — train: {len(train_scenes)}, test: {len(test_scenes)}")
print(f"Images  — train: {len(train_anns)},  test: {len(test_anns)}")

# Stratification quality check
print("\nStratification check (proportions per label field):")
for field in LABEL_FIELDS:
    train_dist  = Counter(a["labels"][field] for a in train_anns if a["labels"].get(field))
    test_dist   = Counter(a["labels"][field] for a in test_anns  if a["labels"].get(field))
    train_total = sum(train_dist.values())
    test_total  = sum(test_dist.values())
    print(f"  {field}:")
    for value in sorted(set(train_dist) | set(test_dist)):
        tr = train_dist.get(value, 0) / train_total
        te = test_dist.get(value, 0)  / test_total
        print(f"    {value:<22} train={tr:.2f}  test={te:.2f}")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def build_coco_detection(img_anns, pan_json, src_img_dir, out_dir):
    """Copy images and write a COCO detection JSON (thing bboxes only)."""
    img_filenames     = {ann["file_name"] for ann in img_anns}
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

    with open(out_dir / "_annotations.coco.json", "w") as f:
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

build_coco_detection(train_anns,   train_pan, RAW_IMG / "train" / "images", OUT_ROOT / "train")
build_coco_detection(test_anns,    train_pan, RAW_IMG / "train" / "images", OUT_ROOT / "test")
build_coco_detection(val_img_anns, val_pan,   RAW_IMG / "val"   / "images", OUT_ROOT / "valid")

# test_unused — no bbox labels, just images + image-level annotations
with open(OUT_ROOT / "test_unused" / "image_annotations.json", "w") as f:
    json.dump({"annotations": test_img_anns}, f)
for ann in test_img_anns:
    src = RAW_IMG / "test" / "images" / ann["file_name"]
    if src.exists():
        shutil.copy2(src, OUT_ROOT / "test_unused" / "images" / ann["file_name"])
print(f"  test_unused: {len(test_img_anns)} images (no bbox labels)")

print("\nDone.")
