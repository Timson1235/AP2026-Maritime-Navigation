"""
Offline data augmentation for RF-DETR training on the LARS dataset.

Reads Data/lars_processed/train/ and writes augmented copies of every training
image into the same images/ folder. The _annotations.coco.json is updated
in-place (original is backed up as _annotations.coco.original.json first).

Augmentations applied (bounding-box-aware via albumentations):
  HorizontalFlip          — mirrors image and all bboxes  (p=0.5)
  RandomBrightnessContrast — brightness ±0.25, contrast ±0.25  (p=0.7)
  HueSaturationValue      — hue ±5, sat ±40, val ±30  (p=0.6)
  GaussianBlur            — kernel 3–7px, simulates haze / camera blur  (p=0.25)
  CLAHE                   — local contrast enhancement, simulates glare  (p=0.25)

All augmentations are stochastic: each copy of an image may look different
from other copies (and from the original). Bboxes that shrink below 50 px²
or become <10 % visible after a geometric transform are dropped automatically.

Usage
-----
    python rfdetr_preprocessing.py                   # 1 augmented copy per image
    python rfdetr_preprocessing.py --copies 2        # 2 copies  (3× dataset)
    python rfdetr_preprocessing.py --undo            # remove augmented data, restore backup
    python rfdetr_preprocessing.py --dry-run         # print stats, write nothing
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).parent
DATA_ROOT = _HERE / "../Data/lars_processed"
TRAIN_DIR = DATA_ROOT / "train"
IMG_DIR   = TRAIN_DIR / "images"
ANN_FILE  = TRAIN_DIR / "_annotations.coco.json"
ANN_BACKUP = TRAIN_DIR / "_annotations.coco.original.json"

# Sentinel suffix used to mark augmented files — must not appear in original names
AUG_SUFFIX = "_aug"


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------
def build_pipeline() -> A.Compose:
    """
    Returns an albumentations Compose pipeline with COCO bbox support.

    All transforms are maritime-motivated:
    - HorizontalFlip     : boats / buoys can face either direction
    - BrightnessContrast : dawn, dusk, overcast vs. bright sun
    - HueSaturationValue : colour temperature shift (golden hour, fog)
    - GaussianBlur       : sea haze, spray, soft focus
    - CLAHE              : local contrast variation (glitter, reflections)
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.7,
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=40,
                val_shift_limit=30,
                p=0.6,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.25),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.25),
        ],
        bbox_params=A.BboxParams(
            format="coco",          # [x, y, w, h] absolute pixels
            label_fields=["category_ids"],
            min_area=50.0,          # drop bboxes smaller than 50 px²
            min_visibility=0.1,     # drop bboxes less than 10 % visible
        ),
    )


# ---------------------------------------------------------------------------
# Core augmentation logic
# ---------------------------------------------------------------------------
def augment_dataset(copies: int, dry_run: bool, seed: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # Validate
    if not ANN_FILE.exists():
        print(f"ERROR: annotation file not found: {ANN_FILE}")
        print("Run 2_DataPreprocessing/datasplit.py first.")
        sys.exit(1)

    with open(ANN_FILE) as f:
        ann = json.load(f)

    # Check for existing augmented data
    aug_images = [img for img in ann["images"] if AUG_SUFFIX in img["file_name"]]
    if aug_images:
        print(
            f"Dataset already contains {len(aug_images)} augmented image entries.\n"
            f"Run with --undo first to reset, then re-run to re-augment."
        )
        sys.exit(0)

    # Dataset stats before augmentation
    n_orig_images = len(ann["images"])
    n_orig_anns   = len(ann["annotations"])
    print(f"Original dataset   : {n_orig_images:,} images, {n_orig_anns:,} annotations")
    print(f"Copies per image   : {copies}  →  ~{n_orig_images * copies:,} new images")
    if dry_run:
        print("[DRY RUN — no files will be written]")

    # Build per-image annotation lookup
    anns_by_img: dict[int, list] = defaultdict(list)
    for a in ann["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # Starting IDs for new entries
    next_img_id = max(img["id"] for img in ann["images"]) + 1
    next_ann_id = max((a["id"] for a in ann["annotations"]), default=0) + 1

    pipeline = build_pipeline()

    new_images: list[dict] = []
    new_anns:   list[dict] = []
    skipped = 0

    for img_entry in tqdm(ann["images"], desc="Augmenting"):
        img_path = IMG_DIR / Path(img_entry["file_name"]).name
        if not img_path.exists():
            skipped += 1
            continue

        img_arr = np.array(Image.open(img_path).convert("RGB"))

        img_anns  = anns_by_img.get(img_entry["id"], [])
        bboxes    = [a["bbox"] for a in img_anns]      # [[x,y,w,h], ...]
        cat_ids   = [a["category_id"] for a in img_anns]

        for copy_idx in range(copies):
            result     = pipeline(image=img_arr, bboxes=bboxes, category_ids=cat_ids)
            aug_arr    = result["image"]
            aug_bboxes = result["bboxes"]
            aug_cats   = result["category_ids"]

            stem     = Path(img_entry["file_name"]).stem
            aug_fname = f"images/{stem}{AUG_SUFFIX}{copy_idx}.jpg"
            aug_path  = IMG_DIR / f"{stem}{AUG_SUFFIX}{copy_idx}.jpg"

            if not dry_run:
                Image.fromarray(aug_arr).save(aug_path, quality=95)

            new_images.append({
                "id":        next_img_id,
                "width":     img_entry["width"],
                "height":    img_entry["height"],
                "file_name": aug_fname,
            })

            for bbox, cat_id in zip(aug_bboxes, aug_cats):
                x, y, w, h = bbox
                new_anns.append({
                    "id":          next_ann_id,
                    "image_id":    next_img_id,
                    "category_id": int(cat_id),  # albumentations may return float
                    "bbox":        [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                    "area":        round(w * h, 1),
                    "iscrowd":     0,
                })
                next_ann_id += 1

            next_img_id += 1

    # Write updated annotations
    if not dry_run:
        if not ANN_BACKUP.exists():
            shutil.copy2(ANN_FILE, ANN_BACKUP)
            print(f"Backup saved       : {ANN_BACKUP.name}")

        ann["images"]      = ann["images"]      + new_images
        ann["annotations"] = ann["annotations"] + new_anns

        with open(ANN_FILE, "w") as f:
            json.dump(ann, f)

    print(f"\nDone.")
    print(f"  Augmented images written : {len(new_images):,}")
    print(f"  Augmented annotations    : {len(new_anns):,}")
    if skipped:
        print(f"  Images skipped (missing) : {skipped}")
    print(f"  Total images now         : {n_orig_images + len(new_images):,}")
    print(f"  Total annotations now    : {n_orig_anns + len(new_anns):,}")
    if not dry_run:
        print(f"  Annotations file updated : {ANN_FILE.name}")


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------
def undo_augmentation() -> None:
    if not ANN_BACKUP.exists():
        print("No backup found. Nothing to undo.")
        sys.exit(0)

    with open(ANN_BACKUP) as f:
        orig_ann = json.load(f)

    orig_names = {img["file_name"] for img in orig_ann["images"]}

    # Delete augmented image files
    deleted = 0
    for jpg in IMG_DIR.glob("*.jpg"):
        if AUG_SUFFIX in jpg.name:
            jpg.unlink()
            deleted += 1

    # Restore annotation file
    shutil.copy2(ANN_BACKUP, ANN_FILE)
    ANN_BACKUP.unlink()

    print(f"Removed {deleted} augmented image files.")
    print(f"Annotation file restored to {len(orig_ann['images'])} images.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline data augmentation for RF-DETR LARS training."
    )
    p.add_argument(
        "--copies", type=int, default=1,
        help="Number of augmented copies to generate per original image (default: 1)",
    )
    p.add_argument(
        "--undo", action="store_true",
        help="Remove all augmented images and restore the original annotation file",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.undo:
        undo_augmentation()
    else:
        augment_dataset(copies=args.copies, dry_run=args.dry_run)
