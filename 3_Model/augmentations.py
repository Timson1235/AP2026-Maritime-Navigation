"""
Shared maritime-augmentation policies for Optuna search scripts.

The previous per-knob search space (~26 individual Albumentations parameters)
was too high-dimensional for TPE on small trial counts. Collapsing it to a
single categorical `aug_policy` keeps the meaningful axes — none vs. light
colour vs. heavy weather vs. sensor/occlusion — while giving Optuna a
tractable choice.
"""

import albumentations as A

AUG_POLICIES = ("none", "light_color", "heavy_weather", "sensor_noise_and_occlusion")


def get_maritime_augmentations(aug_policy: str) -> A.Compose:
    transforms = [A.HorizontalFlip(p=0.5)]

    if aug_policy == "none":
        pass
    elif aug_policy == "light_color":
        transforms.extend([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ])
    elif aug_policy == "heavy_weather":
        transforms.extend([
            A.RandomFog(fog_coef_range=(0.05, 0.2), p=0.4),
            A.RandomRain(slant_range=(-10, 10), drop_length=8, p=0.3),
            A.RandomSunFlare(src_radius=100, p=0.2),
        ])
    elif aug_policy == "sensor_noise_and_occlusion":
        transforms.extend([
            A.ISONoise(intensity=(0.1, 0.3), p=0.4),
            A.ImageCompression(quality_range=(50, 80), p=0.4),
            A.CoarseDropout(
                num_holes_range=(1, 5),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                p=0.4,
            ),
        ])
    else:
        raise ValueError(
            f"Unknown aug_policy {aug_policy!r}. Expected one of {AUG_POLICIES}."
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_area=50.0,
            min_visibility=0.1,
        ),
    )
