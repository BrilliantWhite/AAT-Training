"""Physically plausible augmentation policy for AAT lane images."""

from __future__ import annotations

from typing import Any, Mapping


def validate_augmentation_config(config: Mapping[str, Any]) -> None:
    if bool(config.get("vertical_flip", False)):
        raise ValueError("vertical_flip is prohibited for electrophoresis lanes")
    if bool(config.get("mixup", False)):
        raise ValueError("mixup is prohibited for the preregistered first round")
    if abs(float(config.get("rotation_degrees", 0))) > 2:
        raise ValueError("rotation_degrees must stay within +/-2")
    if float(config.get("crop_scale_min", 1.0)) < 0.9:
        raise ValueError("crop_scale_min would remove physically meaningful bands")


def build_training_transform(config: Mapping[str, Any]):
    """Build tensor-space transforms after validating the physical policy."""

    validate_augmentation_config(config)
    from torchvision import transforms

    rotation = float(config.get("rotation_degrees", 2))
    brightness = float(config.get("brightness", 0.08))
    contrast = float(config.get("contrast", 0.08))
    return transforms.Compose(
        [
            transforms.RandomAffine(rotation, translate=(0.02, 0.02), scale=(0.98, 1.02), fill=1.0),
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=float(config.get("saturation", 0.03))),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, float(config.get("blur_sigma_max", 0.6)))),
        ]
    )

