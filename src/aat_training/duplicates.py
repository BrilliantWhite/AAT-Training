"""Non-destructive exact and perceptual duplicate audit for lane crops."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _average_hash(path: Path, row: Mapping[str, str], size: int = 16) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = image.convert("L")
        if all(row.get(field, "") != "" for field in ("content_left", "content_top", "content_right", "content_bottom")):
            box = tuple(int(row[field]) for field in ("content_left", "content_top", "content_right", "content_bottom"))
            grayscale = grayscale.crop(box)
        pixels = np.asarray(grayscale.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    return (pixels >= pixels.mean()).reshape(-1)


def audit_crop_duplicates(rows: Sequence[Mapping[str, str]], inputs_dir: Path, near_hamming_threshold: int = 4) -> dict[str, object]:
    """Report, but never remove, cross-gel exact and near crop pairs."""

    inputs_dir = Path(inputs_dir)
    records = []
    for row in rows:
        path = inputs_dir / row["crop_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({
            "lane_id": row["lane_id"], "parent_gel": row["parent_gel"], "sha256": _sha256(path), "hash": _average_hash(path, row),
        })
    exact, near = [], []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            if left["parent_gel"] == right["parent_gel"]:
                continue
            item = {"lane_a": left["lane_id"], "gel_a": left["parent_gel"], "lane_b": right["lane_id"], "gel_b": right["parent_gel"]}
            if left["sha256"] == right["sha256"]:
                exact.append({**item, "sha256": left["sha256"]})
                continue
            distance = int(np.count_nonzero(left["hash"] != right["hash"]))
            if distance <= near_hamming_threshold:
                near.append({**item, "hamming_distance": distance})
    return {
        "schema_version": "aat-duplicate-audit-v1",
        "lane_count": len(records),
        "near_hamming_threshold": near_hamming_threshold,
        "exact_cross_gel_pairs": exact,
        "near_cross_gel_pairs": near,
        "automatic_exclusions": [],
    }
