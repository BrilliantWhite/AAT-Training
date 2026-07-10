"""Traceable lane crops, RGB letterboxing, 1D profiles, and QC artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from .snapshot import SnapshotValidationError, load_snapshot_lanes


PREPROCESSING_SCHEMA_VERSION = "aat-preprocessing-v1"
INPUT_FIELDS = (
    "lane_id",
    "image_id",
    "parent_gel",
    "source_filename",
    "snapshot_version",
    "snapshot_lane_index",
    "crop_path",
    "profile_index",
    "source_image_sha256",
    "left_x",
    "right_x",
    "roi_y_start",
    "roi_y_end",
    "scale",
    "resized_height",
    "resized_width",
    "pad_top",
    "pad_bottom",
    "pad_left",
    "pad_right",
    "content_left",
    "content_top",
    "content_right",
    "content_bottom",
    "canonical_label",
    "common_eligible",
    "qc_status",
)


@dataclass(frozen=True)
class LetterboxMetadata:
    scale: float
    resized_height: int
    resized_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    @property
    def content_box(self) -> tuple[int, int, int, int]:
        return (
            self.pad_left,
            self.pad_top,
            self.pad_left + self.resized_width,
            self.pad_top + self.resized_height,
        )


@dataclass(frozen=True)
class InputBuildResult:
    output_dir: Path
    lane_count: int
    profile_length: int
    target_height: int
    target_width: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_lane(image: np.ndarray, lane_row: dict[str, str]) -> np.ndarray:
    """Return the exact reviewed rectangular lane crop."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {image.shape}")
    try:
        left = int(lane_row["left_x"])
        right = int(lane_row["right_x"])
        top = int(lane_row["roi_y_start"])
        bottom = int(lane_row["roi_y_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid lane geometry fields") from exc
    height, width = image.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"Invalid lane geometry ({left}, {top}, {right}, {bottom}) for image {width}x{height}")
    return image[top:bottom, left:right, :].copy()


def letterbox_rgb(
    image: np.ndarray,
    target_height: int = 128,
    target_width: int = 384,
    pad_value: int = 255,
) -> tuple[np.ndarray, LetterboxMetadata]:
    """Resize without distortion and center-pad to the requested HxW shape."""

    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Expected non-empty RGB image, got shape {image.shape}")
    if target_height <= 0 or target_width <= 0 or not 0 <= pad_value <= 255:
        raise ValueError("Invalid letterbox target or pad value")
    source_height, source_width = image.shape[:2]
    scale = min(target_height / source_height, target_width / source_width)
    resized_height = max(1, min(target_height, round(source_height * scale)))
    resized_width = max(1, min(target_width, round(source_width * scale)))
    resized = np.asarray(
        Image.fromarray(image).resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    pad_height = target_height - resized_height
    pad_width = target_width - resized_width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    output = np.full((target_height, target_width, 3), pad_value, dtype=np.uint8)
    output[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width, :] = resized
    return output, LetterboxMetadata(scale, resized_height, resized_width, pad_top, pad_bottom, pad_left, pad_right)


def extract_intensity_profile(
    letterboxed: np.ndarray,
    content_box: tuple[int, int, int, int],
    output_length: int = 128,
) -> np.ndarray:
    """Return normalized dark-band intensity by vertical position, excluding padding."""

    if letterboxed.ndim != 3 or letterboxed.shape[2] != 3:
        raise ValueError(f"Expected RGB letterboxed image, got shape {letterboxed.shape}")
    left, top, right, bottom = content_box
    height, width = letterboxed.shape[:2]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height) or output_length <= 0:
        raise ValueError("Invalid content box or output length")
    content = letterboxed[top:bottom, left:right, :].astype(np.float32)
    luminance = np.tensordot(content, np.array([0.299, 0.587, 0.114], dtype=np.float32), axes=([2], [0]))
    profile = 1.0 - luminance.mean(axis=1) / 255.0
    profile = np.clip(profile, 0.0, 1.0)
    if profile.size != output_length:
        source_positions = np.linspace(0.0, 1.0, num=profile.size)
        target_positions = np.linspace(0.0, 1.0, num=output_length)
        profile = np.interp(target_positions, source_positions, profile)
    return profile.astype(np.float32)


def _load_verified_image(path: Path, expected_hash: str) -> np.ndarray:
    if not path.is_file():
        raise SnapshotValidationError(f"Snapshot source image is missing: {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise SnapshotValidationError(f"Snapshot source image hash mismatch: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _write_inputs_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=INPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_montage(path: Path, crops: list[tuple[str, np.ndarray]], columns: int = 4) -> None:
    if not crops:
        Image.new("RGB", (384, 128), color="white").save(path)
        return
    tile_width, tile_height = 384, 148
    rows = (len(crops) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), color="white")
    draw = ImageDraw.Draw(canvas)
    for index, (lane_id, crop) in enumerate(crops):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(Image.fromarray(crop), (x, y))
        draw.text((x + 4, y + 130), lane_id, fill="black")
    canvas.save(path)


def build_training_inputs(
    snapshot_dir: Path,
    output_dir: Path,
    target_height: int = 128,
    target_width: int = 384,
    pad_value: int = 255,
    profile_length: int = 128,
    image_root: Path | None = None,
    limit: int | None = None,
) -> InputBuildResult:
    """Build crops/profiles from copied snapshot metadata and hash-verified raw images."""

    snapshot_dir = Path(snapshot_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Training input output already exists: {output_dir}")
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lanes = load_snapshot_lanes(snapshot_dir)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        lanes = lanes[:limit]
    source_root = Path(image_root).resolve() if image_root else Path(manifest["source_root"])
    version = str(manifest["version"])
    images: dict[str, dict[str, object]] = manifest["images"]
    temp_dir = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
    image_cache: dict[str, np.ndarray] = {}
    input_rows: list[dict[str, object]] = []
    profiles: list[np.ndarray] = []
    montage_crops: list[tuple[str, np.ndarray]] = []

    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        (temp_dir / "crops").mkdir(parents=True)
        for row_index, lane in enumerate(lanes):
            image_id = lane["image_id"]
            image_info = images.get(image_id)
            if image_info is None:
                raise SnapshotValidationError(f"Image {image_id} is missing from snapshot manifest")
            if image_id not in image_cache:
                image_path = source_root / str(image_info["relative_path"])
                image_cache[image_id] = _load_verified_image(image_path, str(image_info["sha256"]))
            crop = crop_lane(image_cache[image_id], lane)
            letterboxed, metadata = letterbox_rgb(crop, target_height, target_width, pad_value)
            profile = extract_intensity_profile(letterboxed, metadata.content_box, profile_length)
            lane_id = lane["lane_id"]
            crop_relative = Path("crops") / f"{lane_id}.png"
            Image.fromarray(letterboxed).save(temp_dir / crop_relative)
            profiles.append(profile)
            if len(montage_crops) < 24:
                montage_crops.append((lane_id, letterboxed))
            content_left, content_top, content_right, content_bottom = metadata.content_box
            input_rows.append(
                {
                    "lane_id": lane_id,
                    "image_id": image_id,
                    "parent_gel": lane["parent_gel"],
                    "source_filename": lane["source_filename"],
                    "snapshot_version": version,
                    "snapshot_lane_index": row_index,
                    "crop_path": crop_relative.as_posix(),
                    "profile_index": row_index,
                    "source_image_sha256": image_info["sha256"],
                    "left_x": lane["left_x"],
                    "right_x": lane["right_x"],
                    "roi_y_start": lane["roi_y_start"],
                    "roi_y_end": lane["roi_y_end"],
                    **asdict(metadata),
                    "content_left": content_left,
                    "content_top": content_top,
                    "content_right": content_right,
                    "content_bottom": content_bottom,
                    "canonical_label": lane.get("canonical_label", ""),
                    "common_eligible": lane.get("common_eligible", "0"),
                    "qc_status": lane.get("qc_status", ""),
                }
            )
        _write_inputs_csv(temp_dir / "inputs.csv", input_rows)
        profile_array = np.stack(profiles) if profiles else np.empty((0, profile_length), dtype=np.float32)
        np.savez_compressed(temp_dir / "profiles.npz", profiles=profile_array, lane_ids=np.array([row["lane_id"] for row in input_rows]))
        _write_montage(temp_dir / "qc_montage.png", montage_crops)
        summary = {
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "snapshot_version": version,
            "snapshot_manifest_sha256": _sha256(manifest_path),
            "lane_count": len(input_rows),
            "profile_length": profile_length,
            "target_shape": [target_height, target_width, 3],
            "pad_value": pad_value,
            "source_image_count": len(image_cache),
        }
        (temp_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = {
            path.relative_to(temp_dir).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(temp_dir.rglob("*"))
            if path.is_file()
        }
        derived_manifest = {**summary, "immutable": True, "files": files}
        (temp_dir / "manifest.json").write_text(json.dumps(derived_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return InputBuildResult(output_dir, len(input_rows), profile_length, target_height, target_width)
