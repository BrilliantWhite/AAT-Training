"""Build immutable, audited training snapshots from live review exports."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from PIL import Image

from .labels import LabelPolicy, decide_label, load_label_policy


INVENTORY_RELATIVE_PATH = Path("dataset/metadata/image_inventory.csv")
LANES_RELATIVE_PATH = Path("Web/review_exports/training_lanes_export.csv")
SNAPSHOT_SCHEMA_VERSION = "aat-training-snapshot-v1"

LANE_OUTPUT_FIELDS = (
    "lane_id",
    "image_id",
    "parent_gel",
    "source_filename",
    "candidate_index",
    "roi_y_start",
    "roi_y_end",
    "left_x",
    "right_x",
    "x1",
    "y1",
    "x2",
    "y2",
    "x3",
    "y3",
    "x4",
    "y4",
    "original_label",
    "canonical_label",
    "common_eligible",
    "alleles",
    "retrieval_eligible",
    "referral_qc_eligible",
    "qc_status",
    "label_reason",
    "label_version",
    "updated_at",
)

IMAGE_OUTPUT_FIELDS = ("image_id", "source_filename", "relative_path", "width", "height", "channels", "sha256")


class SnapshotValidationError(ValueError):
    """Raised when live inputs cannot safely form a versioned snapshot."""


@dataclass(frozen=True)
class SnapshotBuildResult:
    output_dir: Path
    version: str
    image_count: int
    lane_count: int
    excluded_label_count: int
    formal: bool


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SnapshotValidationError(f"Required CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_int(row: dict[str, str], field: str, lane_key: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotValidationError(f"Invalid integer {field} for {lane_key}") from exc


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=tuple(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_inventory(rows: list[dict[str, str]], source_root: Path) -> tuple[dict[tuple[str, str], dict[str, str]], list[dict[str, object]], dict[str, dict[str, object]]]:
    inventory: dict[tuple[str, str], dict[str, str]] = {}
    image_rows: list[dict[str, object]] = []
    image_manifest: dict[str, dict[str, object]] = {}
    seen_ids: set[str] = set()

    for row in sorted(rows, key=lambda value: (value.get("image_id", ""), value.get("source_filename", ""))):
        image_id = row.get("image_id", "").strip()
        filename = row.get("source_filename", "").strip()
        if not image_id or not filename:
            raise SnapshotValidationError("Inventory contains an empty image_id or source_filename")
        if image_id in seen_ids:
            raise SnapshotValidationError(f"Duplicate inventory image_id: {image_id}")
        seen_ids.add(image_id)
        relative_path = row.get("relative_path", "").strip()
        image_path = source_root / relative_path
        if not image_path.is_file():
            raise SnapshotValidationError(f"Missing image file for {image_id}: {image_path}")
        try:
            expected_width = int(row["width"])
            expected_height = int(row["height"])
            channels = int(row.get("channels", "3"))
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotValidationError(f"Invalid inventory dimensions for {image_id}") from exc
        with Image.open(image_path) as image:
            if image.size != (expected_width, expected_height):
                raise SnapshotValidationError(
                    f"Inventory/image dimension mismatch for {image_id}: inventory={(expected_width, expected_height)} image={image.size}"
                )
        digest = _sha256(image_path)
        key = (image_id, filename)
        inventory[key] = row
        output_row = {
            "image_id": image_id,
            "source_filename": filename,
            "relative_path": relative_path.replace("\\", "/"),
            "width": expected_width,
            "height": expected_height,
            "channels": channels,
            "sha256": digest,
        }
        image_rows.append(output_row)
        image_manifest[image_id] = {
            "source_filename": filename,
            "relative_path": output_row["relative_path"],
            "sha256": digest,
            "width": expected_width,
            "height": expected_height,
        }
    return inventory, image_rows, image_manifest


def _validate_lane_geometry(row: dict[str, str], inventory_row: dict[str, str], lane_key: str) -> dict[str, int]:
    width = int(inventory_row["width"])
    height = int(inventory_row["height"])
    values = {field: _parse_int(row, field, lane_key) for field in ("roi_y_start", "roi_y_end", "left_x", "right_x", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")}
    if not (0 <= values["roi_y_start"] < values["roi_y_end"] <= height):
        raise SnapshotValidationError(f"out of bounds ROI for {lane_key}")
    if not (0 <= values["left_x"] < values["right_x"] <= width):
        raise SnapshotValidationError(f"out of bounds lane bounds for {lane_key}")
    for index in range(1, 5):
        if not (0 <= values[f"x{index}"] <= width and 0 <= values[f"y{index}"] <= height):
            raise SnapshotValidationError(f"out of bounds polygon for {lane_key}")
    points = [(values[f"x{index}"], values[f"y{index}"]) for index in range(1, 5)]
    expected_points = [
        (values["left_x"], values["roi_y_start"]),
        (values["right_x"], values["roi_y_start"]),
        (values["left_x"], values["roi_y_end"]),
        (values["right_x"], values["roi_y_end"]),
    ]
    if points != expected_points:
        raise SnapshotValidationError(f"export rectangle corners do not match ROI/bounds for {lane_key}")
    return values


def build_snapshot(
    source_root: Path,
    output_dir: Path,
    version: str,
    freeze: bool,
    created_utc: str,
    label_policy_path: Path | None = None,
) -> SnapshotBuildResult:
    """Audit live exports once and atomically create a non-overwritable version."""

    source_root = Path(source_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Snapshot output already exists: {output_dir}")
    inventory_path = source_root / INVENTORY_RELATIVE_PATH
    lanes_path = source_root / LANES_RELATIVE_PATH
    policy = load_label_policy(label_policy_path) if label_policy_path else LabelPolicy()
    inventory_rows = _read_csv(inventory_path)
    live_lanes = _read_csv(lanes_path)
    inventory, image_rows, image_manifest = _validate_inventory(inventory_rows, source_root)

    lane_rows: list[dict[str, object]] = []
    seen_lane_keys: set[tuple[str, str]] = set()
    label_counter: Counter[tuple[str, str, str]] = Counter()
    excluded = 0
    unresolved = 0

    def lane_sort_key(row: dict[str, str]) -> tuple[str, int]:
        try:
            index = int(row.get("candidate_index", ""))
        except ValueError:
            index = -1
        return row.get("image_id", ""), index

    for row in sorted(live_lanes, key=lane_sort_key):
        image_id = row.get("image_id", "").strip()
        filename = row.get("source_filename", "").strip()
        inventory_row = inventory.get((image_id, filename))
        if inventory_row is None:
            raise SnapshotValidationError(
                f"Lane must match current inventory by image_id and source_filename: {image_id!r}, {filename!r}"
            )
        candidate_index = row.get("candidate_index", "").strip()
        lane_key = (image_id, candidate_index)
        if lane_key in seen_lane_keys:
            raise SnapshotValidationError(f"duplicate lane key: {image_id}/{candidate_index}")
        seen_lane_keys.add(lane_key)
        label = row.get("label", "").strip()
        if not label:
            raise SnapshotValidationError(f"empty label for {image_id}/{candidate_index}")
        geometry = _validate_lane_geometry(row, inventory_row, f"{image_id}/{candidate_index}")
        decision = decide_label(label, policy)
        if decision.qc_status == "excluded":
            excluded += 1
            if decision.reason not in {"approved_exclusion", "approved_unknown_exclusion"}:
                unresolved += 1
        label_counter[(decision.original_label, decision.canonical_label or "", decision.qc_status)] += 1
        lane_rows.append(
            {
                "lane_id": f"{image_id}_L{int(candidate_index):03d}",
                "image_id": image_id,
                "parent_gel": image_id,
                "source_filename": filename,
                "candidate_index": int(candidate_index),
                **geometry,
                "original_label": decision.original_label,
                "canonical_label": decision.canonical_label or "",
                "common_eligible": int(decision.common_eligible),
                "alleles": ";".join(decision.alleles),
                "retrieval_eligible": int(decision.retrieval_eligible),
                "referral_qc_eligible": int(decision.referral_qc_eligible),
                "qc_status": decision.qc_status,
                "label_reason": decision.reason,
                "label_version": decision.label_version,
                "updated_at": row.get("updated_at", "").strip(),
            }
        )

    if freeze and unresolved:
        raise SnapshotValidationError(f"Formal freeze has {unresolved} unresolved labels; approve mappings/exclusions before frozen_v1")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
    try:
        temp_dir.mkdir()
        source_dir = temp_dir / "source"
        source_dir.mkdir()
        shutil.copyfile(inventory_path, source_dir / "image_inventory.csv")
        shutil.copyfile(lanes_path, source_dir / "training_lanes_export.csv")
        _write_csv(temp_dir / "images.csv", IMAGE_OUTPUT_FIELDS, image_rows)
        _write_csv(temp_dir / "lanes.csv", LANE_OUTPUT_FIELDS, lane_rows)
        summary_rows = [
            {"original_label": original, "canonical_label": canonical, "qc_status": status, "count": count}
            for (original, canonical, status), count in sorted(label_counter.items())
        ]
        _write_csv(temp_dir / "label_summary.csv", ("original_label", "canonical_label", "qc_status", "count"), summary_rows)
        audit = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "version": version,
            "formal": freeze,
            "image_count": len(image_rows),
            "lane_count": len(lane_rows),
            "excluded_label_count": excluded,
            "unresolved_label_count": unresolved,
            "common_eligible_count": sum(int(row["common_eligible"]) for row in lane_rows),
            "retrieval_eligible_count": sum(int(row["retrieval_eligible"]) for row in lane_rows),
            "errors": [],
            "warnings": ([f"{excluded} labels are excluded by the versioned policy"] if excluded else []),
        }
        (temp_dir / "audit_report.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tracked_files = sorted(path for path in temp_dir.rglob("*") if path.is_file())
        files = {
            path.relative_to(temp_dir).as_posix(): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in tracked_files
        }
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "version": version,
            "created_utc": created_utc,
            "immutable": True,
            "formal": freeze,
            "source_root": str(source_root),
            "label_policy_version": policy.label_version,
            "files": files,
            "images": image_manifest,
        }
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return SnapshotBuildResult(output_dir, version, len(image_rows), len(lane_rows), excluded, freeze)


def load_snapshot_lanes(snapshot_dir: Path) -> list[dict[str, str]]:
    """Load and hash-verify copied lane metadata without touching live exports."""

    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    lanes_path = snapshot_dir / "lanes.csv"
    expected_hash = manifest["files"]["lanes.csv"]["sha256"]
    if _sha256(lanes_path) != expected_hash:
        raise SnapshotValidationError(f"Snapshot lanes hash mismatch: {lanes_path}")
    return _read_csv(lanes_path)
