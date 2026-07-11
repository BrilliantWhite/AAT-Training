"""Atomic, fail-closed completed-outer-fold artifacts for CNN resume."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .labels import COMMON_CLASSES


SCHEMA_VERSION = "aat-cnn-completed-fold-v1"
PROBABILITY_FIELDS = tuple(f"prob_{label}" for label in COMMON_CLASSES)


@dataclass(frozen=True)
class CompletedFold:
    outer_fold: int
    predictions: list[dict[str, Any]]
    summary: dict[str, Any]
    checkpoint_path: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        raise FileExistsError(f"Completed-fold bundle already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def write_completed_fold_bundle(
    run_dir: Path,
    *,
    outer_fold: int,
    predictions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    checkpoint_path: Path,
    provenance: Mapping[str, Any],
    resolved_config_sha256: str,
) -> Path:
    run_dir = Path(run_dir).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if run_dir not in checkpoint_path.parents or not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise ValueError("Completed fold checkpoint is missing or outside the experiment")
    relative_checkpoint = checkpoint_path.relative_to(run_dir).as_posix()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "outer_fold": int(outer_fold),
        "provenance": dict(provenance),
        "resolved_config_sha256": str(resolved_config_sha256),
        "checkpoint": {
            "path": relative_checkpoint,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path),
        },
        "summary": dict(summary),
        "predictions": [dict(row) for row in predictions],
    }
    path = run_dir / "folds" / f"outer_fold_{int(outer_fold)}.json"
    _atomic_write_json(path, payload)
    return path


def _normalise_candidate(candidate: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(candidate[key]) for key in ("learning_rate", "weight_decay", "dropout")}


def load_completed_fold_bundles(
    run_dir: Path,
    *,
    expected_outer_folds: set[int],
    expected_lane_ids_by_fold: Mapping[int, set[str]],
    authoritative_records: Mapping[str, Mapping[str, str]],
    expected_provenance: Mapping[str, Any],
    expected_config_sha256: str,
    candidate_grid: Sequence[Mapping[str, Any]],
) -> list[CompletedFold]:
    run_dir = Path(run_dir).resolve()
    folds_dir = run_dir / "folds"
    if not folds_dir.exists():
        return []
    allowed_candidates = [_normalise_candidate(candidate) for candidate in candidate_grid]
    bundles: list[CompletedFold] = []
    seen_lanes: set[str] = set()
    for path in sorted(folds_dir.glob("outer_fold_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Completed-fold schema mismatch: {path.name}")
        outer_fold = int(payload.get("outer_fold", -1))
        expected_name = f"outer_fold_{outer_fold}.json"
        if path.name != expected_name or outer_fold not in expected_outer_folds:
            raise ValueError(f"Unexpected completed outer fold: {path.name}")
        for field, expected in expected_provenance.items():
            if payload.get("provenance", {}).get(field) != expected:
                raise ValueError(f"Completed fold provenance mismatch for {field}: {path.name}")
        if payload.get("resolved_config_sha256") != expected_config_sha256:
            raise ValueError(f"Completed fold resolved config mismatch: {path.name}")
        checkpoint_info = payload.get("checkpoint", {})
        checkpoint_path = (run_dir / str(checkpoint_info.get("path", ""))).resolve()
        if run_dir not in checkpoint_path.parents or not checkpoint_path.is_file():
            raise ValueError(f"Completed fold checkpoint is missing: {path.name}")
        if checkpoint_path.stat().st_size != checkpoint_info.get("bytes") or file_sha256(checkpoint_path) != checkpoint_info.get("sha256"):
            raise ValueError(f"Completed fold checkpoint hash mismatch: {path.name}")
        summary = dict(payload.get("summary", {}))
        selected = summary.get("selected", {}).get("params", {})
        if _normalise_candidate(selected) not in allowed_candidates:
            raise ValueError(f"Completed fold selected candidate is not configured: {path.name}")
        predictions = [dict(row) for row in payload.get("predictions", [])]
        actual_ids = [str(row.get("lane_id", "")) for row in predictions]
        expected_ids = set(expected_lane_ids_by_fold[outer_fold])
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
            raise ValueError(f"Completed fold lane coverage mismatch: {path.name}")
        if len(set(actual_ids)) != len(actual_ids) or seen_lanes.intersection(actual_ids):
            raise ValueError(f"Completed fold has duplicate lane IDs: {path.name}")
        for row in predictions:
            lane_id = str(row["lane_id"])
            record = authoritative_records[lane_id]
            if str(row.get("parent_gel")) != str(record["parent_gel"]):
                raise ValueError(f"Completed fold parent gel mismatch: {lane_id}")
            if str(row.get("true_label")) != str(record["canonical_label"]):
                raise ValueError(f"Completed fold true label mismatch: {lane_id}")
            if int(row.get("outer_fold", -1)) != outer_fold:
                raise ValueError(f"Completed fold prediction fold mismatch: {lane_id}")
            for field, expected in expected_provenance.items():
                if field in row and row[field] != expected:
                    raise ValueError(f"Completed fold prediction provenance mismatch for {field}: {lane_id}")
            probabilities = [float(row[field]) for field in PROBABILITY_FIELDS]
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-5:
                raise ValueError(f"Completed fold probability mismatch: {lane_id}")
            if str(row.get("predicted_label")) != COMMON_CLASSES[max(range(len(probabilities)), key=probabilities.__getitem__)]:
                raise ValueError(f"Completed fold predicted label mismatch: {lane_id}")
        seen_lanes.update(actual_ids)
        bundles.append(CompletedFold(outer_fold, predictions, summary, checkpoint_path))
    return bundles
