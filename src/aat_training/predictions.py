"""Shared prediction-row provenance and probability contract."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from .labels import COMMON_CLASSES


PROVENANCE_FIELDS = (
    "experiment_id",
    "dataset_version",
    "fold_version",
    "config_id",
    "seed",
    "code_revision",
    "lane_id",
    "parent_gel",
    "outer_fold",
    "true_label",
    "predicted_label",
)
PROBABILITY_FIELDS = tuple(f"prob_{label}" for label in COMMON_CLASSES)
PREDICTION_FIELDS = (*PROVENANCE_FIELDS, *PROBABILITY_FIELDS)


def validate_prediction_rows(rows: Iterable[Mapping[str, object]], tolerance: float = 1e-6) -> None:
    """Reject untraceable, invalid, or duplicate common-class predictions."""

    seen: set[tuple[str, str, int]] = set()
    row_count = 0
    for row_index, row in enumerate(rows):
        row_count += 1
        for field in (*PROVENANCE_FIELDS, *PROBABILITY_FIELDS):
            if field not in row or row[field] is None or str(row[field]).strip() == "":
                raise ValueError(f"Prediction row {row_index} is missing {field}")
        true_label = str(row["true_label"])
        predicted_label = str(row["predicted_label"])
        if true_label not in COMMON_CLASSES:
            raise ValueError(f"Prediction row {row_index} has unsupported true_label {true_label!r}")
        if predicted_label not in COMMON_CLASSES:
            raise ValueError(f"Prediction row {row_index} has unsupported predicted_label {predicted_label!r}")
        probabilities = [float(row[field]) for field in PROBABILITY_FIELDS]
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError(f"Prediction row {row_index} has probability outside [0, 1]")
        if abs(sum(probabilities) - 1.0) > tolerance:
            raise ValueError(f"Prediction row {row_index} probabilities must sum to 1")
        key = (str(row["experiment_id"]), str(row["lane_id"]), int(row["outer_fold"]))
        if key in seen:
            raise ValueError(f"Duplicate prediction key: {key}")
        seen.add(key)
    if row_count == 0:
        raise ValueError("Prediction rows are empty")


def write_prediction_rows(path: Path, rows: list[Mapping[str, object]]) -> None:
    """Validate and atomically establish a new common-prediction CSV."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Prediction output already exists: {path}")
    validate_prediction_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    validate_prediction_rows(rows)
    return rows
