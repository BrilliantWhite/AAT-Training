"""Generate dissertation-facing evidence only from complete registered runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .labels import COMMON_CLASSES
from .metrics import cluster_bootstrap_ci, evaluate_common
from .predictions import PROBABILITY_FIELDS, read_prediction_rows


@dataclass(frozen=True)
class ReportResult:
    output_dir: Path
    evidence_index_path: Path
    comparison_path: Path
    experiment_count: int


def _load_complete_experiment(path: Path):
    manifest_path = path / "run_manifest.json"
    prediction_path = path / "predictions.csv"
    if not manifest_path.is_file() or not prediction_path.is_file():
        raise ValueError(f"Experiment lacks manifest or predictions: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Experiment is not complete: {path}")
    rows = read_prediction_rows(prediction_path)
    if {row["experiment_id"] for row in rows} != {manifest["experiment_id"]}:
        raise ValueError(f"Prediction experiment ID disagrees with manifest: {path}")
    return manifest, rows


def _plot_confusion(matrix, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(np.asarray(matrix), cmap="Blues")
    axis.set_xticks(range(len(COMMON_CLASSES)), COMMON_CLASSES)
    axis.set_yticks(range(len(COMMON_CLASSES)), COMMON_CLASSES)
    axis.set_xlabel("Predicted phenotype")
    axis.set_ylabel("True phenotype")
    for row in range(len(COMMON_CLASSES)):
        for column in range(len(COMMON_CLASSES)):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_calibration(rows, output: Path, bins: int = 10) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probabilities = np.asarray([[float(row[field]) for field in PROBABILITY_FIELDS] for row in rows])
    confidence = probabilities.max(axis=1)
    correctness = np.asarray([row["predicted_label"] == row["true_label"] for row in rows], dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    centers, accuracies = [], []
    for index in range(bins):
        mask = (confidence >= edges[index]) & (confidence <= edges[index + 1] if index == bins - 1 else confidence < edges[index + 1])
        if mask.any():
            centers.append(float(confidence[mask].mean()))
            accuracies.append(float(correctness[mask].mean()))
    figure, axis = plt.subplots(figsize=(5.4, 5.4))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot(centers, accuracies, marker="o", label="Observed")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _bootstrap_macro_f1(rows) -> float:
    """Compute macro-F1 on a resample where repeated gel rows are expected."""

    from sklearn.metrics import f1_score

    return float(
        f1_score(
            [row["true_label"] for row in rows],
            [row["predicted_label"] for row in rows],
            labels=list(COMMON_CLASSES),
            average="macro",
            zero_division=0,
        )
    )


def build_evidence_report(experiment_dirs: Sequence[Path], output_dir: Path, bootstrap_iterations: int = 1000, seed: int = 20260710) -> ReportResult:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Report output already exists: {output_dir}")
    if not experiment_dirs:
        raise ValueError("No experiments supplied")
    output_dir.mkdir(parents=True)
    evidence = {"schema_version": "aat-dissertation-evidence-v1", "experiments": []}
    comparison_rows = []
    for directory in experiment_dirs:
        directory = Path(directory)
        manifest, rows = _load_complete_experiment(directory)
        metrics = evaluate_common(rows)
        bootstrap = cluster_bootstrap_ci(rows, _bootstrap_macro_f1, iterations=bootstrap_iterations, seed=seed)
        experiment_id = manifest["experiment_id"]
        confusion_path = output_dir / f"{experiment_id}_confusion_matrix.png"
        calibration_path = output_dir / f"{experiment_id}_calibration.png"
        _plot_confusion(metrics["confusion_matrix"], confusion_path)
        _plot_calibration(rows, calibration_path)
        entry = {
            "experiment_id": experiment_id,
            "provenance": manifest.get("provenance", {}),
            "metrics": metrics,
            "bootstrap": bootstrap,
            "confusion_matrix": confusion_path.name,
            "calibration_plot": calibration_path.name,
            "grad_cam_paths": [],
            "limitations": ["Grad-CAM and rare-case panels are added only after formal CNN/rare runs."],
        }
        evidence["experiments"].append(entry)
        comparison_rows.append({
            "experiment_id": experiment_id, "dataset_version": manifest.get("provenance", {}).get("dataset_version", ""),
            "fold_version": manifest.get("provenance", {}).get("fold_version", ""), "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"], "ece": metrics["ece"], "brier_score": metrics["brier_score"],
            "macro_f1_ci_lower": bootstrap["lower_95"], "macro_f1_ci_upper": bootstrap["upper_95"],
        })
    comparison_path = output_dir / "model_comparison.csv"
    with comparison_path.open("x", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    index_path = output_dir / "evidence_index.json"
    index_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ReportResult(output_dir, index_path, comparison_path, len(comparison_rows))
