"""Grouped nested-CV Logistic Regression and RBF-SVM profile baselines."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .experiments import complete_experiment, create_experiment
from .labels import COMMON_CLASSES
from .metrics import evaluate_common
from .predictions import write_prediction_rows


@dataclass(frozen=True)
class ClassicalRunResult:
    run_dir: Path
    predictions_path: Path
    metrics_path: Path
    oof_count: int
    outer_fold_count: int


def profile_feature_vector(profile: np.ndarray, peak_count: int = 6) -> np.ndarray:
    """Concatenate normalized curve, peak locations/amplitudes, and summaries."""

    profile = np.asarray(profile, dtype=np.float32)
    if profile.ndim != 1 or profile.size < 3 or peak_count <= 0 or not np.isfinite(profile).all():
        raise ValueError("Profile must be a finite 1D curve with at least three values")
    peaks, properties = find_peaks(profile, prominence=0.05, distance=3)
    prominences = properties.get("prominences", np.zeros(len(peaks)))
    selected = np.argsort(-prominences, kind="stable")[:peak_count]
    selected_peaks = np.sort(peaks[selected])
    positions = np.zeros(peak_count, dtype=np.float32)
    amplitudes = np.zeros(peak_count, dtype=np.float32)
    count = len(selected_peaks)
    if count:
        positions[:count] = selected_peaks / (profile.size - 1)
        amplitudes[:count] = profile[selected_peaks]
    summary = np.array(
        [
            min(len(peaks), peak_count) / peak_count,
            float(profile.mean()),
            float(profile.std()),
            float(profile.max()),
            float(np.trapezoid(profile) / (profile.size - 1)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([profile, positions, amplitudes, summary]).astype(np.float32)


def make_classifier(model_name: str, params: Mapping[str, Any], seed: int):
    if model_name == "logistic":
        return LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        )
    if model_name == "rbf_svm":
        gamma: str | float = params.get("gamma", "scale")
        if gamma != "scale" and gamma != "auto":
            gamma = float(gamma)
        return SVC(
            C=float(params.get("C", 1.0)),
            gamma=gamma,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=seed,
        )
    raise ValueError(f"Unsupported classical model: {model_name}")


def fit_profile_pipeline(
    model_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    params: Mapping[str, Any],
    seed: int,
) -> Pipeline:
    pipeline = Pipeline([("scale", StandardScaler()), ("model", make_classifier(model_name, params, seed))])
    pipeline.fit(np.asarray(train_features), np.asarray(train_labels))
    return pipeline


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def _load_feature_map(inputs_dir: Path) -> dict[str, np.ndarray]:
    inputs = _read_csv(inputs_dir / "inputs.csv")
    with np.load(inputs_dir / "profiles.npz") as arrays:
        profiles = arrays["profiles"].copy()
    feature_map: dict[str, np.ndarray] = {}
    for row in inputs:
        index = int(row["profile_index"])
        feature_map[row["lane_id"]] = profile_feature_vector(profiles[index])
    return feature_map


def _select_candidate(
    model_name: str,
    candidates: Sequence[Mapping[str, Any]],
    lane_ids: list[str],
    labels: dict[str, str],
    features: dict[str, np.ndarray],
    inner_folds: dict[str, int],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not candidates:
        raise ValueError("Classical candidate list is empty")
    records: list[dict[str, Any]] = []
    fold_values = sorted(set(inner_folds.values()))
    for candidate_index, candidate in enumerate(candidates):
        scores: list[float] = []
        for inner_fold in fold_values:
            train_ids = [lane_id for lane_id in lane_ids if inner_folds[lane_id] != inner_fold]
            validation_ids = [lane_id for lane_id in lane_ids if inner_folds[lane_id] == inner_fold]
            pipeline = fit_profile_pipeline(
                model_name,
                np.stack([features[lane_id] for lane_id in train_ids]),
                np.array([labels[lane_id] for lane_id in train_ids]),
                candidate,
                seed + inner_fold,
            )
            predictions = pipeline.predict(np.stack([features[lane_id] for lane_id in validation_ids]))
            scores.append(
                float(
                    f1_score(
                        [labels[lane_id] for lane_id in validation_ids],
                        predictions,
                        labels=list(COMMON_CLASSES),
                        average="macro",
                        zero_division=0,
                    )
                )
            )
        records.append(
            {
                "candidate_index": candidate_index,
                "params": dict(candidate),
                "inner_macro_f1": float(np.mean(scores)),
                "inner_fold_scores": scores,
            }
        )
    best = max(records, key=lambda record: (record["inner_macro_f1"], -record["candidate_index"]))
    return dict(best["params"]), records


def run_classical_nested_cv(
    model_name: str,
    inputs_dir: Path,
    lanes_path: Path,
    folds_path: Path,
    experiments_root: Path,
    experiment_id: str,
    provenance: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> ClassicalRunResult:
    """Select on fixed inner folds, fit outer trains, and write complete OOF evidence."""

    inputs_dir = Path(inputs_dir)
    lanes = {row["lane_id"]: row for row in _read_csv(lanes_path) if row.get("common_eligible") == "1"}
    assignments = _read_csv(folds_path)
    features = _load_feature_map(inputs_dir)
    eligible_ids = sorted(set(lanes) & set(features))
    resolved_config = {"schema_version": "aat-classical-training-v1", "model": model_name, "candidates": [dict(value) for value in candidates]}
    run = create_experiment(experiments_root, experiment_id, resolved_config, provenance)
    models_dir = run.path / "models"
    models_dir.mkdir()
    seed = int(provenance["seed"])
    outer_folds = sorted({int(row["outer_fold"]) for row in assignments})
    oof_rows: list[dict[str, object]] = []
    outer_records: list[dict[str, Any]] = []
    model_paths: list[Path] = []

    for outer_fold in outer_folds:
        scenario = [row for row in assignments if int(row["outer_fold"]) == outer_fold]
        test_ids = sorted(row["lane_id"] for row in scenario if row["outer_role"] == "test" and row["lane_id"] in lanes)
        train_rows = [row for row in scenario if row["outer_role"] == "train" and row["lane_id"] in lanes]
        train_ids = sorted(row["lane_id"] for row in train_rows)
        inner_folds = {row["lane_id"]: int(row["inner_fold"]) for row in train_rows}
        best_params, selection_records = _select_candidate(
            model_name,
            candidates,
            train_ids,
            {lane_id: lanes[lane_id]["canonical_label"] for lane_id in eligible_ids},
            features,
            inner_folds,
            seed + outer_fold * 100,
        )
        pipeline = fit_profile_pipeline(
            model_name,
            np.stack([features[lane_id] for lane_id in train_ids]),
            np.array([lanes[lane_id]["canonical_label"] for lane_id in train_ids]),
            best_params,
            seed + outer_fold,
        )
        test_features = np.stack([features[lane_id] for lane_id in test_ids])
        probabilities = pipeline.predict_proba(test_features)
        model_classes = list(pipeline.named_steps["model"].classes_)
        ordered_probabilities = np.zeros((len(test_ids), len(COMMON_CLASSES)), dtype=float)
        for class_index, label in enumerate(COMMON_CLASSES):
            ordered_probabilities[:, class_index] = probabilities[:, model_classes.index(label)]
        predictions = np.array(COMMON_CLASSES)[ordered_probabilities.argmax(axis=1)]
        fold_rows: list[dict[str, object]] = []
        for row_index, lane_id in enumerate(test_ids):
            prediction_row: dict[str, object] = {
                "experiment_id": experiment_id,
                "dataset_version": provenance["dataset_version"],
                "fold_version": provenance["fold_version"],
                "config_id": f"{experiment_id}-outer-{outer_fold}",
                "seed": seed,
                "code_revision": provenance["code_revision"],
                "lane_id": lane_id,
                "parent_gel": lanes[lane_id]["parent_gel"],
                "outer_fold": outer_fold,
                "true_label": lanes[lane_id]["canonical_label"],
                "predicted_label": str(predictions[row_index]),
            }
            prediction_row.update({f"prob_{label}": float(ordered_probabilities[row_index, index]) for index, label in enumerate(COMMON_CLASSES)})
            fold_rows.append(prediction_row)
        oof_rows.extend(fold_rows)
        model_path = models_dir / f"outer_fold_{outer_fold}.joblib"
        joblib.dump(pipeline, model_path)
        model_paths.append(model_path)
        outer_metrics = evaluate_common(fold_rows)
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "train_count": len(train_ids),
                "test_count": len(test_ids),
                "best_params": best_params,
                "candidate_results": selection_records,
                "macro_f1": outer_metrics["macro_f1"],
                "balanced_accuracy": outer_metrics["balanced_accuracy"],
            }
        )

    if len(oof_rows) != len(eligible_ids) or {str(row["lane_id"]) for row in oof_rows} != set(eligible_ids):
        raise ValueError("OOF predictions do not cover every eligible lane exactly once")
    oof_rows.sort(key=lambda row: str(row["lane_id"]))
    predictions_path = run.path / "predictions.csv"
    write_prediction_rows(predictions_path, oof_rows)
    common_metrics = evaluate_common(oof_rows)
    metrics_payload = {**common_metrics, "outer_folds": outer_records}
    metrics_path = run.path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    complete_experiment(
        run,
        [predictions_path, metrics_path, *model_paths],
        {"model": model_name, "oof_count": len(oof_rows), "macro_f1": common_metrics["macro_f1"]},
    )
    return ClassicalRunResult(run.path, predictions_path, metrics_path, len(oof_rows), len(outer_folds))
