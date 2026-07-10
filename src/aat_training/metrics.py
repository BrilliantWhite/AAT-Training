"""Common, calibration, retrieval, allele, referral, and clustered-CI metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .labels import COMMON_CLASSES
from .predictions import PROBABILITY_FIELDS, validate_prediction_rows


def expected_calibration_error(probabilities: np.ndarray, true_indices: np.ndarray, bins: int = 10) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    true_indices = np.asarray(true_indices, dtype=int)
    if probabilities.ndim != 2 or true_indices.shape != (probabilities.shape[0],) or bins <= 0:
        raise ValueError("Invalid probability/target shape or bin count")
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == true_indices
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def multiclass_brier_score(probabilities: np.ndarray, true_indices: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    true_indices = np.asarray(true_indices, dtype=int)
    if probabilities.ndim != 2 or true_indices.shape != (probabilities.shape[0],):
        raise ValueError("Invalid probability/target shape")
    if np.any(true_indices < 0) or np.any(true_indices >= probabilities.shape[1]):
        raise ValueError("Target index outside probability columns")
    targets = np.zeros_like(probabilities)
    targets[np.arange(probabilities.shape[0]), true_indices] = 1.0
    return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def evaluate_common(rows: list[Mapping[str, object]], calibration_bins: int = 10) -> dict[str, Any]:
    validate_prediction_rows(rows)
    class_to_index = {label: index for index, label in enumerate(COMMON_CLASSES)}
    true_indices = np.array([class_to_index[str(row["true_label"])] for row in rows], dtype=int)
    predicted_indices = np.array([class_to_index[str(row["predicted_label"])] for row in rows], dtype=int)
    probabilities = np.array([[float(row[field]) for field in PROBABILITY_FIELDS] for row in rows], dtype=float)
    confusion = np.zeros((len(COMMON_CLASSES), len(COMMON_CLASSES)), dtype=int)
    for truth, prediction in zip(true_indices, predicted_indices):
        confusion[truth, prediction] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1_values: list[float] = []
    for index, label in enumerate(COMMON_CLASSES):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        recalls.append(recall)
        f1_values.append(f1)
    return {
        "class_order": list(COMMON_CLASSES),
        "case_count": len(rows),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "ece": expected_calibration_error(probabilities, true_indices, calibration_bins),
        "brier_score": multiclass_brier_score(probabilities, true_indices),
    }


def evaluate_retrieval(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Retrieval rows are empty")
    top1 = 0
    top3 = 0
    for index, row in enumerate(rows):
        truth = str(row.get("true_label", ""))
        candidates = [str(value) for value in row.get("top_candidates", [])]
        if not truth or not candidates:
            raise ValueError(f"Retrieval row {index} lacks truth or candidates")
        top1 += int(candidates[0] == truth)
        top3 += int(truth in candidates[:3])
    return {"case_count": len(rows), "top1_accuracy": top1 / len(rows), "top3_accuracy": top3 / len(rows)}


def _average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    positives = int(targets.sum())
    if positives == 0:
        raise ValueError("AUPRC is undefined without positive targets")
    order = np.argsort(-scores, kind="stable")
    sorted_targets = targets[order]
    cumulative_true = np.cumsum(sorted_targets)
    positive_ranks = np.flatnonzero(sorted_targets == 1)
    precisions = cumulative_true[positive_ranks] / (positive_ranks + 1)
    return float(precisions.sum() / positives)


def evaluate_alleles(
    targets: np.ndarray,
    scores: np.ndarray,
    allele_names: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if targets.shape != scores.shape or targets.ndim != 2 or targets.shape[1] != len(allele_names):
        raise ValueError("Allele target/score/name shapes do not match")
    if not np.isin(targets, [0, 1]).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("Allele targets or scores are invalid")
    per_allele: dict[str, dict[str, float | int]] = {}
    auprc_values: list[float] = []
    recall_values: list[float] = []
    for column, name in enumerate(allele_names):
        column_targets = targets[:, column]
        positives = int(column_targets.sum())
        if positives == 0:
            per_allele[str(name)] = {"positive_count": 0, "auprc": float("nan"), "recall": float("nan")}
            continue
        auprc = _average_precision(column_targets, scores[:, column])
        predicted = scores[:, column] >= threshold
        recall = float(((predicted == 1) & (column_targets == 1)).sum() / positives)
        per_allele[str(name)] = {"positive_count": positives, "auprc": auprc, "recall": recall}
        auprc_values.append(auprc)
        recall_values.append(recall)
    if not auprc_values:
        raise ValueError("No allele has positive targets")
    return {
        "threshold": threshold,
        "macro_auprc": float(np.mean(auprc_values)),
        "macro_recall": float(np.mean(recall_values)),
        "per_allele": per_allele,
    }


def _binary_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    positive_scores = scores[targets == 1]
    negative_scores = scores[targets == 0]
    if not len(positive_scores) or not len(negative_scores):
        raise ValueError("AUROC requires both rare and common cases")
    comparisons = 0.0
    for positive in positive_scores:
        comparisons += float(np.sum(positive > negative_scores))
        comparisons += 0.5 * float(np.sum(positive == negative_scores))
    return comparisons / (len(positive_scores) * len(negative_scores))


def evaluate_referral(
    is_rare: np.ndarray,
    referral_scores: np.ndarray,
    target_sensitivity: float = 0.9,
) -> dict[str, float]:
    is_rare = np.asarray(is_rare, dtype=int)
    referral_scores = np.asarray(referral_scores, dtype=float)
    if is_rare.shape != referral_scores.shape or is_rare.ndim != 1 or not np.isin(is_rare, [0, 1]).all():
        raise ValueError("Invalid referral targets/scores")
    if not 0.0 < target_sensitivity <= 1.0 or not np.isfinite(referral_scores).all():
        raise ValueError("Invalid referral target sensitivity or score")
    rare_scores = referral_scores[is_rare == 1]
    common_scores = referral_scores[is_rare == 0]
    if not len(rare_scores) or not len(common_scores):
        raise ValueError("Referral evaluation requires rare and common cases")
    candidates: list[tuple[float, float, float]] = []
    for threshold in sorted(set(float(value) for value in referral_scores), reverse=True):
        sensitivity = float(np.mean(rare_scores >= threshold))
        if sensitivity >= target_sensitivity:
            auto_accept = float(np.mean(common_scores < threshold))
            candidates.append((auto_accept, threshold, sensitivity))
    if not candidates:
        raise ValueError("No threshold meets referral sensitivity")
    auto_accept, threshold, sensitivity = max(candidates, key=lambda value: (value[0], value[1]))
    return {
        "auroc": float(_binary_auroc(is_rare, referral_scores)),
        "target_sensitivity": target_sensitivity,
        "threshold": threshold,
        "rare_sensitivity": sensitivity,
        "common_auto_accept_rate": auto_accept,
    }


def cluster_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    metric_fn: Callable[[list[Mapping[str, Any]]], float],
    iterations: int = 1000,
    seed: int = 20260710,
    group_key: str = "parent_gel",
) -> dict[str, float | int]:
    if iterations <= 0 or not rows:
        raise ValueError("Bootstrap requires rows and a positive iteration count")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row.get(group_key, ""))
        if not group:
            raise ValueError(f"Bootstrap row is missing {group_key}")
        grouped[group].append(row)
    group_names = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        selected = rng.choice(group_names, size=len(group_names), replace=True)
        sampled = [row for group in selected for row in grouped[str(group)]]
        values[iteration] = float(metric_fn(sampled))
    point = float(metric_fn(list(rows)))
    return {
        "point_estimate": point,
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "iterations": iterations,
        "seed": seed,
        "group_count": len(group_names),
    }
