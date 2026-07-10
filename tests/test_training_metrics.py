from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def load_modules():
    try:
        from aat_training import metrics, predictions
    except (ModuleNotFoundError, ImportError) as exc:
        raise AssertionError("training metrics/prediction contract has not been implemented") from exc
    return metrics, predictions


def prediction_row(lane: int, true_label: str, predicted_label: str, probabilities: dict[str, float]) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment_id": "EXP-001",
        "dataset_version": "snapshot_v0",
        "fold_version": "folds_v1",
        "config_id": "unit-test",
        "seed": 20260710,
        "code_revision": "abcdef1",
        "lane_id": f"L{lane:03d}",
        "parent_gel": f"G{lane // 2:02d}",
        "outer_fold": lane % 5,
        "true_label": true_label,
        "predicted_label": predicted_label,
    }
    row.update({f"prob_{label}": probabilities.get(label, 0.0) for label in ("M", "MZ", "MS", "SZ", "ZZ", "SS")})
    return row


class TrainingMetricTests(unittest.TestCase):
    def test_prediction_contract_rejects_missing_provenance_and_bad_probability_sum(self) -> None:
        _, predictions = load_modules()
        row = prediction_row(0, "M", "M", {"M": 1.0})
        missing = dict(row)
        missing.pop("code_revision")
        bad_sum = dict(row)
        bad_sum["prob_M"] = 0.8

        with self.assertRaisesRegex(ValueError, "code_revision"):
            predictions.validate_prediction_rows([missing])
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            predictions.validate_prediction_rows([bad_sum])

    def test_prediction_csv_round_trip_is_validated_and_non_overwriting(self) -> None:
        _, predictions = load_modules()
        self.assertTrue(hasattr(predictions, "write_prediction_rows"), "write_prediction_rows has not been implemented")
        self.assertTrue(hasattr(predictions, "read_prediction_rows"), "read_prediction_rows has not been implemented")
        row = prediction_row(0, "M", "M", {"M": 1.0})
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "predictions.csv"

            predictions.write_prediction_rows(path, [row])
            loaded = predictions.read_prediction_rows(path)

            predictions.validate_prediction_rows(loaded)
            self.assertEqual(loaded[0]["lane_id"], "L000")
            self.assertEqual(float(loaded[0]["prob_M"]), 1.0)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                predictions.write_prediction_rows(path, [row])

    def test_perfect_six_class_predictions_have_unit_common_metrics(self) -> None:
        metrics, _ = load_modules()
        classes = ("M", "MZ", "MS", "SZ", "ZZ", "SS")
        rows = [prediction_row(index, label, label, {label: 1.0}) for index, label in enumerate(classes)]

        result = metrics.evaluate_common(rows)

        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(result["ece"], 0.0)
        self.assertEqual(result["brier_score"], 0.0)
        self.assertEqual(result["confusion_matrix"], np.eye(6, dtype=int).tolist())
        self.assertTrue(all(values["f1"] == 1.0 for values in result["per_class"].values()))

    def test_ece_and_multiclass_brier_match_hand_calculation(self) -> None:
        metrics, _ = load_modules()
        probabilities = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=float)
        true_indices = np.array([0, 1])

        ece = metrics.expected_calibration_error(probabilities, true_indices, bins=1)
        brier = metrics.multiclass_brier_score(probabilities, true_indices)

        self.assertAlmostEqual(ece, 0.2)
        self.assertAlmostEqual(brier, 0.4)

    def test_retrieval_reports_top1_and_top3(self) -> None:
        metrics, _ = load_modules()
        rows = [
            {"true_label": "FS", "top_candidates": ["FS", "FZ", "FF"]},
            {"true_label": "MZBristol", "top_candidates": ["MZ", "MZBristol", "MZPratt"]},
            {"true_label": "MMalton", "top_candidates": ["M", "MM", "MMalton"]},
        ]

        result = metrics.evaluate_retrieval(rows)

        self.assertAlmostEqual(result["top1_accuracy"], 1 / 3)
        self.assertEqual(result["top3_accuracy"], 1.0)
        self.assertEqual(result["case_count"], 3)

    def test_allele_metrics_report_perfect_macro_auprc_and_recall(self) -> None:
        metrics, _ = load_modules()
        targets = np.array([[1, 0], [0, 1], [1, 1]], dtype=int)
        scores = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.9]], dtype=float)

        result = metrics.evaluate_alleles(targets, scores, allele_names=["M", "Z"], threshold=0.5)

        self.assertEqual(result["macro_auprc"], 1.0)
        self.assertEqual(result["macro_recall"], 1.0)
        self.assertEqual(result["per_allele"]["M"]["auprc"], 1.0)

    def test_referral_selects_highest_threshold_meeting_rare_sensitivity(self) -> None:
        metrics, _ = load_modules()
        is_rare = np.array([1, 1, 0, 0], dtype=int)
        referral_scores = np.array([0.9, 0.8, 0.7, 0.1], dtype=float)

        result = metrics.evaluate_referral(is_rare, referral_scores, target_sensitivity=0.9)

        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["threshold"], 0.8)
        self.assertEqual(result["rare_sensitivity"], 1.0)
        self.assertEqual(result["common_auto_accept_rate"], 1.0)

    def test_cluster_bootstrap_is_seeded_and_resamples_whole_gels(self) -> None:
        metrics, _ = load_modules()
        rows = [
            {"parent_gel": "G1", "correct": 1.0},
            {"parent_gel": "G1", "correct": 1.0},
            {"parent_gel": "G2", "correct": 0.0},
            {"parent_gel": "G2", "correct": 0.0},
            {"parent_gel": "G3", "correct": 1.0},
        ]
        metric_fn = lambda sampled: sum(row["correct"] for row in sampled) / len(sampled)

        first = metrics.cluster_bootstrap_ci(rows, metric_fn, iterations=1000, seed=7)
        second = metrics.cluster_bootstrap_ci(rows, metric_fn, iterations=1000, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first["iterations"], 1000)
        self.assertLessEqual(first["lower_95"], first["point_estimate"])
        self.assertGreaterEqual(first["upper_95"], first["point_estimate"])


if __name__ == "__main__":
    unittest.main()
