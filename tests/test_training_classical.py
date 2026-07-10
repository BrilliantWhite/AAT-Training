from __future__ import annotations

import csv
import json
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
        from aat_training import classical, folds, predictions
    except (ModuleNotFoundError, ImportError) as exc:
        raise AssertionError("classical training module has not been implemented") from exc
    return classical, folds, predictions


class TrainingClassicalTests(unittest.TestCase):
    def test_profile_feature_vector_contains_curve_peaks_and_summary(self) -> None:
        classical, _, _ = load_modules()
        profile = np.zeros(128, dtype=np.float32)
        profile[20] = 0.8
        profile[80] = 1.0

        features = classical.profile_feature_vector(profile, peak_count=6)

        self.assertEqual(features.shape, (145,))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(features[128], 20 / 127)
        self.assertAlmostEqual(features[129], 80 / 127)
        self.assertAlmostEqual(features[134], 0.8)
        self.assertAlmostEqual(features[135], 1.0)
        self.assertAlmostEqual(features[-5], 2 / 6)

    def test_scaler_is_fitted_only_on_training_features(self) -> None:
        classical, _, _ = load_modules()
        train_features = np.array([[0.0, 0.0], [2.0, 2.0]])
        train_labels = np.array(["M", "MZ"])

        pipeline = classical.fit_profile_pipeline("logistic", train_features, train_labels, {"C": 1.0}, seed=1)

        np.testing.assert_allclose(pipeline.named_steps["scale"].mean_, np.array([1.0, 1.0]))

    def test_model_factory_supports_only_registered_classical_models(self) -> None:
        classical, _, _ = load_modules()
        self.assertEqual(classical.make_classifier("logistic", {"C": 1.0}, 1).__class__.__name__, "LogisticRegression")
        self.assertEqual(classical.make_classifier("rbf_svm", {"C": 1.0, "gamma": "scale"}, 1).__class__.__name__, "SVC")
        with self.assertRaisesRegex(ValueError, "Unsupported classical model"):
            classical.make_classifier("random_forest", {}, 1)

    def test_nested_logistic_smoke_writes_complete_oof_artifacts(self) -> None:
        classical, folds, predictions = load_modules()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            inputs_dir = root / "inputs"
            inputs_dir.mkdir()
            lanes_path = root / "lanes.csv"
            folds_path = root / "folds.csv"
            experiments_root = root / "experiments"
            classes = ("M", "MZ", "MS", "SZ", "ZZ", "SS")
            lane_rows = []
            profiles = []
            for group in range(10):
                for class_index, label in enumerate(classes):
                    lane_id = f"G{group:02d}_{label}"
                    profile = np.full(128, class_index / 10, dtype=np.float32)
                    profile[10 + class_index * 10] = 1.0
                    profiles.append(profile)
                    lane_rows.append(
                        {
                            "lane_id": lane_id,
                            "parent_gel": f"G{group:02d}",
                            "canonical_label": label,
                            "common_eligible": "1",
                        }
                    )
            self._write_csv(lanes_path, lane_rows)
            self._write_csv(inputs_dir / "inputs.csv", [{"lane_id": row["lane_id"], "profile_index": index} for index, row in enumerate(lane_rows)])
            np.savez_compressed(inputs_dir / "profiles.npz", profiles=np.stack(profiles), lane_ids=np.array([row["lane_id"] for row in lane_rows]))
            assignments = folds.build_nested_folds(lane_rows, outer_splits=5, inner_splits=3, seed=9)
            self._write_csv(folds_path, assignments)
            provenance = {
                "dataset_version": "snapshot_v0",
                "dataset_manifest_sha256": "a" * 64,
                "fold_version": "folds_v1",
                "fold_manifest_sha256": "b" * 64,
                "seed": 9,
                "code_revision": "abcdef1",
            }

            result = classical.run_classical_nested_cv(
                model_name="logistic",
                inputs_dir=inputs_dir,
                lanes_path=lanes_path,
                folds_path=folds_path,
                experiments_root=experiments_root,
                experiment_id="EXP-LOGISTIC-SMOKE",
                provenance=provenance,
                candidates=[{"C": 1.0}],
            )

            self.assertEqual(result.oof_count, len(lane_rows))
            self.assertEqual(result.outer_fold_count, 5)
            prediction_rows = predictions.read_prediction_rows(result.predictions_path)
            self.assertEqual({row["lane_id"] for row in prediction_rows}, {row["lane_id"] for row in lane_rows})
            self.assertEqual(len(prediction_rows), len(lane_rows))
            metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
            self.assertIn("macro_f1", metrics)
            self.assertEqual(len(metrics["outer_folds"]), 5)
            manifest = json.loads((result.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertIn("predictions.csv", manifest["artifacts"])

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
