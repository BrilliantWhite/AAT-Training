from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TrainingReportingTests(unittest.TestCase):
    def test_report_requires_complete_registered_experiment_and_writes_evidence(self) -> None:
        from aat_training.labels import COMMON_CLASSES
        from aat_training.predictions import write_prediction_rows
        from aat_training.reporting import build_evidence_report

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            experiment = root / "EXP-REPORT-001"
            experiment.mkdir()
            rows = []
            for index, label in enumerate(COMMON_CLASSES):
                row = {
                    "experiment_id": "EXP-REPORT-001", "dataset_version": "frozen_v1", "fold_version": "folds_v1",
                    "config_id": "cfg", "seed": 1, "code_revision": "abc", "lane_id": f"L{index}",
                    "parent_gel": f"G{index}", "outer_fold": index % 5, "true_label": label, "predicted_label": label,
                }
                row.update({f"prob_{candidate}": 1.0 if candidate == label else 0.0 for candidate in COMMON_CLASSES})
                rows.append(row)
            write_prediction_rows(experiment / "predictions.csv", rows)
            (experiment / "run_manifest.json").write_text(json.dumps({"status": "complete", "experiment_id": "EXP-REPORT-001", "provenance": {"dataset_version": "frozen_v1", "fold_version": "folds_v1"}}), encoding="utf-8")
            output = root / "report"
            result = build_evidence_report([experiment], output, bootstrap_iterations=20, seed=1)
            self.assertTrue((output / "model_comparison.csv").is_file())
            self.assertTrue((output / "EXP-REPORT-001_confusion_matrix.png").is_file())
            self.assertTrue((output / "EXP-REPORT-001_calibration.png").is_file())
            index = json.loads((output / "evidence_index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["experiments"][0]["bootstrap"]["iterations"], 20)
            self.assertEqual(result.experiment_count, 1)


if __name__ == "__main__":
    unittest.main()
