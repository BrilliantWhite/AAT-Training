from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def load_experiments_module():
    try:
        from aat_training import experiments
    except (ModuleNotFoundError, ImportError) as exc:
        raise AssertionError("aat_training.experiments has not been implemented") from exc
    return experiments


class TrainingExperimentRegistryTests(unittest.TestCase):
    def test_completion_accepts_artifacts_when_output_root_is_relative(self) -> None:
        experiments = load_experiments_module()
        provenance = {
            "dataset_version": "snapshot_v0",
            "dataset_manifest_sha256": "a" * 64,
            "fold_version": "folds_v1",
            "fold_manifest_sha256": "b" * 64,
            "seed": 1,
            "code_revision": "abcdef1",
        }
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp_dir:
            relative_root = Path(tmp_dir).relative_to(PROJECT_ROOT)
            previous = Path.cwd()
            try:
                os.chdir(PROJECT_ROOT)
                run = experiments.create_experiment(relative_root, "EXP-RELATIVE-001", {}, provenance)
                artifact = run.path / "predictions.csv"
                artifact.write_text("lane_id\nL1\n", encoding="utf-8")
                experiments.complete_experiment(run, [artifact], {"rows": 1})
            finally:
                os.chdir(previous)
            manifest = json.loads(run.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertIn("predictions.csv", manifest["artifacts"])

    def test_experiment_creation_records_resolved_config_and_provenance(self) -> None:
        experiments = load_experiments_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            provenance = {
                "dataset_version": "snapshot_v0",
                "dataset_manifest_sha256": "a" * 64,
                "fold_version": "folds_v1",
                "fold_manifest_sha256": "b" * 64,
                "seed": 20260710,
                "code_revision": "abcdef1",
            }

            run = experiments.create_experiment(root, "EXP-UNIT-001", {"model": "logistic", "C": 1.0}, provenance)

            self.assertEqual(run.experiment_id, "EXP-UNIT-001")
            self.assertEqual(run.path, root / "EXP-UNIT-001")
            self.assertTrue(run.resolved_config_path.is_file())
            manifest = json.loads(run.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["experiment_id"], "EXP-UNIT-001")
            self.assertEqual(manifest["status"], "running")
            self.assertEqual(manifest["provenance"], provenance)
            self.assertIn("resolved_config.yaml", manifest["initial_files"])

    def test_experiment_id_and_provenance_are_validated_and_outputs_never_overwrite(self) -> None:
        experiments = load_experiments_module()
        provenance = {
            "dataset_version": "snapshot_v0",
            "dataset_manifest_sha256": "a" * 64,
            "fold_version": "folds_v1",
            "fold_manifest_sha256": "b" * 64,
            "seed": 1,
            "code_revision": "abcdef1",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with self.assertRaisesRegex(ValueError, "experiment ID"):
                experiments.create_experiment(root, "bad id", {}, provenance)
            incomplete = dict(provenance)
            incomplete.pop("fold_manifest_sha256")
            with self.assertRaisesRegex(ValueError, "fold_manifest_sha256"):
                experiments.create_experiment(root, "EXP-OK", {}, incomplete)
            experiments.create_experiment(root, "EXP-OK", {}, provenance)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                experiments.create_experiment(root, "EXP-OK", {}, provenance)


if __name__ == "__main__":
    unittest.main()
