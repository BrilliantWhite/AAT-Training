from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def torch_is_available() -> bool:
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


class CnnResumeBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name) / "EXP-RESUME"
        (self.run_dir / "checkpoints").mkdir(parents=True)
        self.checkpoint = self.run_dir / "checkpoints" / "outer_fold_0.pt"
        self.checkpoint.write_bytes(b"checkpoint-zero")
        self.provenance = {
            "experiment_id": "EXP-RESUME",
            "dataset_version": "frozen_v1",
            "dataset_manifest_sha256": "a" * 64,
            "fold_version": "folds_v1",
            "fold_manifest_sha256": "b" * 64,
            "code_revision": "abcdef1",
            "seed": 20260710,
        }
        self.predictions = [
            {
                "lane_id": "L0",
                "parent_gel": "G0",
                "outer_fold": 0,
                "experiment_id": "EXP-RESUME",
                "dataset_version": "frozen_v1",
                "fold_version": "folds_v1",
                "code_revision": "abcdef1",
                "seed": 20260710,
                "true_label": "M",
                "predicted_label": "M",
                "prob_M": 0.9,
                "prob_MZ": 0.02,
                "prob_MS": 0.02,
                "prob_SZ": 0.02,
                "prob_ZZ": 0.02,
                "prob_SS": 0.02,
            }
        ]
        self.summary = {
            "outer_fold": 0,
            "selected": {"params": {"learning_rate": 1e-4, "weight_decay": 1e-4, "dropout": 0.2}},
            "candidate_results": [],
            "best_epoch": 2,
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self):
        from aat_training.cnn_resume import write_completed_fold_bundle

        return write_completed_fold_bundle(
            self.run_dir,
            outer_fold=0,
            predictions=self.predictions,
            summary=self.summary,
            checkpoint_path=self.checkpoint,
            provenance=self.provenance,
            resolved_config_sha256="c" * 64,
        )

    def _load(self):
        from aat_training.cnn_resume import load_completed_fold_bundles

        return load_completed_fold_bundles(
            self.run_dir,
            expected_outer_folds={0, 1},
            expected_lane_ids_by_fold={0: {"L0"}, 1: {"L1"}},
            authoritative_records={"L0": {"parent_gel": "G0", "canonical_label": "M"}, "L1": {"parent_gel": "G1", "canonical_label": "MZ"}},
            expected_provenance=self.provenance,
            expected_config_sha256="c" * 64,
            candidate_grid=[{"learning_rate": 1e-4, "weight_decay": 1e-4, "dropout": 0.2}],
        )

    def test_completed_fold_bundle_round_trip_is_atomic(self) -> None:
        path = self._write()
        self.assertTrue(path.is_file())
        self.assertFalse(path.with_suffix(".json.tmp").exists())
        bundles = self._load()
        self.assertEqual([bundle.outer_fold for bundle in bundles], [0])
        self.assertEqual(bundles[0].predictions, self.predictions)
        self.assertEqual(bundles[0].summary, self.summary)
        self.assertEqual(bundles[0].checkpoint_path, self.checkpoint)

    def test_provenance_mismatch_fails_closed(self) -> None:
        path = self._write()
        payload = json.loads(path.read_text())
        payload["provenance"]["fold_manifest_sha256"] = "x" * 64
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            self._load()

    def test_checkpoint_hash_mismatch_fails_closed(self) -> None:
        self._write()
        self.checkpoint.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
            self._load()

    def test_incomplete_prediction_set_fails_closed(self) -> None:
        path = self._write()
        payload = json.loads(path.read_text())
        payload["predictions"] = []
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "lane coverage mismatch"):
            self._load()

    def test_probability_and_parent_gel_mismatch_fail_closed(self) -> None:
        path = self._write()
        payload = json.loads(path.read_text())
        payload["predictions"][0]["parent_gel"] = "WRONG"
        payload["predictions"][0]["prob_M"] = 1.5
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "parent gel mismatch"):
            self._load()


@unittest.skipUnless(torch_is_available(), "authoritative CNN resume integration runs in the AutoDL PyTorch environment")
class CnnResumeIntegrationTests(unittest.TestCase):
    def test_runner_restores_all_completed_folds_without_training(self) -> None:
        from aat_training.cnn import run_cnn_nested_cv
        from aat_training.cnn_resume import file_sha256, write_completed_fold_bundle
        from aat_training.experiments import create_experiment, fail_experiment

        config = {
            "backbone": "resnet18",
            "pretrained": True,
            "loss": "inverse_sqrt_weighted_ce",
            "batch_size": 32,
            "max_epochs": 60,
            "patience": 8,
            "amp": False,
            "num_workers": 0,
            "learning_rates": [1e-4, 3e-4],
            "weight_decays": [1e-4, 1e-3],
            "dropouts": [0.2, 0.4],
            "augmentations": {},
        }
        provenance = {
            "dataset_version": "frozen_v1",
            "dataset_manifest_sha256": "a" * 64,
            "fold_version": "folds_v1",
            "fold_manifest_sha256": "b" * 64,
            "seed": 20260710,
            "code_revision": "abcdef1",
        }
        selected = {"learning_rate": 1e-4, "weight_decay": 1e-4, "dropout": 0.2}
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            inputs_dir = root / "inputs"
            inputs_dir.mkdir()
            input_rows = [
                {"lane_id": f"L{fold}", "parent_gel": f"G{fold}", "canonical_label": "M", "common_eligible": "1"}
                for fold in range(5)
            ]
            with (inputs_dir / "inputs.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=input_rows[0])
                writer.writeheader()
                writer.writerows(input_rows)
            folds_path = root / "folds.csv"
            fold_rows = []
            for outer_fold in range(5):
                for row in input_rows:
                    fold_rows.append(
                        {
                            "lane_id": row["lane_id"],
                            "parent_gel": row["parent_gel"],
                            "outer_fold": outer_fold,
                            "outer_role": "test" if row["lane_id"] == f"L{outer_fold}" else "train",
                            "inner_fold": int(row["lane_id"][1:]) % 3,
                        }
                    )
            with folds_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fold_rows[0])
                writer.writeheader()
                writer.writerows(fold_rows)
            experiments_root = root / "experiments"
            run = create_experiment(experiments_root, "EXP-ALL-RESTORED", config, provenance)
            (run.path / "checkpoints").mkdir()
            for outer_fold in range(5):
                checkpoint = run.path / "checkpoints" / f"outer_fold_{outer_fold}.pt"
                checkpoint.write_bytes(f"checkpoint-{outer_fold}".encode())
                probabilities = {f"prob_{label}": 0.0 for label in ("M", "MZ", "MS", "SZ", "ZZ", "SS")}
                probabilities["prob_M"] = 1.0
                prediction = {
                    "experiment_id": "EXP-ALL-RESTORED",
                    "dataset_version": "frozen_v1",
                    "fold_version": "folds_v1",
                    "config_id": f"EXP-ALL-RESTORED-outer-{outer_fold}",
                    "seed": 20260710,
                    "code_revision": "abcdef1",
                    "lane_id": f"L{outer_fold}",
                    "parent_gel": f"G{outer_fold}",
                    "outer_fold": outer_fold,
                    "true_label": "M",
                    "predicted_label": "M",
                    **probabilities,
                }
                best = {"params": selected, "macro_f1": 1.0, "fold_scores": [1.0], "epochs": [1]}
                summary = {"outer_fold": outer_fold, "selected": best, "candidate_results": [best], "best_epoch": 1}
                write_completed_fold_bundle(
                    run.path,
                    outer_fold=outer_fold,
                    predictions=[prediction],
                    summary=summary,
                    checkpoint_path=checkpoint,
                    provenance={**provenance, "experiment_id": "EXP-ALL-RESTORED"},
                    resolved_config_sha256=file_sha256(run.resolved_config_path),
                )
            fail_experiment(run.path, RuntimeError("simulated interruption after five folds"))
            with patch("aat_training.cnn._fit_with_validation", side_effect=AssertionError("restored folds must not retrain")):
                output = run_cnn_nested_cv(
                    inputs_dir,
                    folds_path,
                    experiments_root,
                    "EXP-ALL-RESTORED",
                    provenance,
                    config,
                    "cpu",
                    pretrained=False,
                    resume=True,
                )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["summary"]["oof_count"], 5)
            self.assertEqual(len([name for name in manifest["artifacts"] if name.startswith("folds/")]), 5)

if __name__ == "__main__":
    unittest.main()
