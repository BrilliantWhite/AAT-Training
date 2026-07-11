from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_cnn_script():
    path = PROJECT_ROOT / "scripts" / "train_cnn_baselines.py"
    spec = importlib.util.spec_from_file_location("train_cnn_baselines_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainingScriptTests(unittest.TestCase):
    def test_cnn_resume_flag_is_explicit_and_defaults_off(self) -> None:
        module = load_cnn_script()
        required = [
            "--config", "config.yaml", "--inputs-dir", "inputs", "--folds", "folds.csv",
            "--dataset-manifest", "dataset.json", "--fold-manifest", "fold.json", "--experiment-id", "EXP-TEST",
        ]
        self.assertFalse(module.build_parser().parse_args(required).resume)
        self.assertTrue(module.build_parser().parse_args([*required, "--resume"]).resume)


if __name__ == "__main__":
    unittest.main()
