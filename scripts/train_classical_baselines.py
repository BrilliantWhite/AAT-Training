from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aat_training.classical import run_classical_nested_cv  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped nested-CV AAT profile baselines.")
    parser.add_argument("--model", choices=("logistic", "rbf_svm"), required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--lanes", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "training" / "classical_v1.yaml")
    parser.add_argument("--dataset-version", default="snapshot_v0")
    parser.add_argument("--fold-version", default="folds_v1")
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    code_revision = subprocess.check_output(
        ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    provenance = {
        "dataset_version": args.dataset_version,
        "dataset_manifest_sha256": sha256(args.dataset_manifest),
        "fold_version": args.fold_version,
        "fold_manifest_sha256": sha256(args.fold_manifest),
        "seed": args.seed,
        "code_revision": code_revision,
    }
    result = run_classical_nested_cv(
        args.model,
        args.inputs_dir,
        args.lanes,
        args.folds,
        args.experiments_root,
        args.experiment_id,
        provenance,
        config["models"][args.model]["candidates"],
    )
    print(f"completed {args.experiment_id}: {result.oof_count} OOF predictions at {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
