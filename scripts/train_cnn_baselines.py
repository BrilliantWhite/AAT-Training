from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from aat_training.cnn import run_cnn_nested_cv  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed grouped nested-CV AAT CNN baselines.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--dataset-version", default="frozen_v1")
    parser.add_argument("--fold-version", default="folds_v1")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pretrained", action="store_true", help="Smoke tests only; formal runs must use pretrained weights.")
    args = parser.parse_args()
    if args.dataset_version.startswith("frozen") and args.no_pretrained:
        parser.error("Formal frozen-dataset runs must use ImageNet pretrained weights")
    revision = subprocess.check_output(["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    provenance = {
        "dataset_version": args.dataset_version, "dataset_manifest_sha256": digest(args.dataset_manifest),
        "fold_version": args.fold_version, "fold_manifest_sha256": digest(args.fold_manifest),
        "seed": args.seed, "code_revision": revision,
    }
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.dataset_version.startswith("frozen") and not bool(config.get("pretrained")):
        parser.error("Formal frozen-dataset config must declare pretrained: true")
    output = run_cnn_nested_cv(args.inputs_dir, args.folds, args.experiments_root, args.experiment_id, provenance, config, args.device, not args.no_pretrained)
    print(f"completed {args.experiment_id}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
