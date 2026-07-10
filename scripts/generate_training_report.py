from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from aat_training.reporting import build_evidence_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AAT dissertation evidence from complete registered experiments.")
    parser.add_argument("--experiment", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    result = build_evidence_report(args.experiment, args.output_dir, args.bootstrap_iterations, args.seed)
    print(f"created evidence for {result.experiment_count} experiments at {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
