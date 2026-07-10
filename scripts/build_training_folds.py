from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aat_training.folds import build_fold_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed parent-gel grouped 5x3 nested folds for common AAT classes.")
    parser.add_argument("--lanes", type=Path, required=True, help="Versioned snapshot lanes.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="folds_v1")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_fold_artifacts(args.lanes, args.output_dir, args.version, args.outer_splits, args.inner_splits, args.seed)
    print(
        f"created {result.version} at {result.output_dir} "
        f"({result.eligible_lane_count} lanes, {result.parent_gel_count} parent gels, {result.outer_splits}x{result.inner_splits})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
