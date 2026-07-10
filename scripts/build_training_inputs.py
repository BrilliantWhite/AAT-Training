from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aat_training.preprocessing import build_training_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build traceable 128x384 lane crops and 1D profiles from a training snapshot.")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="Override the snapshot source root after moving data to a cloud volume")
    parser.add_argument("--limit", type=int, help="Build only the first N lanes for a smoke test")
    parser.add_argument("--target-height", type=int, default=128)
    parser.add_argument("--target-width", type=int, default=384)
    parser.add_argument("--profile-length", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_training_inputs(
        args.snapshot_dir,
        args.output_dir,
        target_height=args.target_height,
        target_width=args.target_width,
        profile_length=args.profile_length,
        image_root=args.image_root,
        limit=args.limit,
    )
    print(
        f"created inputs at {result.output_dir} "
        f"({result.lane_count} lanes, target={result.target_height}x{result.target_width}, profile={result.profile_length})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
