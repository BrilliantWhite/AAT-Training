from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aat_training.snapshot import build_snapshot  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit live AAT review exports and build an immutable training snapshot.")
    parser.add_argument("--source-root", type=Path, required=True, help="AAT_Project root containing the current inventory and review export")
    parser.add_argument("--output-dir", type=Path, required=True, help="New version directory; it must not already exist")
    parser.add_argument("--version", required=True, help="Version name such as snapshot_v0 or frozen_v1")
    parser.add_argument("--freeze", action="store_true", help="Apply formal-freeze unresolved-label gate")
    parser.add_argument("--label-policy", type=Path, default=PROJECT_ROOT / "configs" / "training" / "label_policy_v1.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    result = build_snapshot(args.source_root, args.output_dir, args.version, args.freeze, created_utc, args.label_policy)
    print(
        f"created {result.version} at {result.output_dir} "
        f"({result.image_count} images, {result.lane_count} lanes, {result.excluded_label_count} excluded labels, formal={result.formal})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
