from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from aat_training.duplicates import audit_crop_duplicates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit cross-gel exact and perceptual duplicate lane crops.")
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-hamming-threshold", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    with (args.inputs_dir / "inputs.csv").open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    report = audit_crop_duplicates(rows, args.inputs_dir, args.near_hamming_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"audited {report['lane_count']} lanes: {len(report['exact_cross_gel_pairs'])} exact, {len(report['near_cross_gel_pairs'])} near cross-gel pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
