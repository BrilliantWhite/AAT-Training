from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aat_training.metrics import evaluate_alleles, evaluate_referral, evaluate_retrieval  # noqa: E402
from aat_training.predictions import read_prediction_rows  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate registered AAT rare-assistance predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-sensitivity", type=float, default=0.9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    rows = read_prediction_rows(args.predictions)
    retrieval_rows = [{"true_label": row["true_label"], "top_candidates": json.loads(row["top_candidates"])} for row in rows]
    allele_names = json.loads(rows[0]["allele_names"])
    allele_targets = np.asarray([json.loads(row["allele_targets"]) for row in rows], dtype=int)
    allele_scores = np.asarray([json.loads(row["allele_scores"]) for row in rows], dtype=float)
    is_rare = np.asarray([int(row["is_rare"]) for row in rows], dtype=int)
    referral_scores = np.asarray([float(row["referral_score"]) for row in rows], dtype=float)
    payload = {
        "retrieval": evaluate_retrieval(retrieval_rows),
        "alleles": evaluate_alleles(allele_targets, allele_scores, allele_names),
        "referral": evaluate_referral(is_rare, referral_scores, target_sensitivity=args.target_sensitivity),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
