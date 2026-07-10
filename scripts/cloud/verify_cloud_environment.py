from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aat_training.folds import validate_group_disjointness  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed AutoDL environment and artifact verifier.")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--minimum-vram-gb", type=float, default=23.0)
    parser.add_argument("--allow-development", action="store_true", help="Permit snapshot_v0 for smoke tests only")
    args = parser.parse_args()

    import csv
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA is not available")
    properties = torch.cuda.get_device_properties(0)
    vram_gb = properties.total_memory / 1024**3
    if vram_gb < args.minimum_vram_gb:
        raise SystemExit(f"FAIL: only {vram_gb:.1f} GiB VRAM")
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    fold_manifest = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
    if not dataset_manifest.get("formal") and not args.allow_development:
        raise SystemExit("FAIL: formal training requires a formal frozen dataset")
    for relative, metadata in dataset_manifest.get("files", {}).items():
        artifact = args.dataset_manifest.parent / relative
        if not artifact.is_file() or sha256(artifact) != metadata["sha256"]:
            raise SystemExit(f"FAIL: dataset artifact hash mismatch: {relative}")
    for relative, metadata in fold_manifest.get("files", {}).items():
        artifact = args.fold_manifest.parent / relative
        if not artifact.is_file() or sha256(artifact) != metadata["sha256"]:
            raise SystemExit(f"FAIL: fold artifact hash mismatch: {relative}")
    with args.folds.open(newline="", encoding="utf-8-sig") as source:
        folds = list(csv.DictReader(source))
    validate_group_disjointness(folds, outer_splits=5, inner_splits=3)
    payload = {
        "status": "pass",
        "gpu": properties.name,
        "vram_gib": round(vram_gb, 2),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dataset_manifest_sha256": sha256(args.dataset_manifest),
        "fold_manifest_sha256": sha256(args.fold_manifest),
        "fold_rows": len(folds),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
