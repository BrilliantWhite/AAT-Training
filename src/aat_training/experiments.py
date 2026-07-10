"""Immutable experiment-directory creation and completion manifests."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]*$")
REQUIRED_PROVENANCE = (
    "dataset_version",
    "dataset_manifest_sha256",
    "fold_version",
    "fold_manifest_sha256",
    "seed",
    "code_revision",
)


@dataclass(frozen=True)
class ExperimentRun:
    experiment_id: str
    path: Path
    resolved_config_path: Path
    run_manifest_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_experiment(
    output_root: Path,
    experiment_id: str,
    resolved_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> ExperimentRun:
    """Create a unique run directory before any model is fitted."""

    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError(f"Invalid experiment ID: {experiment_id!r}")
    for field in REQUIRED_PROVENANCE:
        if field not in provenance or provenance[field] in {None, ""}:
            raise ValueError(f"Experiment provenance is missing {field}")
    output_root = Path(output_root).resolve()
    run_dir = output_root / experiment_id
    if run_dir.exists():
        raise FileExistsError(f"Experiment output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config_path = run_dir / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(dict(resolved_config), sort_keys=True), encoding="utf-8")
    initial_files = {"resolved_config.yaml": {"sha256": _sha256(config_path), "bytes": config_path.stat().st_size}}
    manifest = {
        "schema_version": "aat-experiment-v1",
        "experiment_id": experiment_id,
        "status": "running",
        "provenance": dict(provenance),
        "initial_files": initial_files,
        "artifacts": {},
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ExperimentRun(experiment_id, run_dir, config_path, manifest_path)


def complete_experiment(run: ExperimentRun, artifact_paths: list[Path], summary: Mapping[str, Any]) -> None:
    """Finalize a run once, registering every output by hash."""

    manifest = json.loads(run.run_manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "running":
        raise ValueError(f"Experiment {run.experiment_id} is not running")
    artifacts: dict[str, dict[str, Any]] = {}
    for path in artifact_paths:
        path = Path(path)
        if not path.is_file() or run.path not in path.resolve().parents:
            raise ValueError(f"Artifact is missing or outside experiment directory: {path}")
        relative = path.relative_to(run.path).as_posix()
        artifacts[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest["status"] = "complete"
    manifest["artifacts"] = dict(sorted(artifacts.items()))
    manifest["summary"] = dict(summary)
    run.run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
