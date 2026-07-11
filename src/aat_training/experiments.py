"""Immutable experiment-directory creation and completion manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
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


def fail_experiment(run_dir: Path, error: BaseException) -> None:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "running":
        raise ValueError(f"Experiment is not running: {manifest.get('status')}")
    manifest["status"] = "failed"
    manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes

        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _acquire_experiment_lock(run_dir: Path) -> None:
    lock_path = Path(run_dir) / "experiment.lock"
    if lock_path.exists():
        try:
            existing_pid = int(json.loads(lock_path.read_text(encoding="utf-8"))["pid"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Experiment is locked by an unreadable lock: {lock_path}") from exc
        if _pid_is_alive(existing_pid):
            raise RuntimeError(f"Experiment is locked by live PID {existing_pid}")
        lock_path.unlink()
    with lock_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps({"pid": os.getpid()}, sort_keys=True) + "\n")


def release_experiment_lock(run: ExperimentRun) -> None:
    lock_path = run.path / "experiment.lock"
    if not lock_path.exists():
        return
    try:
        owner = int(json.loads(lock_path.read_text(encoding="utf-8"))["pid"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return
    if owner == os.getpid():
        lock_path.unlink()


def open_experiment_for_resume(
    output_root: Path,
    experiment_id: str,
    resolved_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> ExperimentRun:
    """Reopen an interrupted experiment only when config/provenance are identical."""

    output_root = Path(output_root).resolve()
    run_dir = output_root / experiment_id
    config_path = run_dir / "resolved_config.yaml"
    manifest_path = run_dir / "run_manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment cannot resume without config and manifest: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_status = manifest.get("status")
    if prior_status == "complete":
        raise ValueError(f"Experiment {experiment_id} is already complete")
    if prior_status not in {"running", "failed"}:
        raise ValueError(f"Experiment {experiment_id} has unsupported resume status: {prior_status}")
    if manifest.get("provenance") != dict(provenance):
        raise ValueError(f"Experiment provenance mismatch for resume: {experiment_id}")
    registered_hash = manifest.get("initial_files", {}).get("resolved_config.yaml", {}).get("sha256")
    parsed_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if parsed_config != dict(resolved_config) or _sha256(config_path) != registered_hash:
        raise ValueError(f"Experiment config mismatch for resume: {experiment_id}")
    _acquire_experiment_lock(run_dir)
    history = manifest.setdefault("resume_history", [])
    history.append(
        {
            "resumed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "prior_status": prior_status,
            "prior_failure": manifest.get("failure"),
        }
    )
    manifest["status"] = "running"
    manifest.pop("failure", None)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ExperimentRun(experiment_id, run_dir, config_path, manifest_path)
