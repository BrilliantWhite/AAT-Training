"""Deterministic stratified parent-gel grouped nested fold assignments."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .labels import COMMON_CLASSES


FOLD_SCHEMA_VERSION = "aat-nested-folds-v1"
ASSIGNMENT_FIELDS = ("lane_id", "parent_gel", "canonical_label", "outer_fold", "outer_role", "inner_fold")


class FoldFeasibilityError(ValueError):
    """Raised when requested grouped stratification is scientifically impossible."""


@dataclass(frozen=True)
class FoldBuildResult:
    output_dir: Path
    version: str
    eligible_lane_count: int
    parent_gel_count: int
    outer_splits: int
    inner_splits: int


def _eligible_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    eligible = [
        dict(row)
        for row in rows
        if str(row.get("common_eligible", "")).strip().lower() in {"1", "true", "yes"}
        and row.get("canonical_label", "") in COMMON_CLASSES
    ]
    eligible.sort(key=lambda row: row["lane_id"])
    lane_ids = [row["lane_id"] for row in eligible]
    if len(lane_ids) != len(set(lane_ids)):
        raise ValueError("Duplicate lane_id in fold input")
    for row in eligible:
        if not row.get("parent_gel"):
            raise ValueError(f"Missing parent_gel for {row['lane_id']}")
    return eligible


def _check_feasibility(rows: list[dict[str, str]], n_splits: int, stage: str) -> None:
    if n_splits < 2:
        raise FoldFeasibilityError(f"{stage} split count must be at least 2")
    groups = {row["parent_gel"] for row in rows}
    if len(groups) < n_splits:
        raise FoldFeasibilityError(f"{stage} has {len(groups)} parent gels but requires {n_splits}")
    class_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        class_groups[row["canonical_label"]].add(row["parent_gel"])
    missing = [label for label in COMMON_CLASSES if not class_groups[label]]
    if missing:
        raise FoldFeasibilityError(f"{stage} is missing common classes: {', '.join(missing)}")
    for label in COMMON_CLASSES:
        count = len(class_groups[label])
        if count < n_splits:
            raise FoldFeasibilityError(f"{stage} class {label} is present in {count} parent gels but requires {n_splits}")


def _assign_groups(rows: list[dict[str, str]], n_splits: int, seed: int) -> dict[str, int]:
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_totals: Counter[str] = Counter()
    group_sizes: Counter[str] = Counter()
    for row in rows:
        group = row["parent_gel"]
        label = row["canonical_label"]
        group_counts[group][label] += 1
        group_sizes[group] += 1
        class_totals[label] += 1

    randomizer = random.Random(seed)
    tie_breaks = {group: randomizer.random() for group in sorted(group_counts)}

    def priority(group: str) -> tuple[float, int, float, str]:
        rarity = sum(count / class_totals[label] for label, count in group_counts[group].items())
        return (-rarity, -group_sizes[group], tie_breaks[group], group)

    ordered_groups = sorted(group_counts, key=priority)
    fold_counts = [Counter() for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]
    fold_group_counts = [0 for _ in range(n_splits)]
    target_counts = {label: class_totals[label] / n_splits for label in COMMON_CLASSES}
    target_size = sum(group_sizes.values()) / n_splits
    assignment: dict[str, int] = {}

    for index, group in enumerate(ordered_groups):
        if index < n_splits:
            chosen = index
        else:
            scores: list[tuple[float, int, int, int]] = []
            target_group_count = len(ordered_groups) / n_splits
            for candidate_fold in range(n_splits):
                label_score = 0.0
                for label in COMMON_CLASSES:
                    for fold in range(n_splits):
                        proposed = fold_counts[fold][label]
                        if fold == candidate_fold:
                            proposed += group_counts[group][label]
                        label_score += ((proposed - target_counts[label]) / (target_counts[label] + 1.0)) ** 2
                size_score = sum(
                    (
                        (
                            fold_sizes[fold]
                            + (group_sizes[group] if fold == candidate_fold else 0)
                            - target_size
                        )
                        / (target_size + 1.0)
                    )
                    ** 2
                    for fold in range(n_splits)
                )
                group_score = sum(
                    (
                        (fold_group_counts[fold] + (1 if fold == candidate_fold else 0) - target_group_count)
                        / (target_group_count + 1.0)
                    )
                    ** 2
                    for fold in range(n_splits)
                )
                scores.append(
                    (
                        label_score + 0.15 * size_score + 0.05 * group_score,
                        fold_sizes[candidate_fold],
                        fold_group_counts[candidate_fold],
                        candidate_fold,
                    )
                )
            chosen = min(scores)[3]
        assignment[group] = chosen
        fold_counts[chosen].update(group_counts[group])
        fold_sizes[chosen] += group_sizes[group]
        fold_group_counts[chosen] += 1
    return assignment


def build_nested_folds(
    rows: list[dict[str, str]],
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 20260710,
) -> list[dict[str, object]]:
    """Create long-form fixed outer-test and inner-validation assignments."""

    eligible = _eligible_rows(rows)
    _check_feasibility(eligible, outer_splits, "outer")
    outer_group_fold = _assign_groups(eligible, outer_splits, seed)
    assignments: list[dict[str, object]] = []

    for outer_fold in range(outer_splits):
        outer_train = [row for row in eligible if outer_group_fold[row["parent_gel"]] != outer_fold]
        _check_feasibility(outer_train, inner_splits, f"outer {outer_fold} inner")
        inner_group_fold = _assign_groups(outer_train, inner_splits, seed + 1000 + outer_fold)
        for row in eligible:
            is_test = outer_group_fold[row["parent_gel"]] == outer_fold
            assignments.append(
                {
                    "lane_id": row["lane_id"],
                    "parent_gel": row["parent_gel"],
                    "canonical_label": row["canonical_label"],
                    "outer_fold": outer_fold,
                    "outer_role": "test" if is_test else "train",
                    "inner_fold": "" if is_test else inner_group_fold[row["parent_gel"]],
                }
            )
    assignments.sort(key=lambda row: (int(row["outer_fold"]), str(row["lane_id"])))
    validate_group_disjointness(assignments, outer_splits, inner_splits)
    return assignments


def validate_group_disjointness(assignments: list[dict[str, object]], outer_splits: int, inner_splits: int) -> None:
    """Raise if a parent gel crosses any outer or inner partition."""

    for outer_fold in range(outer_splits):
        scenario = [row for row in assignments if int(row["outer_fold"]) == outer_fold]
        if not scenario:
            raise ValueError(f"Missing outer fold scenario {outer_fold}")
        roles: dict[str, set[str]] = defaultdict(set)
        inner_by_group: dict[str, set[int]] = defaultdict(set)
        observed_inner: set[int] = set()
        for row in scenario:
            group = str(row["parent_gel"])
            role = str(row["outer_role"])
            roles[group].add(role)
            if role == "train":
                try:
                    inner_fold = int(row["inner_fold"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Missing inner fold for train group {group}") from exc
                inner_by_group[group].add(inner_fold)
                observed_inner.add(inner_fold)
            elif role == "test":
                if row.get("inner_fold") not in {"", None}:
                    raise ValueError(f"Outer-test row has inner assignment for {group}")
            else:
                raise ValueError(f"Invalid outer role {role!r}")
        if any(len(group_roles) != 1 for group_roles in roles.values()):
            raise ValueError(f"parent-gel leakage across outer roles in fold {outer_fold}")
        if any(len(group_folds) != 1 for group_folds in inner_by_group.values()):
            raise ValueError(f"parent-gel leakage across inner folds in outer fold {outer_fold}")
        if observed_inner != set(range(inner_splits)):
            raise ValueError(f"Outer fold {outer_fold} does not contain all {inner_splits} inner folds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fold_artifacts(
    lanes_path: Path,
    output_dir: Path,
    version: str = "folds_v1",
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 20260710,
) -> FoldBuildResult:
    """Write immutable fold CSV, summary, and content hashes."""

    lanes_path = Path(lanes_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Fold output already exists: {output_dir}")
    with lanes_path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    eligible = _eligible_rows(rows)
    assignments = build_nested_folds(rows, outer_splits, inner_splits, seed)
    temp_dir = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir()
        with (temp_dir / "folds.csv").open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=ASSIGNMENT_FIELDS)
            writer.writeheader()
            writer.writerows(assignments)
        class_counts = Counter(row["canonical_label"] for row in eligible)
        class_groups: dict[str, set[str]] = defaultdict(set)
        for row in eligible:
            class_groups[row["canonical_label"]].add(row["parent_gel"])
        outer_summary = []
        for fold in range(outer_splits):
            test_rows = [row for row in assignments if row["outer_fold"] == fold and row["outer_role"] == "test"]
            outer_summary.append(
                {
                    "outer_fold": fold,
                    "test_lane_count": len(test_rows),
                    "test_parent_gel_count": len({row["parent_gel"] for row in test_rows}),
                    "test_class_counts": dict(sorted(Counter(row["canonical_label"] for row in test_rows).items())),
                }
            )
        summary = {
            "schema_version": FOLD_SCHEMA_VERSION,
            "version": version,
            "outer_splits": outer_splits,
            "inner_splits": inner_splits,
            "seed": seed,
            "eligible_lane_count": len(eligible),
            "parent_gel_count": len({row["parent_gel"] for row in eligible}),
            "class_lane_counts": dict(sorted(class_counts.items())),
            "class_parent_gel_counts": {label: len(class_groups[label]) for label in sorted(class_groups)},
            "outer_folds": outer_summary,
            "leakage_validation": "pass",
        }
        (temp_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(temp_dir.iterdir())
            if path.is_file()
        }
        manifest = {
            "schema_version": FOLD_SCHEMA_VERSION,
            "version": version,
            "immutable": True,
            "source_lanes_path": str(lanes_path),
            "source_lanes_sha256": _sha256(lanes_path),
            "files": files,
        }
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return FoldBuildResult(output_dir, version, len(eligible), len({row["parent_gel"] for row in eligible}), outer_splits, inner_splits)
