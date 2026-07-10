from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def load_folds_module():
    try:
        from aat_training import folds
    except (ModuleNotFoundError, ImportError) as exc:
        raise AssertionError("aat_training.folds has not been implemented") from exc
    return folds


def balanced_rows(group_count: int = 12) -> list[dict[str, str]]:
    classes = ("M", "MZ", "MS", "SZ", "ZZ", "SS")
    return [
        {
            "lane_id": f"G{group:02d}_{label}",
            "parent_gel": f"G{group:02d}",
            "canonical_label": label,
            "common_eligible": "1",
        }
        for group in range(group_count)
        for label in classes
    ]


class TrainingFoldTests(unittest.TestCase):
    def test_nested_folds_are_deterministic_and_never_split_parent_gels(self) -> None:
        folds = load_folds_module()
        rows = balanced_rows()

        first = folds.build_nested_folds(rows, outer_splits=5, inner_splits=3, seed=20260710)
        second = folds.build_nested_folds(list(reversed(rows)), outer_splits=5, inner_splits=3, seed=20260710)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(rows) * 5)
        folds.validate_group_disjointness(first, outer_splits=5, inner_splits=3)
        for outer_fold in range(5):
            scenario = [row for row in first if row["outer_fold"] == outer_fold]
            test_groups = {row["parent_gel"] for row in scenario if row["outer_role"] == "test"}
            train_groups = {row["parent_gel"] for row in scenario if row["outer_role"] == "train"}
            self.assertTrue(test_groups)
            self.assertTrue(train_groups)
            self.assertTrue(test_groups.isdisjoint(train_groups))
            self.assertEqual({row["inner_fold"] for row in scenario if row["outer_role"] == "train"}, {0, 1, 2})

    def test_ineligible_rows_are_not_assigned_to_common_class_folds(self) -> None:
        folds = load_folds_module()
        rows = balanced_rows()
        rows.append({"lane_id": "rare", "parent_gel": "G99", "canonical_label": "FS", "common_eligible": "0"})

        assignments = folds.build_nested_folds(rows, outer_splits=5, inner_splits=3, seed=1)

        self.assertNotIn("rare", {row["lane_id"] for row in assignments})

    def test_outer_folds_reject_a_class_seen_in_too_few_parent_gels(self) -> None:
        folds = load_folds_module()
        rows = [row for row in balanced_rows() if row["canonical_label"] != "SS"]
        rows.extend(
            {"lane_id": f"G{group:02d}_SS", "parent_gel": f"G{group:02d}", "canonical_label": "SS", "common_eligible": "1"}
            for group in range(4)
        )

        with self.assertRaisesRegex(folds.FoldFeasibilityError, "SS.*4 parent gels.*5"):
            folds.build_nested_folds(rows, outer_splits=5, inner_splits=3, seed=1)

    def test_uneven_parent_gel_sizes_still_produce_balanced_outer_folds(self) -> None:
        folds = load_folds_module()
        classes = ("M", "MZ", "MS", "SZ", "ZZ", "SS")
        group_repetitions = (12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1)
        rows = [
            {
                "lane_id": f"G{group:02d}_{label}_{repeat}",
                "parent_gel": f"G{group:02d}",
                "canonical_label": label,
                "common_eligible": "1",
            }
            for group, repetitions in enumerate(group_repetitions)
            for label in classes
            for repeat in range(repetitions)
        ]

        assignments = folds.build_nested_folds(rows, outer_splits=5, inner_splits=3, seed=1)
        test_lane_counts = []
        test_group_counts = []
        for outer_fold in range(5):
            test_rows = [row for row in assignments if row["outer_fold"] == outer_fold and row["outer_role"] == "test"]
            test_lane_counts.append(len(test_rows))
            test_group_counts.append(len({row["parent_gel"] for row in test_rows}))

        self.assertLessEqual(max(test_group_counts) - min(test_group_counts), 1)
        self.assertLessEqual(max(test_lane_counts) - min(test_lane_counts), max(group_repetitions) * len(classes))

    def test_validator_detects_tampered_group_leakage(self) -> None:
        folds = load_folds_module()
        assignments = folds.build_nested_folds(balanced_rows(), outer_splits=5, inner_splits=3, seed=1)
        target = next(row for row in assignments if row["outer_fold"] == 0 and row["outer_role"] == "test")
        target["outer_role"] = "train"
        target["inner_fold"] = 0

        with self.assertRaisesRegex(ValueError, "parent-gel leakage"):
            folds.validate_group_disjointness(assignments, outer_splits=5, inner_splits=3)

    def test_fold_artifacts_are_immutable_and_hash_registered(self) -> None:
        folds = load_folds_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            lanes_path = root / "lanes.csv"
            output_dir = root / "folds_v1"
            rows = balanced_rows()
            with lanes_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            result = folds.build_fold_artifacts(lanes_path, output_dir, "folds_v1", 5, 3, 20260710)

            self.assertEqual(result.eligible_lane_count, len(rows))
            self.assertEqual(result.parent_gel_count, 12)
            self.assertTrue((output_dir / "folds.csv").is_file())
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["outer_splits"], 5)
            self.assertEqual(summary["inner_splits"], 3)
            self.assertEqual(manifest["version"], "folds_v1")
            self.assertIn("sha256", manifest["files"]["folds.csv"])
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                folds.build_fold_artifacts(lanes_path, output_dir, "folds_v1", 5, 3, 20260710)


if __name__ == "__main__":
    unittest.main()
