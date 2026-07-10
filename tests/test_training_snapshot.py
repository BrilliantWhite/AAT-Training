from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def load_snapshot_module():
    try:
        from aat_training import snapshot
    except ModuleNotFoundError as exc:
        raise AssertionError("aat_training.snapshot has not been implemented") from exc
    return snapshot


class TrainingSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_root = self.root / "source"
        self.output_dir = self.root / "versions" / "snapshot_v0"
        self.image_path = self.source_root / "dataset" / "Originial" / "gel.png"
        self.image_path.parent.mkdir(parents=True)
        Image.new("RGB", (10, 8), color=(120, 80, 160)).save(self.image_path)
        self._write_inventory(
            [
                {
                    "image_id": "IMG_0001",
                    "source_filename": "gel.png",
                    "relative_path": "dataset/Originial/gel.png",
                    "file_ext": ".png",
                    "width": "10",
                    "height": "8",
                    "channels": "3",
                }
            ]
        )
        self._write_lanes([self._valid_lane()])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_inventory(self, rows: list[dict[str, str]]) -> None:
        self._write_csv(
            self.source_root / "dataset" / "metadata" / "image_inventory.csv",
            ["image_id", "source_filename", "relative_path", "file_ext", "width", "height", "channels"],
            rows,
        )

    def _write_lanes(self, rows: list[dict[str, str]]) -> None:
        self._write_csv(
            self.source_root / "Web" / "review_exports" / "training_lanes_export.csv",
            [
                "image_id",
                "source_filename",
                "candidate_index",
                "roi_y_start",
                "roi_y_end",
                "left_x",
                "right_x",
                "x1",
                "y1",
                "x2",
                "y2",
                "x3",
                "y3",
                "x4",
                "y4",
                "label",
                "updated_at",
            ],
            rows,
        )

    def _valid_lane(self, **changes: str) -> dict[str, str]:
        row = {
            "image_id": "IMG_0001",
            "source_filename": "gel.png",
            "candidate_index": "1",
            "roi_y_start": "1",
            "roi_y_end": "6",
            "left_x": "2",
            "right_x": "6",
            "x1": "2",
            "y1": "1",
            "x2": "6",
            "y2": "1",
            "x3": "2",
            "y3": "6",
            "x4": "6",
            "y4": "6",
            "label": "MM",
            "updated_at": "2026-07-10T12:00:00Z",
        }
        row.update(changes)
        return row

    def test_build_snapshot_copies_manifests_and_records_hashes_and_labels(self) -> None:
        snapshot = load_snapshot_module()

        try:
            result = snapshot.build_snapshot(
                source_root=self.source_root,
                output_dir=self.output_dir,
                version="snapshot_v0",
                freeze=False,
                created_utc="2026-07-10T12:30:00Z",
            )
        except snapshot.SnapshotValidationError as exc:
            self.fail(f"Valid Web-export row-major rectangle was rejected: {exc}")

        self.assertEqual(result.image_count, 1)
        self.assertEqual(result.lane_count, 1)
        self.assertEqual(result.excluded_label_count, 0)
        expected_files = {
            "lanes.csv",
            "images.csv",
            "label_summary.csv",
            "audit_report.json",
            "manifest.json",
            "source/image_inventory.csv",
            "source/training_lanes_export.csv",
        }
        self.assertEqual(
            {path.relative_to(self.output_dir).as_posix() for path in self.output_dir.rglob("*") if path.is_file()},
            expected_files,
        )

        with (self.output_dir / "lanes.csv").open(newline="", encoding="utf-8") as csv_file:
            lane = next(csv.DictReader(csv_file))
        self.assertEqual(lane["original_label"], "MM")
        self.assertEqual(lane["canonical_label"], "M")
        self.assertEqual(lane["common_eligible"], "1")
        self.assertEqual(lane["alleles"], "M")
        self.assertEqual(lane["parent_gel"], "IMG_0001")

        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], "snapshot_v0")
        self.assertTrue(manifest["immutable"])
        self.assertFalse(manifest["formal"])
        self.assertEqual(manifest["created_utc"], "2026-07-10T12:30:00Z")
        lanes_hash = hashlib.sha256((self.output_dir / "lanes.csv").read_bytes()).hexdigest()
        self.assertEqual(manifest["files"]["lanes.csv"]["sha256"], lanes_hash)
        self.assertEqual(manifest["images"]["IMG_0001"]["sha256"], hashlib.sha256(self.image_path.read_bytes()).hexdigest())

    def test_snapshot_is_deterministically_sorted_and_loads_copied_lanes(self) -> None:
        snapshot = load_snapshot_module()
        self._write_lanes([self._valid_lane(candidate_index="2", left_x="6", right_x="9", x1="6", x2="9", x3="6", x4="9"), self._valid_lane()])
        snapshot.build_snapshot(self.source_root, self.output_dir, "snapshot_v0", False, "2026-07-10T12:30:00Z")

        copied = snapshot.load_snapshot_lanes(self.output_dir)
        self._write_lanes([])

        self.assertEqual([row["candidate_index"] for row in copied], ["1", "2"])
        self.assertEqual([row["candidate_index"] for row in snapshot.load_snapshot_lanes(self.output_dir)], ["1", "2"])

    def test_snapshot_rejects_inventory_filename_mismatch(self) -> None:
        snapshot = load_snapshot_module()
        self._write_lanes([self._valid_lane(source_filename="wrong.png")])

        with self.assertRaisesRegex(snapshot.SnapshotValidationError, "image_id and source_filename"):
            snapshot.build_snapshot(self.source_root, self.output_dir, "snapshot_v0", False, "2026-07-10T12:30:00Z")

    def test_snapshot_rejects_invalid_geometry_duplicate_keys_and_empty_labels(self) -> None:
        snapshot = load_snapshot_module()
        cases = {
            "out of bounds": [self._valid_lane(right_x="12", x2="12", x3="12")],
            "duplicate lane key": [self._valid_lane(), self._valid_lane()],
            "empty label": [self._valid_lane(label="")],
        }
        for message, rows in cases.items():
            with self.subTest(message=message):
                self._write_lanes(rows)
                with self.assertRaisesRegex(snapshot.SnapshotValidationError, message):
                    snapshot.build_snapshot(self.source_root, self.output_dir, "snapshot_v0", False, "2026-07-10T12:30:00Z")

    def test_snapshot_refuses_to_overwrite_existing_version(self) -> None:
        snapshot = load_snapshot_module()
        self.output_dir.mkdir(parents=True)
        (self.output_dir / "keep.txt").write_text("do not replace", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "already exists"):
            snapshot.build_snapshot(self.source_root, self.output_dir, "snapshot_v0", False, "2026-07-10T12:30:00Z")
        self.assertEqual((self.output_dir / "keep.txt").read_text(encoding="utf-8"), "do not replace")

    def test_formal_freeze_rejects_unresolved_labels(self) -> None:
        snapshot = load_snapshot_module()
        self._write_lanes([self._valid_lane(label="MS?")])

        with self.assertRaisesRegex(snapshot.SnapshotValidationError, "unresolved labels"):
            snapshot.build_snapshot(self.source_root, self.output_dir, "frozen_v1", True, "2026-07-10T12:30:00Z")

    def test_formal_freeze_allows_versioned_approved_exclusions(self) -> None:
        snapshot = load_snapshot_module()
        self._write_lanes([self._valid_lane(label="unknown")])

        result = snapshot.build_snapshot(self.source_root, self.output_dir, "frozen_v1", True, "2026-07-10T12:30:00Z")

        self.assertTrue(result.formal)
        audit = json.loads((self.output_dir / "audit_report.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["excluded_label_count"], 1)
        self.assertEqual(audit["unresolved_label_count"], 0)


if __name__ == "__main__":
    unittest.main()
