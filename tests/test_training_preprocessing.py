from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def load_preprocessing_module():
    try:
        from aat_training import preprocessing
    except (ModuleNotFoundError, ImportError) as exc:
        raise AssertionError("aat_training.preprocessing has not been implemented") from exc
    return preprocessing


class TrainingPreprocessingTests(unittest.TestCase):
    def test_crop_lane_returns_exact_reviewed_rectangle(self) -> None:
        preprocessing = load_preprocessing_module()
        image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        lane = {"left_x": "2", "right_x": "6", "roi_y_start": "1", "roi_y_end": "6"}

        crop = preprocessing.crop_lane(image, lane)

        np.testing.assert_array_equal(crop, image[1:6, 2:6, :])
        self.assertFalse(np.shares_memory(crop, image))

    def test_crop_lane_rejects_invalid_or_out_of_bounds_geometry(self) -> None:
        preprocessing = load_preprocessing_module()
        image = np.zeros((8, 10, 3), dtype=np.uint8)

        for lane in (
            {"left_x": "6", "right_x": "2", "roi_y_start": "1", "roi_y_end": "6"},
            {"left_x": "2", "right_x": "12", "roi_y_start": "1", "roi_y_end": "6"},
        ):
            with self.subTest(lane=lane), self.assertRaisesRegex(ValueError, "lane geometry"):
                preprocessing.crop_lane(image, lane)

    def test_letterbox_preserves_aspect_ratio_and_uses_horizontal_padding(self) -> None:
        preprocessing = load_preprocessing_module()
        crop = np.full((100, 20, 3), 80, dtype=np.uint8)

        output, metadata = preprocessing.letterbox_rgb(crop, target_height=128, target_width=384, pad_value=255)

        self.assertEqual(output.shape, (128, 384, 3))
        self.assertEqual((metadata.resized_height, metadata.resized_width), (128, 26))
        self.assertEqual((metadata.pad_top, metadata.pad_bottom), (0, 0))
        self.assertEqual((metadata.pad_left, metadata.pad_right), (179, 179))
        self.assertAlmostEqual(metadata.scale, 1.28)
        self.assertTrue(np.all(output[:, : metadata.pad_left] == 255))
        self.assertTrue(np.all(output[:, metadata.pad_left : 384 - metadata.pad_right] == 80))

    def test_profile_ignores_padding_and_reports_dark_band_intensity(self) -> None:
        preprocessing = load_preprocessing_module()
        letterboxed = np.full((4, 6, 3), 255, dtype=np.uint8)
        letterboxed[:, 2:4, :] = 255
        letterboxed[1, 2:4, :] = 0

        profile = preprocessing.extract_intensity_profile(letterboxed, (2, 0, 4, 4), output_length=4)

        np.testing.assert_allclose(profile, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)

    def test_build_inputs_uses_snapshot_metadata_and_writes_traceable_outputs(self) -> None:
        preprocessing = load_preprocessing_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_root = root / "source"
            snapshot_dir = root / "snapshot_v0"
            output_dir = root / "derived" / "preprocessing_v1"
            image_path = source_root / "dataset" / "Originial" / "gel.png"
            image_path.parent.mkdir(parents=True)
            image = np.full((8, 10, 3), 255, dtype=np.uint8)
            image[3, 2:6, :] = 0
            Image.fromarray(image).save(image_path)
            self._write_snapshot_fixture(snapshot_dir, source_root, image_path)

            result = preprocessing.build_training_inputs(snapshot_dir, output_dir, target_height=128, target_width=384)

            self.assertEqual(result.lane_count, 1)
            self.assertEqual(result.profile_length, 128)
            crop_path = output_dir / "crops" / "IMG_0001_L001.png"
            self.assertTrue(crop_path.is_file())
            with Image.open(crop_path) as crop_image:
                self.assertEqual(crop_image.size, (384, 128))
            with (output_dir / "inputs.csv").open(newline="", encoding="utf-8") as csv_file:
                row = next(csv.DictReader(csv_file))
            self.assertEqual(row["lane_id"], "IMG_0001_L001")
            self.assertEqual(row["snapshot_version"], "snapshot_v0")
            expected_image_hash = __import__("hashlib").sha256(image_path.read_bytes()).hexdigest()
            self.assertEqual(row["source_image_sha256"], expected_image_hash)
            with np.load(output_dir / "profiles.npz") as arrays:
                self.assertEqual(arrays["profiles"].shape, (1, 128))
                self.assertEqual(arrays["lane_ids"].tolist(), ["IMG_0001_L001"])
            summary = json.loads((output_dir / "qc_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["lane_count"], 1)
            self.assertEqual(summary["target_shape"], [128, 384, 3])
            self.assertTrue((output_dir / "qc_montage.png").is_file())

    def test_build_inputs_refuses_to_overwrite_existing_output(self) -> None:
        preprocessing = load_preprocessing_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "existing"
            output_dir.mkdir()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                preprocessing.build_training_inputs(Path(tmp_dir) / "missing", output_dir)

    def _write_snapshot_fixture(self, snapshot_dir: Path, source_root: Path, image_path: Path) -> None:
        import hashlib

        snapshot_dir.mkdir(parents=True)
        lanes_path = snapshot_dir / "lanes.csv"
        with lanes_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "lane_id",
                    "image_id",
                    "parent_gel",
                    "source_filename",
                    "candidate_index",
                    "roi_y_start",
                    "roi_y_end",
                    "left_x",
                    "right_x",
                    "canonical_label",
                    "common_eligible",
                    "qc_status",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "lane_id": "IMG_0001_L001",
                    "image_id": "IMG_0001",
                    "parent_gel": "IMG_0001",
                    "source_filename": "gel.png",
                    "candidate_index": "1",
                    "roi_y_start": "1",
                    "roi_y_end": "6",
                    "left_x": "2",
                    "right_x": "6",
                    "canonical_label": "M",
                    "common_eligible": "1",
                    "qc_status": "common",
                }
            )
        lanes_hash = hashlib.sha256(lanes_path.read_bytes()).hexdigest()
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        manifest = {
            "version": "snapshot_v0",
            "source_root": str(source_root),
            "files": {"lanes.csv": {"sha256": lanes_hash}},
            "images": {
                "IMG_0001": {
                    "relative_path": image_path.relative_to(source_root).as_posix(),
                    "source_filename": "gel.png",
                    "sha256": image_hash,
                    "width": 10,
                    "height": 8,
                }
            },
        }
        (snapshot_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
