from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TrainingDuplicateAuditTests(unittest.TestCase):
    def test_exact_and_near_duplicates_are_reported_across_gels_only(self) -> None:
        from aat_training.duplicates import audit_crop_duplicates

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for lane_id, value in (("L1", 20), ("L2", 20), ("L3", 21), ("L4", 200)):
                image = Image.new("L", (32, 32), color=255)
                if lane_id == "L4":
                    for x in range(8, 24):
                        image.putpixel((x, 16), value)
                else:
                    for y in range(8, 24):
                        image.putpixel((16, y), value)
                image.convert("RGB").save(root / f"{lane_id}.png")
            rows = [
                {"lane_id": "L1", "parent_gel": "G1", "crop_path": "L1.png"},
                {"lane_id": "L2", "parent_gel": "G2", "crop_path": "L2.png"},
                {"lane_id": "L3", "parent_gel": "G3", "crop_path": "L3.png"},
                {"lane_id": "L4", "parent_gel": "G1", "crop_path": "L4.png"},
            ]
            report = audit_crop_duplicates(rows, root, near_hamming_threshold=2)
            exact_pairs = {(item["lane_a"], item["lane_b"]) for item in report["exact_cross_gel_pairs"]}
            self.assertIn(("L1", "L2"), exact_pairs)
            self.assertTrue(any({item["lane_a"], item["lane_b"]} == {"L1", "L3"} for item in report["near_cross_gel_pairs"]))
            self.assertFalse(any({item["lane_a"], item["lane_b"]} == {"L1", "L4"} for item in report["near_cross_gel_pairs"]))


if __name__ == "__main__":
    unittest.main()
