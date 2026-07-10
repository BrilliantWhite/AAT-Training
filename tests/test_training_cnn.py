from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TrainingCnnTests(unittest.TestCase):
    def test_registered_backbones_and_pretrained_contract(self) -> None:
        from aat_training.cnn import build_backbone

        for name in ("resnet18", "efficientnet_b0", "resnet50", "inception_v3"):
            model = build_backbone(name, class_count=6, dropout=0.2, pretrained=False)
            self.assertEqual(model.backbone_name, name)
            self.assertFalse(model.pretrained_loaded)
        with self.assertRaisesRegex(ValueError, "Unsupported backbone"):
            build_backbone("vit_b_16", 6, 0.2, pretrained=False)

    def test_config_validation_enforces_preregistered_search_spaces(self) -> None:
        from aat_training.cnn import validate_cnn_config

        valid = {
            "backbone": "resnet18",
            "pretrained": True,
            "batch_size": 32,
            "max_epochs": 60,
            "patience": 8,
            "amp": True,
            "learning_rates": [1e-4, 3e-4],
            "weight_decays": [1e-4, 1e-3],
            "dropouts": [0.2, 0.4],
            "loss": "inverse_sqrt_weighted_ce",
        }
        validate_cnn_config(valid)
        invalid = dict(valid, learning_rates=[1e-2])
        with self.assertRaisesRegex(ValueError, "learning_rates"):
            validate_cnn_config(invalid)
        invalid = dict(valid, loss="diou")
        with self.assertRaisesRegex(ValueError, "loss"):
            validate_cnn_config(invalid)

    def test_prohibited_augmentations_are_rejected(self) -> None:
        from aat_training.augmentations import validate_augmentation_config

        validate_augmentation_config({"rotation_degrees": 2, "vertical_flip": False, "mixup": False, "crop_scale_min": 0.95})
        for field, value in (("vertical_flip", True), ("mixup", True), ("rotation_degrees", 10), ("crop_scale_min", 0.7)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_augmentation_config({"rotation_degrees": 2, "vertical_flip": False, "mixup": False, "crop_scale_min": 0.95, field: value})

    def test_inverse_sqrt_weights_and_cpu_two_batch_training(self) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from aat_training.cnn import build_backbone, inverse_sqrt_class_weights, train_batches

        weights = inverse_sqrt_class_weights([100, 25, 4])
        np.testing.assert_allclose(weights.numpy(), np.array([0.2, 0.4, 1.0]), rtol=1e-6)
        model = build_backbone("resnet18", class_count=6, dropout=0.2, pretrained=False)
        loader = DataLoader(TensorDataset(torch.rand(4, 3, 128, 384), torch.tensor([0, 1, 2, 3])), batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        result = train_batches(model, loader, optimizer, torch.ones(6), device="cpu", amp=False, max_batches=2)
        self.assertEqual(result["batches"], 2)
        self.assertTrue(np.isfinite(result["mean_loss"]))

    def test_training_accepts_traceable_batches_with_lane_ids(self) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from aat_training.cnn import build_backbone, train_batches

        model = build_backbone("resnet18", class_count=6, dropout=0.2, pretrained=False)
        dataset = TensorDataset(torch.rand(2, 3, 128, 384), torch.tensor([0, 1]), torch.tensor([101, 102]))
        loader = DataLoader(dataset, batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        result = train_batches(model, loader, optimizer, torch.ones(6), device="cpu", amp=False, max_batches=1)
        self.assertEqual(result["batches"], 1)

    def test_checkpoint_contains_required_training_provenance(self) -> None:
        import torch
        from aat_training.cnn import build_backbone, save_checkpoint

        model = build_backbone("resnet18", 6, 0.2, pretrained=False)
        required = {
            "experiment_id": "EXP-CNN-001",
            "dataset_version": "snapshot_v0",
            "dataset_manifest_sha256": "a" * 64,
            "fold_version": "folds_v1",
            "fold_manifest_sha256": "b" * 64,
            "code_revision": "abcdef1",
            "seed": 1,
            "outer_fold": 0,
            "config": {"backbone": "resnet18"},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "checkpoint.pt"
            save_checkpoint(path, model, required, epoch=2, validation_macro_f1=0.4)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["provenance"], required)
            self.assertEqual(payload["epoch"], 2)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                save_checkpoint(path, model, required, epoch=3, validation_macro_f1=0.5)


if __name__ == "__main__":
    unittest.main()
