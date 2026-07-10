from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TrainingRareTests(unittest.TestCase):
    def test_allele_encoding_is_multihot_and_named_variants_are_not_guessed(self) -> None:
        from aat_training.rare import build_allele_vocabulary, encode_alleles

        vocabulary = build_allele_vocabulary([("M", "Z"), ("F", "S"), ("M",)])
        self.assertEqual(vocabulary, ("F", "M", "S", "Z"))
        np.testing.assert_array_equal(encode_alleles(("M", "Z"), vocabulary), [0, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, "unknown allele"):
            encode_alleles(("MZBristol",), vocabulary)

    def test_cosine_top3_excludes_every_reference_from_query_gel(self) -> None:
        from aat_training.rare import ReferenceItem, retrieve_top_k

        bank = [
            ReferenceItem("same", "G1", "MZ", np.array([1.0, 0.0])),
            ReferenceItem("z", "G2", "ZZ", np.array([0.9, 0.1])),
            ReferenceItem("m", "G3", "M", np.array([0.7, 0.3])),
            ReferenceItem("s", "G4", "SS", np.array([0.0, 1.0])),
        ]
        result = retrieve_top_k(np.array([1.0, 0.0]), "G1", bank, k=3)
        self.assertEqual([item.lane_id for item in result], ["z", "m", "s"])
        self.assertNotIn("same", [item.lane_id for item in result])

    def test_singleton_is_returned_as_case_not_class_accuracy(self) -> None:
        from aat_training.rare import summarize_rare_cases

        summary = summarize_rare_cases([{"true_label": "MZBristol", "referred": True, "top3_labels": ["MZBristol", "MZ", "MS"]}])
        self.assertEqual(summary["class_accuracy"], {})
        self.assertTrue(summary["singleton_cases"][0]["top3_hit"])

    def test_multitask_weight_is_preregistered(self) -> None:
        from aat_training.rare import validate_multitask_weight

        for value in (0.2, 0.5, 1.0):
            validate_multitask_weight(value)
        with self.assertRaises(ValueError):
            validate_multitask_weight(0.7)

    def test_best_backbone_upgrade_emits_all_three_outputs(self) -> None:
        import torch
        from aat_training.rare import build_multitask_backbone

        model = build_multitask_backbone("resnet18", common_classes=6, allele_classes=4, embedding_dim=16, pretrained=False)
        outputs = model(torch.rand(2, 3, 128, 384))
        self.assertEqual(tuple(outputs["common_logits"].shape), (2, 6))
        self.assertEqual(tuple(outputs["allele_logits"].shape), (2, 4))
        self.assertEqual(tuple(outputs["embedding"].shape), (2, 16))
        torch.testing.assert_close(outputs["embedding"].norm(dim=1), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
