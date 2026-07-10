from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def load_labels_module():
    try:
        from aat_training import labels
    except ModuleNotFoundError as exc:
        raise AssertionError("aat_training.labels has not been implemented") from exc
    return labels


class TrainingLabelPolicyTests(unittest.TestCase):
    def test_common_classes_and_mm_are_canonicalized_without_losing_original(self) -> None:
        labels = load_labels_module()

        for original, canonical, alleles in (
            ("M", "M", ("M",)),
            ("MM", "M", ("M",)),
            ("MZ", "MZ", ("M", "Z")),
            ("SS", "SS", ("S",)),
        ):
            with self.subTest(original=original):
                decision = labels.decide_label(original)
                self.assertEqual(decision.original_label, original)
                self.assertEqual(decision.canonical_label, canonical)
                self.assertTrue(decision.common_eligible)
                self.assertEqual(decision.alleles, alleles)
                self.assertEqual(decision.qc_status, "common")

    def test_simple_non_common_biallelic_label_is_allele_and_retrieval_eligible(self) -> None:
        labels = load_labels_module()

        decision = labels.decide_label("FS")

        self.assertEqual(decision.original_label, "FS")
        self.assertEqual(decision.canonical_label, "FS")
        self.assertFalse(decision.common_eligible)
        self.assertEqual(decision.alleles, ("F", "S"))
        self.assertTrue(decision.retrieval_eligible)
        self.assertEqual(decision.qc_status, "rare_simple")

    def test_named_variant_remains_distinct_and_retrieval_only(self) -> None:
        labels = load_labels_module()

        decision = labels.decide_label("MZBristol")

        self.assertEqual(decision.canonical_label, "MZBristol")
        self.assertFalse(decision.common_eligible)
        self.assertEqual(decision.alleles, ())
        self.assertTrue(decision.retrieval_eligible)
        self.assertEqual(decision.qc_status, "named_variant")

    def test_degraded_label_is_excluded_from_common_loss_and_kept_for_referral_qc(self) -> None:
        labels = load_labels_module()

        decision = labels.decide_label("MS DEGRADED")

        self.assertIsNone(decision.canonical_label)
        self.assertFalse(decision.common_eligible)
        self.assertEqual(decision.alleles, ())
        self.assertFalse(decision.retrieval_eligible)
        self.assertTrue(decision.referral_qc_eligible)
        self.assertEqual(decision.qc_status, "degraded")

    def test_unknown_and_malformed_labels_are_excluded_with_reasons(self) -> None:
        labels = load_labels_module()

        unknown = labels.decide_label("unknown")
        malformed = labels.decide_label("MS]")

        self.assertIsNone(unknown.canonical_label)
        self.assertEqual(unknown.qc_status, "excluded")
        self.assertEqual(unknown.reason, "approved_unknown_exclusion")
        self.assertIsNone(malformed.canonical_label)
        self.assertEqual(malformed.qc_status, "excluded")
        self.assertEqual(malformed.reason, "malformed_label")

    def test_empty_label_is_excluded_instead_of_raising(self) -> None:
        labels = load_labels_module()

        decision = labels.decide_label("  ")

        self.assertEqual(decision.original_label, "")
        self.assertIsNone(decision.canonical_label)
        self.assertEqual(decision.reason, "empty_label")

    def test_policy_can_explicitly_approve_an_exclusion_without_guessing_a_label(self) -> None:
        labels = load_labels_module()
        policy = labels.LabelPolicy(approved_exclusions=frozenset({"MS]"}))
        decision = labels.decide_label("MS]", policy)
        self.assertIsNone(decision.canonical_label)
        self.assertEqual(decision.reason, "approved_exclusion")
        self.assertEqual(decision.original_label, "MS]")


if __name__ == "__main__":
    unittest.main()
