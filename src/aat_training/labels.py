"""Auditable AAT label normalization and task-eligibility decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


COMMON_CLASSES = ("M", "MZ", "MS", "SZ", "ZZ", "SS")
DEFAULT_SIMPLE_ALLELE_CODES = frozenset("BCEFGILMPQSTWXYZ")
DEFAULT_NAMED_VARIANTS = {
    "MZBRISTOL": "MZBristol",
    "MZPRATT": "MZPratt",
    "MMALTON": "MMalton",
    "ZZBRISTOL": "ZZBristol",
}
DEFAULT_UNKNOWNS = frozenset({"UNKNOWN", "UNLABELLED", "UNLABELED"})
ALNUM_LABEL = re.compile(r"^[A-Za-z0-9]+$")


@dataclass(frozen=True)
class LabelPolicy:
    schema_version: str = "aat-label-policy-v1"
    label_version: str = "v1"
    common_classes: tuple[str, ...] = COMMON_CLASSES
    canonical_aliases: dict[str, str] = field(default_factory=lambda: {"MM": "M"})
    named_variants: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_NAMED_VARIANTS))
    unknown_labels: frozenset[str] = DEFAULT_UNKNOWNS
    approved_exclusions: frozenset[str] = frozenset()
    simple_allele_codes: frozenset[str] = DEFAULT_SIMPLE_ALLELE_CODES
    degraded_token: str = "DEGRADED"


@dataclass(frozen=True)
class LabelDecision:
    original_label: str
    canonical_label: str | None
    common_eligible: bool
    alleles: tuple[str, ...]
    retrieval_eligible: bool
    referral_qc_eligible: bool
    qc_status: str
    reason: str
    label_version: str


def _unique_alleles(label: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(label))


def decide_label(original_label: str | None, policy: LabelPolicy | None = None) -> LabelDecision:
    """Derive training fields while preserving the stripped source label."""

    policy = policy or LabelPolicy()
    original = (original_label or "").strip()
    upper = original.upper()

    if not original:
        return _excluded(original, policy, "empty_label")
    if policy.degraded_token in upper:
        return LabelDecision(original, None, False, (), False, True, "degraded", "degraded_sample", policy.label_version)
    if upper in policy.approved_exclusions:
        return _excluded(original, policy, "approved_exclusion")
    if upper in policy.unknown_labels:
        return _excluded(original, policy, "approved_unknown_exclusion")
    if not ALNUM_LABEL.fullmatch(original):
        return _excluded(original, policy, "malformed_label")

    canonical = policy.canonical_aliases.get(upper, upper)
    if canonical in policy.common_classes:
        return LabelDecision(
            original,
            canonical,
            True,
            _unique_alleles(canonical),
            True,
            False,
            "common",
            "approved_common_class",
            policy.label_version,
        )

    if upper in policy.named_variants:
        return LabelDecision(
            original,
            policy.named_variants[upper],
            False,
            (),
            True,
            False,
            "named_variant",
            "named_variant_retrieval_only",
            policy.label_version,
        )

    if 1 <= len(upper) <= 2 and all(code in policy.simple_allele_codes for code in upper):
        return LabelDecision(
            original,
            upper,
            False,
            _unique_alleles(upper),
            True,
            False,
            "rare_simple",
            "simple_allele_label",
            policy.label_version,
        )

    if len(original) > 2:
        return LabelDecision(
            original,
            original,
            False,
            (),
            True,
            False,
            "named_variant",
            "unparsed_named_variant_retrieval_only",
            policy.label_version,
        )

    return _excluded(original, policy, "unsupported_allele_code")


def _excluded(original: str, policy: LabelPolicy, reason: str) -> LabelDecision:
    return LabelDecision(original, None, False, (), False, False, "excluded", reason, policy.label_version)


def load_label_policy(path: Path) -> LabelPolicy:
    """Load an explicit versioned YAML policy."""

    import yaml

    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return LabelPolicy(
        schema_version=str(data["schema_version"]),
        label_version=str(data["label_version"]),
        common_classes=tuple(str(value) for value in data["common_classes"]),
        canonical_aliases={str(key).upper(): str(value) for key, value in data["canonical_aliases"].items()},
        named_variants={str(key).upper(): str(value) for key, value in data["named_variants"].items()},
        unknown_labels=frozenset(str(value).upper() for value in data["unknown_labels"]),
        approved_exclusions=frozenset(str(value).upper() for value in data.get("approved_exclusions", [])),
        simple_allele_codes=frozenset(str(value).upper() for value in data["simple_allele_codes"]),
        degraded_token=str(data["degraded_token"]).upper(),
    )
