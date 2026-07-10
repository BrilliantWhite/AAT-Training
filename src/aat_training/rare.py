"""Allele targets, embedding retrieval, and rare-case reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MULTITASK_WEIGHTS = (0.2, 0.5, 1.0)


@dataclass(frozen=True)
class ReferenceItem:
    lane_id: str
    parent_gel: str
    label: str
    embedding: np.ndarray


@dataclass(frozen=True)
class RetrievalResult:
    lane_id: str
    parent_gel: str
    label: str
    cosine_similarity: float


def build_allele_vocabulary(allele_rows: Iterable[Sequence[str]]) -> tuple[str, ...]:
    vocabulary = sorted({str(allele) for row in allele_rows for allele in row if str(allele)})
    if not vocabulary:
        raise ValueError("Allele vocabulary is empty")
    return tuple(vocabulary)


def encode_alleles(alleles: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    indices = {allele: index for index, allele in enumerate(vocabulary)}
    unknown = [allele for allele in alleles if allele not in indices]
    if unknown:
        raise ValueError(f"Cannot encode unknown allele: {unknown[0]}")
    result = np.zeros(len(vocabulary), dtype=np.float32)
    for allele in alleles:
        result[indices[allele]] = 1.0
    return result


def validate_multitask_weight(value: float) -> None:
    if float(value) not in MULTITASK_WEIGHTS:
        raise ValueError(f"Multi-task weight must be one of {MULTITASK_WEIGHTS}")


def _normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a finite vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Embedding cannot be the zero vector")
    return vector / norm


def retrieve_top_k(query_embedding: np.ndarray, query_parent_gel: str, bank: Sequence[ReferenceItem], k: int = 3) -> list[RetrievalResult]:
    """Cosine rank a training-only bank while excluding the entire query gel."""

    if k <= 0:
        raise ValueError("k must be positive")
    query = _normalized(query_embedding)
    candidates: list[RetrievalResult] = []
    for item in bank:
        if item.parent_gel == query_parent_gel:
            continue
        similarity = float(np.dot(query, _normalized(item.embedding)))
        candidates.append(RetrievalResult(item.lane_id, item.parent_gel, item.label, similarity))
    candidates.sort(key=lambda item: (-item.cosine_similarity, item.lane_id))
    return candidates[:k]


def summarize_rare_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report singletons as auditable cases rather than misleading accuracies."""

    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["true_label"])
        counts[label] = counts.get(label, 0) + 1
    singleton_cases: list[dict[str, Any]] = []
    class_accuracy: dict[str, float] = {}
    for label in sorted(counts):
        label_rows = [row for row in rows if str(row["true_label"]) == label]
        if counts[label] == 1:
            row = label_rows[0]
            singleton_cases.append(
                {
                    "true_label": label,
                    "referred": bool(row["referred"]),
                    "top3_labels": list(row["top3_labels"]),
                    "top3_hit": label in row["top3_labels"],
                }
            )
        else:
            class_accuracy[label] = sum(label in row["top3_labels"] for row in label_rows) / len(label_rows)
    return {"class_accuracy": class_accuracy, "singleton_cases": singleton_cases}


def build_multitask_head(feature_dim: int, common_classes: int, allele_classes: int, embedding_dim: int, dropout: float = 0.2):
    """Return a head emitting common logits, allele logits, and L2 embeddings."""

    import torch.nn as nn

    class MultiTaskHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.common = nn.Linear(feature_dim, common_classes)
            self.allele = nn.Linear(feature_dim, allele_classes)
            self.embedding = nn.Linear(feature_dim, embedding_dim)

        def forward(self, features):
            import torch.nn.functional as functional

            hidden = self.dropout(features)
            return {
                "common_logits": self.common(hidden),
                "allele_logits": self.allele(hidden),
                "embedding": functional.normalize(self.embedding(hidden), dim=1),
            }

    return MultiTaskHead()


def build_multitask_backbone(backbone_name: str, common_classes: int, allele_classes: int, embedding_dim: int = 128, dropout: float = 0.2, pretrained: bool = True):
    """Upgrade the selected CNN backbone with common, allele, and embedding heads."""

    import torch.nn as nn
    from .cnn import build_backbone

    backbone = build_backbone(backbone_name, common_classes, dropout, pretrained=pretrained)
    if backbone_name.startswith("resnet") or backbone_name == "inception_v3":
        feature_dim = backbone.fc[-1].in_features
        backbone.fc = nn.Identity()
    elif backbone_name == "efficientnet_b0":
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
    else:  # build_backbone is the authoritative registry, retained defensively.
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    head = build_multitask_head(feature_dim, common_classes, allele_classes, embedding_dim, dropout)

    class MultiTaskBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.backbone_name = backbone_name

        def forward(self, images):
            features = self.backbone(images)
            if hasattr(features, "logits"):
                features = features.logits
            return self.head(features)

    return MultiTaskBackbone()
