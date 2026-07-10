"""ImageNet-pretrained CNN registry and small, testable training primitives."""

from __future__ import annotations

import csv
import itertools
import math
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .labels import COMMON_CLASSES


REGISTERED_BACKBONES = ("resnet18", "efficientnet_b0", "resnet50", "inception_v3")
TUNED_BACKBONES = ("resnet18", "efficientnet_b0")
EXPECTED_GRIDS = {
    "learning_rates": {1e-4, 3e-4},
    "weight_decays": {1e-4, 1e-3},
    "dropouts": {0.2, 0.4},
}
CHECKPOINT_PROVENANCE_FIELDS = (
    "experiment_id",
    "dataset_version",
    "dataset_manifest_sha256",
    "fold_version",
    "fold_manifest_sha256",
    "code_revision",
    "seed",
    "outer_fold",
    "config",
)


def validate_cnn_config(config: Mapping[str, Any]) -> None:
    backbone = str(config.get("backbone", ""))
    if backbone not in REGISTERED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if config.get("loss") != "inverse_sqrt_weighted_ce":
        raise ValueError("loss must be inverse_sqrt_weighted_ce; DIoU is not a classification loss")
    if int(config.get("batch_size", 0)) != 32 or int(config.get("max_epochs", 0)) != 60 or int(config.get("patience", 0)) != 8:
        raise ValueError("batch_size/max_epochs/patience must match preregistered defaults")
    if backbone in TUNED_BACKBONES:
        for field, expected in EXPECTED_GRIDS.items():
            actual = {float(value) for value in config.get(field, [])}
            if actual != expected:
                raise ValueError(f"{field} must equal the preregistered grid {sorted(expected)}")
    else:
        for field in ("learning_rate", "weight_decay", "dropout"):
            if field not in config:
                raise ValueError(f"Proposal comparator requires fixed {field}")


def inverse_sqrt_class_weights(counts: Sequence[int]):
    import torch

    tensor = torch.as_tensor(list(counts), dtype=torch.float32)
    if tensor.ndim != 1 or tensor.numel() < 2 or bool((tensor <= 0).any()):
        raise ValueError("Every class count must be positive")
    weights = tensor.rsqrt()
    return weights / weights.max()


def build_backbone(backbone_name: str, class_count: int, dropout: float, pretrained: bool = True):
    """Construct one registered torchvision model; downloads occur only when requested."""

    if backbone_name not in REGISTERED_BACKBONES:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    if class_count < 2 or not 0 <= float(dropout) < 1:
        raise ValueError("Invalid class_count or dropout")
    from torch import nn
    from torchvision import models

    if backbone_name in ("resnet18", "resnet50"):
        constructor = getattr(models, backbone_name)
        weights_enum = getattr(models, f"{backbone_name.replace('resnet', 'ResNet')}_Weights")
        model = constructor(weights=weights_enum.DEFAULT if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(model.fc.in_features, class_count))
    elif backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(in_features, class_count))
    else:
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT if pretrained else None,
            aux_logits=pretrained,
            init_weights=pretrained,
        )
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(model.fc.in_features, class_count))
    model.backbone_name = backbone_name
    model.pretrained_loaded = bool(pretrained)
    return model


def train_batches(model, loader, optimizer, class_weights, device: str, amp: bool, max_batches: int | None = None) -> dict[str, float | int]:
    """Train a bounded number of batches, shared by smoke tests and the full runner."""

    import torch
    from torch import nn

    model.to(device)
    model.train()
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    use_amp = bool(amp and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses: list[float] = []
    for batch_index, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            if hasattr(logits, "logits"):
                logits = logits.logits
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError("Training produced no finite batches")
    return {"batches": len(losses), "mean_loss": sum(losses) / len(losses)}


def save_checkpoint(path, model, provenance: Mapping[str, Any], epoch: int, validation_macro_f1: float) -> None:
    """Write a non-overwriting, self-describing model checkpoint."""

    import torch
    from pathlib import Path

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    missing = [field for field in CHECKPOINT_PROVENANCE_FIELDS if field not in provenance]
    if missing:
        raise ValueError(f"Checkpoint provenance is missing: {', '.join(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "aat-cnn-checkpoint-v1",
            "model_state_dict": model.state_dict(),
            "provenance": dict(provenance),
            "epoch": int(epoch),
            "validation_macro_f1": float(validation_macro_f1),
        },
        path,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _candidate_configs(config: Mapping[str, Any]) -> list[dict[str, float]]:
    if config["backbone"] in TUNED_BACKBONES:
        return [
            {"learning_rate": lr, "weight_decay": wd, "dropout": dropout}
            for lr, wd, dropout in itertools.product(config["learning_rates"], config["weight_decays"], config["dropouts"])
        ]
    return [{field: float(config[field]) for field in ("learning_rate", "weight_decay", "dropout")}]


def _make_image_dataset(records: Sequence[Mapping[str, str]], inputs_dir: Path, training: bool, augmentation_config: Mapping[str, Any]):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms
    from .augmentations import build_training_transform

    augmentation = build_training_transform(augmentation_config) if training else None
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    label_indices = {label: index for index, label in enumerate(COMMON_CLASSES)}

    class LaneDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            record = records[index]
            with Image.open(inputs_dir / record["crop_path"]) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            if augmentation is not None:
                tensor = augmentation(tensor)
                noise_std = float(augmentation_config.get("noise_std", 0.01))
                if noise_std:
                    tensor = (tensor + torch.randn_like(tensor) * noise_std).clamp(0, 1)
            return normalize(tensor), label_indices[record["canonical_label"]], record["lane_id"]

    return LaneDataset()


def _evaluate_batches(model, loader, device: str) -> tuple[float, list[tuple[str, np.ndarray, int]]]:
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    rows: list[tuple[str, np.ndarray, int]] = []
    with torch.no_grad():
        for images, targets, lane_ids in loader:
            logits = model(images.to(device))
            if hasattr(logits, "logits"):
                logits = logits.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            for lane_id, probability, target in zip(lane_ids, probabilities, targets.numpy()):
                rows.append((str(lane_id), probability, int(target)))
    score = f1_score(
        [target for _, _, target in rows],
        [int(probability.argmax()) for _, probability, _ in rows],
        labels=list(range(len(COMMON_CLASSES))),
        average="macro",
        zero_division=0,
    )
    return float(score), rows


def _fit_with_validation(
    train_records,
    validation_records,
    inputs_dir: Path,
    config: Mapping[str, Any],
    candidate: Mapping[str, float],
    seed: int,
    device: str,
    pretrained: bool,
):
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = build_backbone(config["backbone"], len(COMMON_CLASSES), candidate["dropout"], pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    counts = [sum(row["canonical_label"] == label for row in train_records) for label in COMMON_CLASSES]
    weights = inverse_sqrt_class_weights(counts)
    train_loader = DataLoader(
        _make_image_dataset(train_records, inputs_dir, True, config["augmentations"]),
        batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config.get("num_workers", 0)), pin_memory=str(device).startswith("cuda"),
    )
    validation_loader = DataLoader(
        _make_image_dataset(validation_records, inputs_dir, False, config["augmentations"]),
        batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config.get("num_workers", 0)), pin_memory=str(device).startswith("cuda"),
    )
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_batches(model, train_loader, optimizer, weights, device, bool(config["amp"]))
        score, _ = _evaluate_batches(model, validation_loader, device)
        if score > best_score + 1e-12:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    model.load_state_dict(best_state)
    return model, best_score, best_epoch


def _fit_fixed_epochs(records, inputs_dir: Path, config: Mapping[str, Any], candidate: Mapping[str, float], seed: int, device: str, pretrained: bool, epochs: int):
    """Refit the selected configuration on every outer-training gel."""

    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    model = build_backbone(config["backbone"], len(COMMON_CLASSES), candidate["dropout"], pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"])
    counts = [sum(row["canonical_label"] == label for row in records) for label in COMMON_CLASSES]
    weights = inverse_sqrt_class_weights(counts)
    loader = DataLoader(
        _make_image_dataset(records, inputs_dir, True, config["augmentations"]),
        batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config.get("num_workers", 0)), pin_memory=str(device).startswith("cuda"),
    )
    for _ in range(int(epochs)):
        train_batches(model, loader, optimizer, weights, device, bool(config["amp"]))
    return model


def run_cnn_nested_cv(
    inputs_dir: Path,
    folds_path: Path,
    experiments_root: Path,
    experiment_id: str,
    provenance: Mapping[str, Any],
    config: Mapping[str, Any],
    device: str,
    pretrained: bool = True,
):
    """Execute fixed grouped nested CV and register checkpoints plus OOF predictions."""

    from torch.utils.data import DataLoader
    from .experiments import create_experiment, complete_experiment
    from .metrics import evaluate_common
    from .predictions import write_prediction_rows

    validate_cnn_config(config)
    inputs_dir = Path(inputs_dir)
    records = {row["lane_id"]: row for row in _read_csv(inputs_dir / "inputs.csv") if row["common_eligible"] == "1"}
    assignments = _read_csv(folds_path)
    run = create_experiment(experiments_root, experiment_id, dict(config), provenance)
    checkpoints = run.path / "checkpoints"
    checkpoints.mkdir()
    candidates = _candidate_configs(config)
    seed = int(provenance["seed"])
    all_predictions: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    checkpoint_paths: list[Path] = []
    for outer_fold in sorted({int(row["outer_fold"]) for row in assignments}):
        scenario = [row for row in assignments if int(row["outer_fold"]) == outer_fold]
        train_rows = [row for row in scenario if row["outer_role"] == "train" and row["lane_id"] in records]
        test_ids = sorted(row["lane_id"] for row in scenario if row["outer_role"] == "test" and row["lane_id"] in records)
        candidate_scores = []
        for candidate_index, candidate in enumerate(candidates):
            inner_scores, inner_epochs = [], []
            for inner_fold in sorted({int(row["inner_fold"]) for row in train_rows}):
                train_ids = [row["lane_id"] for row in train_rows if int(row["inner_fold"]) != inner_fold]
                validation_ids = [row["lane_id"] for row in train_rows if int(row["inner_fold"]) == inner_fold]
                _, score, epoch = _fit_with_validation(
                    [records[lane_id] for lane_id in train_ids], [records[lane_id] for lane_id in validation_ids], inputs_dir,
                    config, candidate, seed + outer_fold * 1000 + candidate_index * 10 + inner_fold, device, pretrained,
                )
                inner_scores.append(score)
                inner_epochs.append(epoch)
            candidate_scores.append({"params": candidate, "macro_f1": float(np.mean(inner_scores)), "fold_scores": inner_scores, "epochs": inner_epochs})
        best_index = max(range(len(candidate_scores)), key=lambda index: (candidate_scores[index]["macro_f1"], -index))
        best = candidate_scores[best_index]
        best_epoch = max(1, int(round(float(np.median(best["epochs"])))))
        validation_score = float(best["macro_f1"])
        model = _fit_fixed_epochs(
            [records[row["lane_id"]] for row in train_rows], inputs_dir, config, best["params"], seed + outer_fold, device, pretrained, best_epoch
        )
        test_loader = DataLoader(_make_image_dataset([records[lane_id] for lane_id in test_ids], inputs_dir, False, config["augmentations"]), batch_size=int(config["batch_size"]), shuffle=False)
        _, predictions = _evaluate_batches(model, test_loader, device)
        for lane_id, probabilities, target in predictions:
            predicted = int(probabilities.argmax())
            row = {
                "experiment_id": experiment_id, "dataset_version": provenance["dataset_version"], "fold_version": provenance["fold_version"],
                "config_id": f"{experiment_id}-outer-{outer_fold}", "seed": seed, "code_revision": provenance["code_revision"],
                "lane_id": lane_id, "parent_gel": records[lane_id]["parent_gel"], "outer_fold": outer_fold,
                "true_label": COMMON_CLASSES[target], "predicted_label": COMMON_CLASSES[predicted],
            }
            row.update({f"prob_{label}": float(probabilities[index]) for index, label in enumerate(COMMON_CLASSES)})
            all_predictions.append(row)
        checkpoint_path = checkpoints / f"outer_fold_{outer_fold}.pt"
        checkpoint_provenance = {**dict(provenance), "experiment_id": experiment_id, "outer_fold": outer_fold, "config": {**dict(config), "selected": best["params"]}}
        save_checkpoint(checkpoint_path, model, checkpoint_provenance, best_epoch, validation_score)
        checkpoint_paths.append(checkpoint_path)
        fold_summaries.append({"outer_fold": outer_fold, "selected": best, "candidate_results": candidate_scores, "best_epoch": best_epoch})
    predictions_path = run.path / "predictions.csv"
    all_predictions.sort(key=lambda row: row["lane_id"])
    write_prediction_rows(predictions_path, all_predictions)
    metrics = evaluate_common(all_predictions)
    metrics["outer_folds"] = fold_summaries
    metrics_path = run.path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    complete_experiment(run, [predictions_path, metrics_path, *checkpoint_paths], {"backbone": config["backbone"], "oof_count": len(all_predictions), "macro_f1": metrics["macro_f1"]})
    return run.path
