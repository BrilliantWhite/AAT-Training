from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aat_training.cnn import build_backbone, train_batches  # noqa: E402


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA is unavailable")
    model = build_backbone("resnet18", class_count=6, dropout=0.2, pretrained=False)
    # Two batches verify CUDA forward/backward, AMP, optimizer, and the 128x384 contract.
    dataset = TensorDataset(torch.rand(4, 3, 128, 384), torch.tensor([0, 1, 2, 3]))
    loader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    result = train_batches(model, loader, optimizer, torch.ones(6), device="cuda", amp=True, max_batches=2)
    result.update({
        "status": "pass",
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    })
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

