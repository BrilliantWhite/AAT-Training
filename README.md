# AAT Training

Reproducible training pipeline for AAT electrophoresis lane classification and rare-phenotype decision support.

## Scope

- Six common phenotype classes: `M`, `MZ`, `MS`, `SZ`, `ZZ`, `SS`.
- Explainable 1D Logistic Regression and RBF-SVM baselines.
- ImageNet-pretrained ResNet-18, EfficientNet-B0, ResNet-50, and Inception-v3 comparisons.
- Allele multi-label output, cosine Top-3 reference retrieval, and low-confidence referral.
- Fixed outer 5-fold / inner 3-fold stratified group nested cross-validation by parent gel.

## Privacy boundary

This repository contains code, configurations, tests, and documentation only. Raw gel images, versioned datasets, crops, folds derived from private labels, checkpoints, predictions, and experiment outputs must never be committed.

## Local verification

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -p "test_training_*.py" -v
```

## Workflow

1. Build an immutable `snapshot_v0` during development.
2. After expert review, create formal `frozen_v1`.
3. Create traceable 128×384 letterboxed inputs and fixed grouped folds.
4. Run classical baselines locally.
5. Run CNN nested cross-validation on one 24GB-or-larger cloud GPU.
6. Upgrade the selected backbone with common, allele, and embedding heads.
7. Generate registered OOF metrics and dissertation evidence.

See [pipeline usage](docs/aat_training_pipeline_usage.md) and the [AutoDL beginner guide](docs/aat_cloud_training_beginner_guide.md).

