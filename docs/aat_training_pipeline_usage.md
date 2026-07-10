# AAT Training Pipeline Usage

All formal commands consume `frozen_v1`; only the snapshot builder may read a live review export.

## 1. Create the formal snapshot

```bash
python scripts/build_training_snapshot.py \
  --source-root /private/AAT_Project \
  --output-dir /private/data/frozen_v1 \
  --version frozen_v1 \
  --freeze
```

The command is non-overwriting. Preserve `original_label`; only the versioned policy derives `canonical_label` and eligibility.

## 2. Build inputs and fixed folds

```bash
python scripts/build_training_inputs.py \
  --snapshot-dir /private/data/frozen_v1 \
  --image-root /private/AAT_Project \
  --output-dir /private/data/frozen_v1_inputs_v1

python scripts/build_training_folds.py \
  --lanes /private/data/frozen_v1/lanes.csv \
  --output-dir /private/data/frozen_v1_folds_v1 \
  --version folds_v1
```

The fold artifact contains five outer scenarios. In each scenario, complete parent gels are outer-train or outer-test; the outer-train gels are divided into three inner folds. Every eligible lane receives exactly one outer-test prediction.

## 3. Run classical baselines

Use `scripts/train_classical_baselines.py` once for `logistic` and once for `rbf_svm`. Supply the snapshot manifest and fold manifest so their SHA-256 values are recorded with the run.

## 4. Verify a cloud GPU

```bash
python scripts/cloud/verify_cloud_environment.py \
  --dataset-manifest /private/data/frozen_v1/manifest.json \
  --fold-manifest /private/data/frozen_v1_folds_v1/manifest.json \
  --folds /private/data/frozen_v1_folds_v1/folds.csv
```

Formal training stops if CUDA is unavailable, VRAM is below the configured threshold, the dataset is not formal, a registered hash differs, or gel leakage is detected.

## 5. Run CNN baselines

```bash
python scripts/train_cnn_baselines.py \
  --config configs/training/resnet18_v1.yaml \
  --inputs-dir /private/data/frozen_v1_inputs_v1 \
  --folds /private/data/frozen_v1_folds_v1/folds.csv \
  --dataset-manifest /private/data/frozen_v1/manifest.json \
  --fold-manifest /private/data/frozen_v1_folds_v1/manifest.json \
  --experiment-id EXP-FROZEN-V1-RESNET18-001 \
  --dataset-version frozen_v1 \
  --fold-version folds_v1 \
  --device cuda
```

Repeat with the EfficientNet-B0, ResNet-50 proposal, and Inception-v3 proposal configuration files. ResNet-18 and EfficientNet-B0 search the preregistered inner grid; proposal comparators use fixed configurations.

## 6. Generate evidence

```bash
python scripts/generate_training_report.py \
  --experiment experiments/EXP-FROZEN-V1-RESNET18-001 \
  --experiment experiments/EXP-FROZEN-V1-EFFICIENTNET-B0-001 \
  --output-dir reports/FORMAL-CNN-COMPARISON-001 \
  --bootstrap-iterations 1000
```

Only complete registered runs are accepted. The report records model comparison metrics, gel-clustered confidence intervals, confusion matrices, calibration plots, and provenance.

