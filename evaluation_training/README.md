# LeFusion Evaluation Training Pipeline

A comprehensive, modular pipeline for reproducing evaluation results from the LeFusion paper: **"Synthesizing Pathological Medical Images using Controllable Diffusion Models"**

> **Note**: This pipeline was restructured from `evaluation_pipeline_v2` to `evaluation_training` for better clarity and organization.

## 📋 Overview

This pipeline reproduces the exact evaluation metrics from the LeFusion paper with integrated official evaluation metrics:
- **Table 1**: Lung Nodule Segmentation (LIDC-IDRI) - DICE and NSD metrics
- **Table 2**: Cardiac Lesion Segmentation (EMIDEC) - DICE metrics
- **Official Metrics**: Integrated from [LeFusion repository](https://github.com/M3DV/LeFusion)

## 🏗️ Architecture

```
LeFusion/                               # Project root
├── evaluation_training/                # Main evaluation pipeline (this directory)
│   ├── configs/
│   │   └── experiment_config.yaml     # Central configuration
│   ├── evaluation_metrics/            # Official metrics from LeFusion
│   │   ├── get_Dice.py               # Official DICE implementation
│   │   ├── get_NSD.py                # Official NSD implementation
│   │   └── README.md                  # Metrics documentation
│   ├── synthetic_generation/
│   │   └── generate_synthetic_data.py # Synthetic data generation
│   ├── training/
│   │   └── train_segmentation.py     # Model training (nnU-Net, SwinUNETR)
│   ├── evaluation/
│   │   ├── evaluate_models.py        # Model evaluation with official metrics
│   │   └── compare_metrics.py        # Metric comparison utility
│   ├── utils/
│   │   └── test_paths.py             # Path verification utility
│   ├── run_complete_evaluation.py    # Master orchestrator
│   └── test_official_metrics.py      # Metrics validation suite
├── utility_training_resources/        # Shared resources
│   ├── datasets/
│   │   ├── LIDC_real/                # LIDC dataset splits
│   │   └── EMIDEC_real/              # EMIDEC dataset splits
│   └── DiffTumor/
│       └── STEP3.SegmentationModel/  # Segmentation models
├── LeFusion/                          # LeFusion models
│   └── LeFusion_Model/
├── DiffMask/                          # DiffMask models
│   └── DiffMask_Model/
└── data/                              # Raw datasets
    ├── LIDC/
    └── EMIDEC/
```

## 🚀 Quick Start

### Prerequisites

```bash
# Navigate to evaluation_training directory (from project root)
cd evaluation_training

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify paths are correct
python test_paths.py
```

### Complete Pipeline Execution

```bash
# Run from evaluation_training directory
cd evaluation_training

# Full paper reproduction with all methods (baseline, lefusion, lefusion_h, lefusion_h_diffmask)
python run_complete_evaluation.py \
    --dataset lidc \
    --model-types pretrained \
    --methods baseline lefusion lefusion_h lefusion_h_diffmask

# Quick test with baseline only
python run_complete_evaluation.py --quick-test

# Skip synthetic generation (if already generated)
python run_complete_evaluation.py \
    --dataset lidc \
    --model-types pretrained \
    --skip-synthetic

# Skip training (if models already trained)
python run_complete_evaluation.py \
    --dataset lidc \
    --model-types pretrained \
    --skip-synthetic \
    --skip-training
```

## 📊 Methods Overview

| Method | Description | Synthetic Data Types | Performance Gain |
|--------|-------------|---------------------|------------------|
| **Baseline** | Real pathological only | None (P only) | Reference |
| **LeFusion** | Base diffusion model | P+P' | +0.5% DICE |
| **LeFusion-H** | Histogram-conditioned | P+P', P+N' | +2.3% DICE |
| **LeFusion-H-DiffMask** | Enhanced with mask generation | P+N', P+N'', P+P'+N'' | **+5.2% DICE** |

**Notation:**
- P: Real pathological cases
- P': Synthetic pathological from pathological
- N': Synthetic pathological from normal
- N'': Enhanced synthetic with DiffMask

## 🔧 Detailed Usage

### Phase 1: Synthetic Data Generation

#### Generate All Methods (Recommended)

```bash
# Make sure you're in evaluation_training directory
# If at project root:
cd evaluation_training

# All methods with pretrained models
python synthetic_generation/generate_synthetic_data.py \
    --dataset lidc \
    --model-type pretrained \
    --methods all \
    --resume

# All methods with from-scratch models
python synthetic_generation/generate_synthetic_data.py \
    --dataset lidc \
    --model-type from_scratch \
    --methods all \
    --resume

# Both datasets (LIDC + EMIDEC)
python synthetic_generation/generate_synthetic_data.py \
    --dataset all \
    --model-type all \
    --methods all \
    --resume
```

#### Individual Method Generation

**LeFusion Base:**
```bash
python synthetic_generation/generate_synthetic_data.py \
    --dataset lidc \
    --model-type pretrained \
    --methods lefusion
```

**LeFusion-H (Histogram):**
```bash
python synthetic_generation/generate_synthetic_data.py \
    --dataset lidc \
    --model-type pretrained \
    --methods lefusion_h
```

**LeFusion-H-DiffMask:**
```bash
python synthetic_generation/generate_synthetic_data.py \
    --dataset lidc \
    --model-type pretrained \
    --methods lefusion_h_diffmask
```

#### Output Structure

```
evaluation_training/synthetic_data/
├── lidc/
│   ├── pretrained/
│   │   ├── lefusion/
│   │   │   └── P_P_prime/
│   │   │       ├── imagesTr/       # Synthetic images
│   │   │       └── labelsTr/       # Corresponding masks
│   │   ├── lefusion_h/
│   │   │   ├── P_P_prime/         # Pathological → Pathological
│   │   │   └── P_N_prime/         # Normal → Pathological
│   │   └── lefusion_h_diffmask/
│   │       ├── P_N_prime/         # Base enhancement
│   │       ├── P_N_double_prime/  # Double enhancement
│   │       └── P_P_prime_N_double_prime/  # Combined
│   └── from_scratch/
│       └── [same structure]
└── emidec/
    └── [same structure]
```

### Phase 2: Training Segmentation Models

#### Train All Models

```bash
# Make sure you're in evaluation_training directory
cd evaluation_training

# All methods with both architectures
python training/train_segmentation.py \
    --dataset lidc \
    --methods all \
    --model-types pretrained from_scratch \
    --seg-models nnunet swinunetr

# Specific configuration
python training/train_segmentation.py \
    --dataset lidc \
    --methods lefusion_h_diffmask \
    --model-types pretrained \
    --seg-models nnunet
```

#### Training Parameters
- **Epochs**: 200 (configurable)
- **Batch Size**: 1 (for 3D volumes)
- **Learning Rate**: 0.0004
- **Optimizer**: AdamW
- **Validation**: Every 200 iterations

### Phase 3: Evaluation with Official Metrics

#### Evaluate Models

```bash
# Make sure you're in evaluation_training directory
cd evaluation_training

# Evaluate all models and compare with paper
python evaluation/evaluate_models.py \
    --dataset lidc \
    --methods all \
    --model-types all \
    --seg-models nnunet swinunetr \
    --compare-paper

# Specific evaluation
python evaluation/evaluate_models.py \
    --dataset lidc \
    --method lefusion_h_diffmask \
    --model-type pretrained \
    --seg-model nnunet \
    --use-best-checkpoint
```

#### Epoch Selection for Evaluation

The evaluation pipeline supports selecting specific checkpoints or epochs for evaluation:

```bash
# Evaluate using the best checkpoint (best_metric_model.pth)
python evaluation/evaluate_models.py \
    --dataset lidc \
    --methods all \
    --use-best-checkpoint

# Evaluate at a specific epoch (e.g., epoch 100)
python evaluation/evaluate_models.py \
    --dataset lidc \
    --methods all \
    --checkpoint-epoch 100

# Evaluate at epoch 150 for a specific configuration
python evaluation/evaluate_models.py \
    --dataset lidc \
    --method lefusion_h_diffmask \
    --model-type pretrained \
    --seg-model swinunetr \
    --checkpoint-epoch 150
```

**Available Checkpoint Options:**
- `--use-best-checkpoint`: Uses `best_metric_model.pth` if available (highest validation DICE)
- `--checkpoint-epoch N`: Uses checkpoint from epoch N (e.g., `epoch_100.pt`)
- Default (no flag): Uses the latest available checkpoint in order:
  1. `model_final.pt` (completed training)
  2. `model.pt` (standard checkpoint)
  3. `best_metric_model.pth` (best validation)
  4. Latest `epoch_*.pt` file

**Notes on Checkpoint Selection:**
- The best checkpoint typically gives optimal results but may not exist if training was interrupted
- Specific epoch selection is useful for:
  - Comparing performance across training progress
  - Debugging convergence issues
  - Reproducing specific results
- If the requested epoch checkpoint doesn't exist, the evaluation will fail with an appropriate error message

**Evaluating Multiple Epochs:**
To evaluate models at multiple epochs for performance tracking:

```bash
# Evaluate at multiple specific epochs
for epoch in 50 100 150 200; do
    echo "Evaluating at epoch $epoch"
    python evaluation/evaluate_models.py \
        --dataset lidc \
        --methods lefusion_h_diffmask \
        --model-type pretrained \
        --seg-model nnunet \
        --checkpoint-epoch $epoch
done

# Compare early vs late training performance
python evaluation/evaluate_models.py --checkpoint-epoch 50   # Early training
python evaluation/evaluate_models.py --checkpoint-epoch 150  # Mid training
python evaluation/evaluate_models.py --use-best-checkpoint    # Best performance
```

#### Metric Validation

```bash
# Make sure you're in evaluation_training directory
cd evaluation_training

# Test official metrics integration
python test_official_metrics.py

# Compare metric implementations
python evaluation/compare_metrics.py --tolerance 1.0
```

## 📈 Expected Results

### LIDC-IDRI Dataset (Lung Nodules)

| Method | nnU-Net |  | SwinUNETR |  |
|--------|---------|-----|-----------|-----|
|        | DICE↑ | NSD↑ | DICE↑ | NSD↑ |
| Baseline | 78.26 | 88.90 | 78.38 | 88.67 |
| LeFusion | 78.77 | 89.25 | 78.43 | 88.54 |
| LeFusion-H | 80.62 | 90.90 | 80.95 | 90.98 |
| **LeFusion-H-DiffMask** | **83.44** | **93.35** | **83.13** | **93.20** |

### EMIDEC Dataset (Cardiac Lesions)

| Method | nnU-Net |  | SwinUNETR |  |
|--------|---------|-----|-----------|-----|
|        | MI↑ | PMO↑ | MI↑ | PMO↑ |
| Baseline | 68.61 | 36.32 | 57.79 | 35.76 |
| LeFusion | 69.88 | 34.79 | 57.85 | 35.63 |
| LeFusion-H | 69.95 | 38.01 | 59.61 | 37.99 |
| **LeFusion-H-DiffMask** | **71.28** | **43.41** | **59.30** | **42.49** |

## 🎯 Model Types

### Pretrained Models
- Faster convergence (minutes to hours)
- Stable performance
- Recommended for reproduction

**Required files (relative to project root):**
```
LeFusion/LeFusion_Model/LIDC/lidc.pt
LeFusion/LeFusion_Model/EMIDEC/emidec.pt
DiffMask/DiffMask_Model/diffmask.pt
```

### From-Scratch Models
- Trained from random initialization
- Potentially better performance
- Longer training time (hours to days)

**Required files (relative to project root):**
```
LeFusion/LeFusion_Model/LIDC/model-50.pt
LeFusion/LeFusion_Model/EMIDEC/model-50.pt
DiffMask/DiffMask_Model/model-80.pt
```

## 🔄 Resume Capability

All phases support checkpoint-based resumption within evaluation_training:

```bash
# Synthetic generation checkpoint
evaluation_training/synthetic_data/[dataset]/[model_type]/generation_checkpoint.json

# Training automatically detects existing models
evaluation_training/trained_models/[dataset]/[method]/[model_type]/[seg_model]/

# Evaluation skips completed evaluations
evaluation_training/evaluation_results/[timestamp]/
```

## 📊 Evaluation Metrics

### DICE Coefficient
- Measures volumetric overlap
- Range: 0-100% (higher is better)
- Official implementation from LeFusion repository

### Normalized Surface Distance (NSD)
- Measures surface alignment at 1mm tolerance
- Range: 0-100% (higher is better)
- Uses MONAI's compute_surface_dice

## 🛠️ Configuration

Edit `configs/experiment_config.yaml`:

```yaml
datasets:
  lidc:
    normal_dir: "../data/LIDC/Normal/Image"
    pathological_dir: "../data/LIDC/Pathological"
    real_data_dir: "../utility_training_resources/datasets/LIDC_real"
    
  emidec:
    normal_dir: "../data/EMIDEC/Normal"
    pathological_dir: "../data/EMIDEC/Pathological"
    real_data_dir: "../utility_training_resources/datasets/EMIDEC_real"
    
evaluation:
  nsd_tolerance: 1.0  # mm (paper default)
  
training:
  max_epochs: 200
  batch_size: 1
  learning_rate: 0.0004
```

## 📦 Data Preparation

### Real Data Splits

Before training, prepare split files:

```bash
# Make sure you're in evaluation_training directory
cd evaluation_training

# Auto-generate splits for LIDC
python utils/create_data_splits.py --dataset lidc

# Auto-generate splits for EMIDEC
python utils/create_data_splits.py --dataset emidec

# Verify structure (data will be in utility_training_resources)
ls -la ../utility_training_resources/datasets/LIDC_real/
# Should show:
# ├── imagesTr/
# ├── labelsTr/
# ├── real_lung_train_0.txt
# ├── real_lung_val_0.txt
# └── test.txt
```

## 🐛 Troubleshooting

### GPU Memory Issues
```yaml
# Reduce batch size in config
training:
  batch_size: 1
```

### Missing Dependencies
```bash
pip install torch monai nibabel scipy pandas matplotlib
```

### Validation Failed
```bash
# Make sure you're in evaluation_training directory
cd evaluation_training

# Test metrics independently
python test_official_metrics.py

# Check specific model
python evaluation/evaluate_models.py \
    --dataset lidc \
    --method baseline \
    --dry-run
```

## 📈 Visualization

Generate comprehensive visualizations:

```bash
# Run from evaluation_training directory
cd evaluation_training

# Generate for all methods
python generate_organized_visualizations.py --model-type both

# Specific model type
python generate_organized_visualizations.py --model-type pretrained

# Output will be in evaluation_training/visualizations/
```

## 🚢 Model Upload to Hugging Face

Upload trained models to Hugging Face Hub:

```bash
# Run from project root (not evaluation_training)
cd ..

# Upload all models
bash upload_models.sh your-username/lefusion-models

# Include trained segmentation models from evaluation_training
bash upload_models.sh your-username/lefusion-models --include-trained

# Test without uploading
bash upload_models.sh your-username/lefusion-models --dry-run
```

## 📝 Citation

If you use this pipeline, please cite:

```bibtex
@article{lefusion2024,
  title={LeFusion: Synthesizing Pathological Medical Images using Controllable Diffusion Models},
  author={...},
  year={2024}
}
```

## 🔗 Resources

- [Original LeFusion Repository](https://github.com/M3DV/LeFusion)
- [Paper](https://arxiv.org/...)
- [Hugging Face Models](https://huggingface.co/...)

## 📊 Key Features

- ✅ **Official Metrics**: Integrated from original repository
- ✅ **Modular Design**: Run phases independently
- ✅ **Resume Support**: Checkpoint-based continuation
- ✅ **Multi-GPU**: Automatic device selection
- ✅ **Validation**: Comprehensive test suite
- ✅ **Visualization**: Generate paper-quality figures
- ✅ **Model Upload**: Direct to Hugging Face Hub
