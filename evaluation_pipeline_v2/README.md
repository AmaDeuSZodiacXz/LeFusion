# LeFusion Paper Evaluation Pipeline v2

A modular, organized pipeline for reproducing the evaluation results from the LeFusion paper.

## 📋 Overview

This pipeline reproduces the exact evaluation tables from the LeFusion paper:
- **Table 1**: Lung Nodule Segmentation (LIDC dataset) - DICE and NSD metrics
- **Table 2**: Cardiac Lesion Segmentation (EMIDEC dataset) - DICE metrics

## 🏗️ Pipeline Structure

```
evaluation_pipeline_v2/
├── configs/
│   └── experiment_config.yaml      # Central configuration
├── synthetic_generation/
│   └── generate_synthetic_data.py  # Synthetic data generation
├── training/
│   └── train_segmentation.py       # Model training
├── evaluation/
│   └── evaluate_models.py          # Model evaluation
├── run_complete_evaluation.py      # Master orchestrator
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Run everything for LIDC dataset with pretrained models
python run_complete_evaluation.py --dataset lidc --model-types pretrained

# Quick test with baseline only
python run_complete_evaluation.py --quick-test

# Run specific methods
python run_complete_evaluation.py --methods lefusion lefusion_h --model-types pretrained
```

## 📊 Methods Supported

According to the paper tables:

| Method | Description | Training Setting |
|--------|-------------|-----------------|
| **Baseline** | Real pathological cases only | P |
| **LeFusion** | Base diffusion model | P+P' |
| **LeFusion-H** | Histogram-conditioned | P+P', P+N' |
| **LeFusion-H+DiffMask** | Enhanced with DiffMask | P+N', P+N'', P+P'+N'' |

Where:
- **P**: Real pathological cases
- **P'**: Synthetic pathological from pathological
- **N'**: Synthetic pathological from normal
- **N''**: Enhanced synthetic data

## 🔧 Modular Usage

### Phase 1: Synthetic Data Generation

```bash
# Generate synthetic data for LIDC with pretrained models
python synthetic_generation/generate_synthetic_data.py \
    --dataset lidc \
    --model-type pretrained \
    --methods lefusion lefusion_h lefusion_h_diffmask

# Resume from checkpoint
python synthetic_generation/generate_synthetic_data.py \
    --dataset lidc \
    --model-type pretrained \
    --resume
```

**Output Structure:**
```
synthetic_data/
├── lidc/
│   ├── pretrained/
│   │   ├── lefusion/
│   │   │   └── P_P_prime/
│   │   │       ├── imagesTr/
│   │   │       └── labelsTr/
│   │   ├── lefusion_h/
│   │   │   ├── P_P_prime/
│   │   │   └── P_N_prime/
│   │   └── lefusion_h_diffmask/
│   │       ├── P_N_prime/
│   │       ├── P_N_double_prime/
│   │       └── P_P_prime_N_double_prime/
│   └── from_scratch/
│       └── [same structure]
└── emidec/
    └── [same structure]
```

### Phase 2: Training Segmentation Models

```bash
# Train all models for LIDC
python training/train_segmentation.py \
    --dataset lidc \
    --methods baseline lefusion lefusion_h lefusion_h_diffmask \
    --model-types pretrained from_scratch \
    --seg-models nnunet swinunetr

# Train specific configuration
python training/train_segmentation.py \
    --dataset lidc \
    --methods lefusion_h \
    --model-types pretrained \
    --seg-models nnunet
```

**Output Structure:**
```
trained_models/
├── lidc/
│   ├── baseline/
│   │   └── pretrained/
│   │       ├── nnunet/
│   │       │   └── best_metric_model.pth
│   │       └── swinunetr/
│   │           └── best_metric_model.pth
│   ├── lefusion/
│   │   ├── pretrained/
│   │   └── from_scratch/
│   └── [other methods]
└── emidec/
    └── [same structure]
```

### Phase 3: Evaluation

```bash
# Evaluate all models and compare with paper
python evaluation/evaluate_models.py \
    --dataset lidc \
    --compare-paper

# Evaluate specific models
python evaluation/evaluate_models.py \
    --dataset lidc \
    --methods lefusion_h_diffmask \
    --model-types pretrained \
    --seg-models nnunet swinunetr
```

**Output:**
- CSV files with detailed metrics
- Paper-formatted tables in console
- Comparison with paper values

## 📈 Expected Results (from Paper)

### LIDC Dataset (Table 1)

| Method | nnU-Net DICE | nnU-Net NSD | SwinUNETR DICE | SwinUNETR NSD |
|--------|--------------|-------------|----------------|---------------|
| Baseline | 78.26 | 88.90 | 78.38 | 88.67 |
| LeFusion | 78.77 | 89.25 | 78.43 | 88.54 |
| LeFusion-H | 80.62 | 90.90 | 80.95 | 90.98 |
| LeFusion-H+DiffMask | **83.44** | **93.35** | **83.13** | **93.20** |

### EMIDEC Dataset (Table 2)

| Method | nnU-Net MI | nnU-Net PMO | SwinUNETR MI | SwinUNETR PMO |
|--------|------------|-------------|--------------|---------------|
| Baseline | 68.61 | 36.32 | 57.79 | 35.76 |
| LeFusion | 69.88 | 34.79 | 57.85 | 35.63 |
| LeFusion-H | 69.95 | 38.01 | 59.61 | 37.99 |
| LeFusion-H+DiffMask | **71.28** | **43.41** | **59.30** | **42.49** |

## 🔄 Resume Capability

All phases support resuming from checkpoints:

```bash
# Resume synthetic generation
python synthetic_generation/generate_synthetic_data.py --resume

# Resume training
python training/train_segmentation.py --resume

# The complete pipeline automatically uses resume
python run_complete_evaluation.py
```

## 📝 Configuration

Edit `configs/experiment_config.yaml` to modify:
- Dataset paths
- Model weights locations
- Training hyperparameters
- Evaluation metrics settings

## 🐛 Troubleshooting

### Missing Model Weights

Ensure these files exist:
- **Pretrained LeFusion**: 
  - `../LeFusion/LeFusion_Model/LIDC/lidc.pt`
  - `../LeFusion/LeFusion_Model/EMIDEC/emidec.pt`
- **Pretrained DiffMask**: 
  - `../DiffMask/DiffMask_Model/diffmask.pt`
- **From Scratch**: 
  - `../LeFusion/LeFusion_Model/LIDC/model-50.pt`
  - `../DiffMask/DiffMask_Model/model-80.pt`

### GPU Memory Issues

Reduce batch size in `configs/experiment_config.yaml`:
```yaml
training:
  batch_size: 1  # Reduce if needed
```

### Missing Dependencies

```bash
pip install torch torchvision monai nibabel scipy pandas pyyaml
```

## 📊 Metrics

- **DICE**: Volumetric overlap coefficient (higher is better)
- **NSD**: Normalized Surface Distance at 2mm tolerance (higher is better)

## 🎯 Tips for Reproduction

1. **Start with Baseline**: Test the pipeline with baseline first
2. **Use Pretrained Models**: Faster than training from scratch
3. **Monitor GPU Memory**: Use `nvidia-smi` to check usage
4. **Check Checkpoints**: Resume capability saves time
5. **Validate Paths**: Ensure all data and model paths are correct

## 📧 Support

For issues or questions about reproducing the paper results, please check:
1. The original LeFusion paper
2. The configuration file paths
3. GPU/CUDA compatibility

---

**Note**: This pipeline is designed to exactly reproduce the evaluation tables from the LeFusion paper. The modular structure allows for easy experimentation with individual components. 