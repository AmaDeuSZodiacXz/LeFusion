# LeFusion Evaluation Training Pipeline

A comprehensive pipeline for reproducing evaluation results from the LeFusion paper: **"Synthesizing Pathological Medical Images using Controllable Diffusion Models"**

## 🏗️ Project Structure

```
LeFusion/                               # Project root
├── evaluation_training/                # Main evaluation pipeline (this directory)
│   ├── configs/                       # Configuration files
│   ├── evaluation_metrics/            # Official metrics implementation
│   ├── synthetic_generation/          # Synthetic data generation scripts
│   ├── training/                      # Segmentation model training
│   ├── evaluation/                    # Model evaluation scripts
│   ├── utils/                         # Utility functions
│   ├── synthetic_data/                # Generated synthetic datasets
│   ├── trained_models/                # Trained segmentation models
│   └── evaluation_results/            # Evaluation outputs
├── utility_training_resources/        # Shared training resources
│   ├── datasets/                      # Dataset splits (LIDC_real, EMIDEC_real)
│   └── DiffTumor/                     # DiffTumor segmentation framework
├── LeFusion/                          # LeFusion diffusion models
│   └── LeFusion_Model/                # Pretrained and from-scratch models
├── DiffMask/                          # DiffMask mask generation
│   └── DiffMask_Model/                # DiffMask model weights
└── data/                              # Raw datasets
    ├── LIDC/                          # LIDC-IDRI dataset
    └── EMIDEC/                        # EMIDEC cardiac dataset
```

## 📋 Overview

This pipeline reproduces the evaluation metrics from the LeFusion paper, demonstrating improved segmentation performance through synthetic data augmentation:
- **Table 1**: Lung Nodule Segmentation (LIDC-IDRI) - DICE and NSD metrics
- **Table 2**: Cardiac Lesion Segmentation (EMIDEC) - MI Dice and PMO Dice metrics

## 🚀 Quick Start

### Option 1: Using Pre-generated Synthetic Datasets (Recommended)

Download and use pre-generated synthetic datasets directly without running generation:

```bash
# 1. Download pre-generated synthetic data from Hugging Face
# Visit: https://huggingface.co/datasets/Pakawat-Phasook/synthetic_data
# Download the dataset archive and extract to evaluation_training/synthetic_data/

# 2. Navigate to evaluation_training directory
cd evaluation_training

# 3. Extract the downloaded data (if downloaded as archive)
tar -xzf synthetic_data.tar.gz  # or unzip synthetic_data.zip

# 4. Verify the structure
ls synthetic_data/
# Should show: lidc/ and emidec/ directories with pretrained/from_scratch subdirectories

# 5. Skip directly to training with pre-generated data
python run_complete_evaluation.py \
    --dataset lidc \
    --model-types pretrained \
    --skip-synthetic  # Skip generation since data already exists

# Or train specific methods
python training/train_segmentation.py \
    --dataset lidc \
    --methods lefusion_h_diffmask \
    --model-types pretrained \
    --seg-models nnunet
```

### Option 2: Generate Synthetic Data Yourself

```bash
# Prerequisites
cd evaluation_training
pip install -r requirements.txt

# Generate synthetic data
python synthetic_generation/generate_synthetic_data.py \
    --dataset lidc \
    --model-type pretrained \
    --methods all \
    --resume

# Run complete pipeline
python run_complete_evaluation.py \
    --dataset lidc \
    --model-types pretrained \
    --methods all
```

## 📊 Methods & Results

### Methods Overview

| Method | Description | Performance Gain |
|--------|-------------|------------------|
| **Baseline** | Real pathological only | Reference |
| **LeFusion** | Base diffusion model | +0.5% DICE |
| **LeFusion-H** | Histogram-conditioned | +2.3% DICE |
| **LeFusion-H-DiffMask** | Enhanced with mask generation | **+5.2% DICE** |

### Expected Results

#### LIDC-IDRI (Lung Nodules)
| Method | nnU-Net DICE↑ | nnU-Net NSD↑ | SwinUNETR DICE↑ | SwinUNETR NSD↑ |
|--------|--------------|--------------|-----------------|----------------|
| Baseline | 78.26 | 88.90 | 78.38 | 88.67 |
| LeFusion | 78.77 | 89.25 | 78.43 | 88.54 |
| LeFusion-H | 80.62 | 90.90 | 80.95 | 90.98 |
| **LeFusion-H-DiffMask** | **83.44** | **93.35** | **83.13** | **93.20** |

#### EMIDEC (Cardiac Lesions)
| Method | nnU-Net MI↑ | nnU-Net PMO↑ | SwinUNETR MI↑ | SwinUNETR PMO↑ |
|--------|-------------|--------------|---------------|----------------|
| Baseline | 68.61 | 36.32 | 57.79 | 35.76 |
| LeFusion | 69.88 | 34.79 | 57.85 | 35.63 |
| LeFusion-H | 69.95 | 38.01 | 59.61 | 37.99 |
| **LeFusion-H-DiffMask** | **71.28** | **43.41** | **59.30** | **42.49** |

**Metrics Explained:**
- **MI Dice**: Myocardial Infarction segmentation accuracy
- **PMO Dice**: Persistent Microvascular Obstruction segmentation accuracy

## 🔧 Detailed Usage

### Phase 1: Synthetic Data (Skip if using pre-generated)

```bash
# Generate all methods
python synthetic_generation/generate_synthetic_data.py \
    --dataset lidc \
    --model-type pretrained \
    --methods all \
    --resume
```

### Phase 2: Training Segmentation Models

```bash
# Train all models
python training/train_segmentation.py \
    --dataset lidc \
    --methods all \
    --model-types pretrained \
    --seg-models nnunet swinunetr

# Train specific configuration
python training/train_segmentation.py \
    --dataset lidc \
    --methods lefusion_h_diffmask \
    --model-types pretrained \
    --seg-models nnunet
```

### Phase 3: Evaluation

```bash
# Evaluate all models
python evaluation/evaluate_models.py \
    --dataset lidc \
    --methods all \
    --model-types all \
    --seg-models nnunet swinunetr \
    --compare-paper

# Evaluate with specific checkpoint
python evaluation/evaluate_models.py \
    --dataset lidc \
    --methods all \
    --use-best-checkpoint  # Use best validation checkpoint

# Evaluate at specific epoch
python evaluation/evaluate_models.py \
    --dataset lidc \
    --methods all \
    --checkpoint-epoch 150
```

## 📦 Pre-generated Dataset Structure

When using the Hugging Face dataset:

```
synthetic_data/
├── lidc/
│   ├── pretrained/
│   │   ├── lefusion/
│   │   │   └── P_P_prime/
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
    └── [similar structure]
```

## ⚠️ EMIDEC DiffMask Limitations

The pretrained DiffMask model only works for LIDC dataset due to dimension differences:
- **LIDC**: 64×64×32
- **EMIDEC**: 72×72×10

For EMIDEC with DiffMask, you must train a separate model:
```bash
# From project root
bash diffmask_emidec_train.sh
```

## 📊 Checkpoint Selection

Select specific checkpoints for evaluation:

```bash
# Use best checkpoint
python evaluation/evaluate_models.py --use-best-checkpoint

# Use specific epoch
python evaluation/evaluate_models.py --checkpoint-epoch 100

# Compare multiple epochs
for epoch in 50 100 150 200; do
    python evaluation/evaluate_models.py --checkpoint-epoch $epoch
done
```

## 🔗 Resources

- [Original LeFusion Repository](https://github.com/M3DV/LeFusion)
- [Pre-generated Synthetic Data](https://huggingface.co/datasets/Pakawat-Phasook/synthetic_data)
- [Paper](https://arxiv.org/...)

## 📝 Citation

```bibtex
@article{lefusion2024,
  title={LeFusion: Synthesizing Pathological Medical Images using Controllable Diffusion Models},
  author={...},
  year={2024}
}
```