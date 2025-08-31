# Utility Training Resources

This directory contains essential shared resources and utilities needed by the evaluation_training pipeline.

## Purpose

This folder was restructured from the original `evaluation_pipeline` to keep only the essential components that are actively used by the new `evaluation_training` pipeline. All unnecessary files have been removed to maintain a clean project structure.

## Contents

### `/datasets/`
Contains real dataset splits and configurations needed for training:
- **LIDC_real/**: Real lung nodule data splits for LIDC-IDRI dataset
  - `real_lung_train_0.txt`: Training split file list
  - `real_lung_val_0.txt`: Validation split file list
  - `imagesTr/`: Training images
  - `labelsTr/`: Training labels
- **EMIDEC_real/**: Real cardiac lesion data splits for EMIDEC dataset
  - Similar structure to LIDC_real

### `/DiffTumor/`
Contains the DiffTumor model components, specifically:
- **STEP3.SegmentationModel/**: Segmentation model implementation
  - Core training scripts (main.py)
  - Network architectures (nnU-Net, SwinUNETR)
  - External dependencies (surface-distance metrics)
  - Model utilities and callbacks

## Usage

These resources are referenced by the main evaluation pipeline at `../evaluation_training/`. The paths are configured in:
- `../evaluation_training/configs/experiment_config.yaml`

## Important Notes

1. **Do not modify** the structure of this directory without updating the corresponding paths in `evaluation_training`
2. The dataset split files are essential for reproducible training
3. The DiffTumor components provide the segmentation model implementations used in training

## Related Directories

- `../evaluation_training/`: Main evaluation and training pipeline
- `../data/`: Raw medical image datasets (LIDC, EMIDEC)
- `../LeFusion/`: LeFusion model implementations
- `../DiffMask/`: DiffMask model for mask generation