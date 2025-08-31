#!/bin/bash

# DiffMask Resume Training Script for EMIDEC Dataset
# This script allows you to resume training from a specific checkpoint for EMIDEC

# Training parameters for EMIDEC
dataset=emidec_mask
root_dir=data/EMIDEC/Pathological  # Root directory containing Image/ and Mask/
test_txt_dir=data/EMIDEC/Pathological/test.txt

# EMIDEC-specific dimensions (matching inference configuration)
diffusion_img_size=72   # EMIDEC uses 72x72 images
diffusion_depth_size=10  # EMIDEC uses depth of 10
out_dim=1
unet_num_channels=2
train_num_steps=80001  # Target total steps
batch_size=10  # Smaller batch size due to larger images
results_folder=DiffMask/DiffMask_Model_EMIDEC/

# Checkpoint parameters
# Set this to the path of your checkpoint file (e.g., "DiffMask/DiffMask_Model_EMIDEC/model-15.pt" for step 15,000)
checkpoint_path="DiffMask/DiffMask_Model_EMIDEC/model-40.pt"  # Adjust to your checkpoint

echo "Resuming DiffMask training for EMIDEC dataset"
echo "Loading checkpoint from: $checkpoint_path"
echo "Output directory: $results_folder"

# Resume training from checkpoint
python DiffMask/train/train.py \
    dataset=$dataset \
    dataset.test_txt_dir=$test_txt_dir \
    dataset.root_dir=$root_dir \
    model.diffusion_img_size=$diffusion_img_size \
    model.diffusion_depth_size=$diffusion_depth_size \
    model.train_num_steps=$train_num_steps \
    model.results_folder=$results_folder \
    model.unet_num_channels=$unet_num_channels \
    model.out_dim=$out_dim \
    model.batch_size=$batch_size \
    model.load_milestone=$checkpoint_path

echo "Training resumed and complete. Model saved to: $results_folder"