#!/bin/bash

# DiffMask Resume Training Script - Auto Latest Checkpoint
# This script automatically finds the latest checkpoint and resumes training

# Training parameters
dataset=lidc_mask
root_dir=data/LIDC/Pathological/Image
test_txt_dir=data/LIDC/Pathological/test.txt
diffusion_img_size=64
diffusion_depth_size=32
out_dim=1
unet_num_channels=2
target_steps=80001  # Target total steps (e.g., 80,000)
batch_size=20
results_folder=DiffMask/DiffMask_Model/

# Checkpoint parameters
# Use -1 to automatically load the latest checkpoint
checkpoint_milestone=-1

echo "Starting DiffMask training with automatic checkpoint loading..."
echo "Target steps: $target_steps"
echo "Results folder: $results_folder"
echo "Checkpoint milestone: $checkpoint_milestone (auto-detect latest)"

# Resume training from latest checkpoint
python DiffMask/train/train.py \
    dataset=$dataset \
    dataset.test_txt_dir=$test_txt_dir \
    dataset.root_dir=$root_dir \
    model.diffusion_img_size=$diffusion_img_size \
    model.diffusion_depth_size=$diffusion_depth_size \
    model.train_num_steps=$target_steps \
    model.results_folder=$results_folder \
    model.unet_num_channels=$unet_num_channels \
    model.out_dim=$out_dim \
    model.batch_size=$batch_size \
    model.load_milestone=$checkpoint_milestone 