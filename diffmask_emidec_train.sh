#!/bin/bash

# DiffMask Training Script for EMIDEC Dataset
# This script trains DiffMask from scratch for EMIDEC cardiac data
# Note: EMIDEC has different dimensions than LIDC, requiring separate training

# Training parameters for EMIDEC
dataset=emidec_mask
root_dir=data/EMIDEC/Pathological/Image
test_txt_dir=data/EMIDEC/Pathological/test.txt

# EMIDEC-specific dimensions (adjust based on actual data)
# EMIDEC typically has different image dimensions than LIDC
diffusion_img_size=128  # Adjust based on EMIDEC image size
diffusion_depth_size=16  # Adjust based on EMIDEC depth
out_dim=1
unet_num_channels=2
train_num_steps=80001
batch_size=10  # Smaller batch size due to larger images
results_folder=DiffMask/DiffMask_Model_EMIDEC/

echo "Training DiffMask for EMIDEC dataset from scratch"
echo "Output directory: $results_folder"
echo "Image size: ${diffusion_img_size}x${diffusion_img_size}x${diffusion_depth_size}"

# Create output directory if it doesn't exist
mkdir -p $results_folder

# Train DiffMask for EMIDEC
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
    model.batch_size=$batch_size

echo "Training complete. Model saved to: $results_folder"