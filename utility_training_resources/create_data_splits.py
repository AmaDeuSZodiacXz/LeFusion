#!/usr/bin/env python3
"""
Create train/val split files for DiffTumor training
"""

import os
import glob
import random

def create_data_splits(data_dir, output_dir, train_ratio=0.8):
    """
    Create train/val split files for DiffTumor training
    """
    print(f"Creating data splits for {data_dir}")
    
    # Get all image files
    images_dir = os.path.join(data_dir, 'imagesTr')
    labels_dir = os.path.join(data_dir, 'labelsTr')
    
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.nii.gz')))
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.nii.gz')))
    
    print(f"Found {len(image_files)} training images")
    
    # Create pairs
    pairs = []
    for img_file, label_file in zip(image_files, label_files):
        # Get relative paths with leading slash to ensure proper path concatenation
        img_rel = '/' + os.path.relpath(img_file, data_dir)
        label_rel = '/' + os.path.relpath(label_file, data_dir)
        pairs.append((img_rel, label_rel))
    
    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write train file
    train_file = os.path.join(output_dir, 'real_liver_train_0.txt')
    with open(train_file, 'w') as f:
        for img_rel, label_rel in train_pairs:
            f.write(f"{img_rel} {label_rel}\n")
    
    # Write val file
    val_file = os.path.join(output_dir, 'real_liver_val_0.txt')
    with open(val_file, 'w') as f:
        for img_rel, label_rel in val_pairs:
            f.write(f"{img_rel} {label_rel}\n")
    
    print(f"✓ Created train file: {train_file}")
    print(f"✓ Created val file: {val_file}")
    
    return train_file, val_file

def main():
    data_dir = "datasets/LIDC_real"
    output_dir = "datasets/LIDC_real"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found!")
        return
    
    create_data_splits(data_dir, output_dir)

if __name__ == "__main__":
    main() 