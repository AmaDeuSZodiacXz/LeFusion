#!/usr/bin/env python3
"""
Generate test.txt file for EMIDEC dataset
This script creates a test.txt file with 20% of the samples for testing
"""

import os
import glob
import random

def generate_test_txt(data_dir="data/EMIDEC/Pathological", test_ratio=0.2):
    """
    Generate test.txt file for EMIDEC dataset
    
    Args:
        data_dir: Path to EMIDEC Pathological directory
        test_ratio: Ratio of data to use for testing (default 0.2 = 20%)
    """
    
    # Check which directories exist
    image_dirs = []
    if os.path.exists(os.path.join(data_dir, "Image")):
        image_dirs.append(os.path.join(data_dir, "Image"))
        print(f"Found Image directory: {os.path.join(data_dir, 'Image')}")
    if os.path.exists(os.path.join(data_dir, "images")):
        image_dirs.append(os.path.join(data_dir, "images"))
        print(f"Found images directory: {os.path.join(data_dir, 'images')}")
    
    if not image_dirs:
        print(f"❌ Error: No image directory found in {data_dir}")
        print(f"   Expected 'Image/' or 'images/' subdirectory")
        return False
    
    # Use the first found directory
    image_dir = image_dirs[0]
    
    # Get all .nii.gz files
    image_files = glob.glob(os.path.join(image_dir, "*.nii.gz"))
    
    if not image_files:
        print(f"❌ Error: No .nii.gz files found in {image_dir}")
        return False
    
    # Get just the filenames
    filenames = [os.path.basename(f) for f in image_files]
    filenames.sort()  # Sort alphabetically for consistency
    
    print(f"Found {len(filenames)} total files")
    
    # Calculate number of test samples
    num_test = max(1, int(len(filenames) * test_ratio))
    
    # Select last N files for test (consistent approach)
    test_files = filenames[-num_test:]
    
    # Write test.txt
    test_txt_path = os.path.join(data_dir, "test.txt")
    with open(test_txt_path, 'w') as f:
        for filename in test_files:
            f.write(filename + '\n')
    
    print(f"✅ Created {test_txt_path}")
    print(f"   - Total files: {len(filenames)}")
    print(f"   - Test files: {num_test} ({test_ratio*100:.0f}%)")
    print(f"   - Training files: {len(filenames) - num_test}")
    
    # Show sample of test files
    print(f"\nFirst 5 test files:")
    for i, filename in enumerate(test_files[:5]):
        print(f"   {i+1}. {filename}")
    
    return True

def create_directory_structure(data_dir="data/EMIDEC/Pathological"):
    """
    Create Image/ and Mask/ directories if they don't exist
    and copy/link files from images/ and labels/
    """
    
    # Check if source directories exist
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory {data_dir} does not exist")
        print(f"Please ensure EMIDEC dataset is downloaded and extracted")
        return False
    
    # Create target directories
    image_target = os.path.join(data_dir, "Image")
    mask_target = os.path.join(data_dir, "Mask")
    
    os.makedirs(image_target, exist_ok=True)
    os.makedirs(mask_target, exist_ok=True)
    
    print(f"Created directories:")
    print(f"  - {image_target}")
    print(f"  - {mask_target}")
    
    # Copy/link files if source directories exist
    if os.path.exists(images_dir):
        image_files = glob.glob(os.path.join(images_dir, "*.nii.gz"))
        print(f"\nCopying {len(image_files)} files from images/ to Image/")
        for src in image_files:
            dst = os.path.join(image_target, os.path.basename(src))
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)  # Try hard link first
                except:
                    import shutil
                    shutil.copy2(src, dst)  # Fall back to copy
        print(f"✅ Copied image files")
    
    if os.path.exists(labels_dir):
        label_files = glob.glob(os.path.join(labels_dir, "*.nii.gz"))
        print(f"\nCopying {len(label_files)} files from labels/ to Mask/")
        for src in label_files:
            dst = os.path.join(mask_target, os.path.basename(src))
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)  # Try hard link first
                except:
                    import shutil
                    shutil.copy2(src, dst)  # Fall back to copy
        print(f"✅ Copied mask files")
    
    return True

if __name__ == "__main__":
    print("="*50)
    print("EMIDEC Test File Generator")
    print("="*50)
    
    # Try to create directory structure first
    print("\nStep 1: Setting up directory structure...")
    print("-"*40)
    create_directory_structure()
    
    # Generate test.txt
    print("\nStep 2: Generating test.txt file...")
    print("-"*40)
    success = generate_test_txt()
    
    if success:
        print("\n" + "="*50)
        print("✅ SUCCESS: EMIDEC data is ready for DiffMask training!")
        print("="*50)
        print("\nNext steps:")
        print("1. Run training:     bash diffmask_emidec_train.sh")
        print("2. Resume training:  bash diffmask_emidec_resume_train.sh")
    else:
        print("\n" + "="*50)
        print("❌ ERROR: Failed to generate test.txt")
        print("Please check that EMIDEC data is properly downloaded")
        print("="*50)