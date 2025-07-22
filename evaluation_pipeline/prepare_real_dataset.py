import os
import shutil
import json
import argparse
from tqdm import tqdm
import glob

def prepare_dataset(source_image_dir, source_mask_dir, test_txt_path, output_dir):
    """
    Organizes the raw LIDC dataset into the nnU-Net format for training and testing.
    This version recursively searches for .nii.gz files.
    """
    print("--- Starting Dataset Preparation (Corrected Version) ---")

    # 1. Define and create the output directory structure
    dirs_to_create = {
        'imagesTr': os.path.join(output_dir, 'imagesTr'),
        'labelsTr': os.path.join(output_dir, 'labelsTr'),
        'imagesTs': os.path.join(output_dir, 'imagesTs'),
        'labelsTs': os.path.join(output_dir, 'labelsTs')
    }
    for path in dirs_to_create.values():
        os.makedirs(path, exist_ok=True)
    print(f"Created directory structure inside: {output_dir}")

    # 2. Load the list of test case identifiers
    with open(test_txt_path, 'r') as f:
        test_cases = {line.strip() for line in f}
    print(f"Loaded {len(test_cases)} test case identifiers from {test_txt_path}")

    # 3. Process and copy files
    search_pattern = os.path.join(source_image_dir, '**', '*.nii.gz')
    all_image_paths = sorted(glob.glob(search_pattern, recursive=True))
    
    if not all_image_paths:
        print(f"Error: No .nii.gz files found in {source_image_dir}. Please check the path.")
        return
        
    print(f"Found {len(all_image_paths)} total .nii.gz files to process.")
    
    for source_image_path in tqdm(all_image_paths, desc="Processing cases"):
        image_filename = os.path.basename(source_image_path)
        case_id = image_filename.replace('.nii.gz', '')
        
        mask_filename = image_filename.replace('_Vol_', '_Mask_')
        source_mask_path = source_image_path.replace(source_image_dir, source_mask_dir).replace(image_filename, mask_filename)
        
        if not os.path.exists(source_mask_path):
            print(f"Warning: Mask not found for {image_filename}, skipping.")
            continue

        if case_id in test_cases:
            dest_img_dir = dirs_to_create['imagesTs']
            dest_lbl_dir = dirs_to_create['labelsTs']
        else:
            dest_img_dir = dirs_to_create['imagesTr']
            dest_lbl_dir = dirs_to_create['labelsTr']
            
        new_image_filename = f"{case_id}_0000.nii.gz"
        new_mask_filename = f"{case_id}.nii.gz"
        
        shutil.copy(source_image_path, os.path.join(dest_img_dir, new_image_filename))
        shutil.copy(source_mask_path, os.path.join(dest_lbl_dir, new_mask_filename))

    # 4. Generate the dataset.json file
    num_training = len(os.listdir(dirs_to_create['imagesTr']))
    num_testing = len(os.listdir(dirs_to_create['imagesTs']))
    
    dataset_info = {
        "name": "LIDC_Nodules",
        "description": "Lung nodules from LIDC-IDRI dataset",
        "numTraining": num_training,
        "numTest": num_testing,
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "nodule"}
    }
    
    json_path = os.path.join(output_dir, 'dataset.json')
    with open(json_path, 'w') as f:
        json.dump(dataset_info, f, indent=4)
    print(f"Generated dataset.json with {num_training} training samples and {num_testing} test samples.")
    
    print("--- Dataset Preparation Complete! ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare LIDC dataset for segmentation.")
    parser.add_argument('--source_image_dir', type=str, required=True)
    parser.add_argument('--source_mask_dir', type=str, required=True)
    parser.add_argument('--test_txt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    prepare_dataset(args.source_image_dir, args.source_mask_dir, args.test_txt_path, args.output_dir)