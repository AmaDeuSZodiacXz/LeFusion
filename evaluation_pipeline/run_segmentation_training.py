import argparse
import subprocess
import os
import shutil
import json

def prepare_combined_dataset(real_data_path, synthetic_data_path, output_path):
    print(f"--- Preparing combined dataset at {output_path} ---")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    shutil.copytree(real_data_path, output_path)
    
    synthetic_images_path = os.path.join(synthetic_data_path, 'imagesTr')
    synthetic_labels_path = os.path.join(synthetic_data_path, 'labelsTr')
    target_images_path = os.path.join(output_path, 'imagesTr')
    target_labels_path = os.path.join(output_path, 'labelsTr')
    
    for filename in os.listdir(synthetic_images_path):
        shutil.copy(os.path.join(synthetic_images_path, filename), target_images_path)
    for filename in os.listdir(synthetic_labels_path):
        shutil.copy(os.path.join(synthetic_labels_path, filename), target_labels_path)
        
    dataset_json_path = os.path.join(output_path, 'dataset.json')
    with open(dataset_json_path, 'r+') as f:
        data = json.load(f)
        num_training_new = len(os.listdir(target_images_path))
        data['numTraining'] = num_training_new
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
    print(f"Combined dataset ready. Total training images: {num_training_new}")

def main():
    parser = argparse.ArgumentParser(description="Wrapper to train a segmentation model.")
    parser.add_argument('--real_data_dir', type=str, required=True)
    parser.add_argument('--synthetic_data_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='SwinUNETR')
    parser.add_argument('--output_model_dir', type=str, required=True)
    args = parser.parse_args()

    combined_data_path = os.path.join(os.getcwd(), 'temp_combined_dataset')
    training_data_path = combined_data_path if args.synthetic_data_dir else args.real_data_dir
    if args.synthetic_data_dir:
        prepare_combined_dataset(args.real_data_dir, args.synthetic_data_dir, combined_data_path)

    os.makedirs(args.output_model_dir, exist_ok=True)
    
    main_script_path = 'DiffTumor/STEP3.SegmentationModel/main.py'
    
    # --- THESE ARE THE CORRECTED ARGUMENTS ---
    training_command = [
        'python',
        main_script_path,
        '--data_root', training_data_path,      # Changed from --data_root_path
        '--logdir', args.output_model_dir,      # Changed from --checkpoints_path
        '--model_name', args.model_name,
        '--max_epochs', '200',                  # Changed from --num_epochs
        '--save_checkpoint'                     # Added flag to ensure model is saved
    ]
    
    print("\n--- Starting segmentation training ---")
    print(f"Command: {' '.join(training_command)}")
    
    try:
        subprocess.run(training_command, check=True)
        print("--- Training completed successfully! ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Training failed with error: {e} ---")
    finally:
        if os.path.exists(combined_data_path):
            shutil.rmtree(combined_data_path)

if __name__ == '__main__':
    main()