import argparse
import subprocess
import os
import nibabel as nib
import numpy as np
import pandas as pd
from monai.metrics import DiceMetric, HausdorffDistanceMetric

def calculate_metrics(pred_dir, gt_dir):
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    case_ids = [f.replace('.nii.gz', '') for f in os.listdir(gt_dir) if f.endswith('.nii.gz')]
    
    for case_id in case_ids:
        gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
        pred_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
        
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction not found for {case_id}, skipping.")
            continue
            
        gt_data = np.expand_dims(nib.load(gt_path).get_fdata(), axis=(0, 1))
        gt_data[gt_data > 0] = 1
        
        pred_data = np.expand_dims(nib.load(pred_path).get_fdata(), axis=(0, 1))
        pred_data[pred_data > 0] = 1
        
        dice_metric(y_pred=pred_data, y=gt_data)
        hausdorff_metric(y_pred=pred_data, y=gt_data)
        
    mean_dice = dice_metric.aggregate().item()
    mean_hd95 = hausdorff_metric.aggregate().item()
    return mean_dice * 100, mean_hd95

def main():
    parser = argparse.ArgumentParser(description="Wrapper to evaluate a segmentation model.")
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--trained_model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, choices=['SwinUNETR', 'nnUNet'], default='SwinUNETR')
    parser.add_argument('--output_pred_dir', type=str, required=True)
    parser.add_argument('--results_csv', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_pred_dir, exist_ok=True)
    
    # --- THIS IS THE CORRECTED LINE ---
    main_script_path = 'DiffTumor/STEP3.SegmentationModel/main.py'
    
    inference_command = [
        'python',
        main_script_path,
        '--phase', 'test',
        '--data_root_path', os.path.dirname(args.test_data_dir.rstrip('/')),
        '--model_name', args.model_name,
        '--checkpoints_path', os.path.dirname(args.trained_model_path),
        '--load_checkpoint_path', args.trained_model_path,
        '--output_path', args.output_pred_dir
    ]
    
    print("\n--- Starting segmentation inference ---")
    print(f"Command: {' '.join(inference_command)}")
    subprocess.run(inference_command, check=True)
    print("--- Inference completed. ---")
    
    print("\n--- Calculating metrics ---")
    avg_dice, avg_nsd_proxy = calculate_metrics(args.output_pred_dir, args.gt_dir)
    print(f"Average Dice: {avg_dice:.2f}%")
    print(f"Average 95% HD (NSD Proxy): {avg_nsd_proxy:.2f}")
    
    results_df = pd.DataFrame([{'Experiment': args.experiment_name, 'Model': args.model_name, 'Dice': avg_dice, 'NSD_Proxy_HD95': avg_nsd_proxy}])
    
    if os.path.exists(args.results_csv):
        results_df.to_csv(args.results_csv, mode='a', header=False, index=False)
    else:
        results_df.to_csv(args.results_csv, mode='w', header=True, index=False)
        
    print(f"--- Results saved to {args.results_csv} ---")

if __name__ == '__main__':
    main()