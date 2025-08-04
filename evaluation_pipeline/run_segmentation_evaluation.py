import argparse
import subprocess
import os
import nibabel as nib
import numpy as np
import pandas as pd
import sys

# Add the surface_distance library to path
sys.path.append('DiffTumor/STEP3.SegmentationModel/external/surface-distance')
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

def calculate_dice(pred_mask, gt_mask):
    """
    Calculate Dice coefficient exactly as in the paper
    """
    # Ensure binary masks
    pred_mask = (pred_mask > 0).astype(bool)
    gt_mask = (gt_mask > 0).astype(bool)
    
    volume_sum = gt_mask.sum() + pred_mask.sum()
    if volume_sum == 0:
        return 1.0 if gt_mask.sum() == 0 and pred_mask.sum() == 0 else 0.0
    
    volume_intersect = (gt_mask & pred_mask).sum()
    dice = 2 * volume_intersect / volume_sum
    return dice

def calculate_nsd_paper(pred_mask, gt_mask, spacing_mm=(1, 1, 1), tolerance=2):
    """
    Calculate NSD using the exact implementation from the paper
    """
    # Ensure binary masks
    pred_mask = (pred_mask > 0).astype(bool)
    gt_mask = (gt_mask > 0).astype(bool)
    
    # Calculate NSD using surface_distance library
    try:
        surface_distances = compute_surface_distances(gt_mask, pred_mask, spacing_mm=spacing_mm)
        nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance)
    except Exception as e:
        print(f"Warning: NSD calculation failed: {e}")
        nsd = 0.0
    
    return nsd

def calculate_metrics(pred_dir, gt_dir):
    """
    Calculate DICE and NSD metrics for all cases using paper implementation
    """
    case_ids = [f.replace('.nii.gz', '') for f in os.listdir(gt_dir) if f.endswith('.nii.gz')]
    
    dice_scores = []
    nsd_scores = []
    
    for case_id in case_ids:
        gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
        pred_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
        
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction not found for {case_id}, skipping.")
            continue
            
        # Load data
        gt_data = nib.load(gt_path).get_fdata()
        pred_data = nib.load(pred_path).get_fdata()
        
        # Calculate metrics using paper implementation
        dice = calculate_dice(pred_data, gt_data)
        nsd = calculate_nsd_paper(pred_data, gt_data)
        
        dice_scores.append(dice)
        nsd_scores.append(nsd)
        
        print(f"Case {case_id}: DICE={dice:.4f}, NSD={nsd:.4f}")
    
    # Calculate mean metrics
    mean_dice = np.mean(dice_scores) * 100  # Convert to percentage
    mean_nsd = np.mean(nsd_scores) * 100    # Convert to percentage
    
    # Calculate standard deviation
    std_dice = np.std(dice_scores) * 100
    std_nsd = np.std(nsd_scores) * 100
    
    print(f"\nSummary Statistics:")
    print(f"Mean DICE: {mean_dice:.2f}% ± {std_dice:.2f}%")
    print(f"Mean NSD: {mean_nsd:.2f}% ± {std_nsd:.2f}%")
    print(f"Number of cases: {len(dice_scores)}")
    
    return mean_dice, mean_nsd, std_dice, std_nsd

def main():
    parser = argparse.ArgumentParser(description="Wrapper to evaluate a segmentation model with DICE and NSD metrics (paper implementation).")
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
    
    print("\n--- Calculating DICE and NSD metrics (paper implementation) ---")
    avg_dice, avg_nsd, std_dice, std_nsd = calculate_metrics(args.output_pred_dir, args.gt_dir)
    
    # Save results with standard deviations
    results_df = pd.DataFrame([{
        'Experiment': args.experiment_name, 
        'Model': args.model_name, 
        'DICE_Mean': avg_dice, 
        'DICE_Std': std_dice,
        'NSD_Mean': avg_nsd, 
        'NSD_Std': std_nsd
    }])
    
    if os.path.exists(args.results_csv):
        results_df.to_csv(args.results_csv, mode='a', header=False, index=False)
    else:
        results_df.to_csv(args.results_csv, mode='w', header=True, index=False)
        
    print(f"--- Results saved to {args.results_csv} ---")

if __name__ == '__main__':
    main()