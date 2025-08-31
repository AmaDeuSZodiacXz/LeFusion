#!/usr/bin/env python3
"""
Compare official metrics with existing implementations
This utility helps validate that our integrated metrics match the official repository
"""

import os
import sys
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
import argparse
from tabulate import tabulate

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import official metrics
from evaluation_metrics import dice as official_dice, compute_dice, nsd as official_nsd

# Import our evaluation pipeline
from evaluation.evaluate_models import ModelEvaluator


def create_test_volumes():
    """Create synthetic test volumes for comparison"""
    # Create simple test cases
    test_cases = []
    
    # Case 1: Identical volumes (perfect match)
    vol1 = np.zeros((64, 64, 64))
    vol1[20:40, 20:40, 20:40] = 1
    test_cases.append(("Perfect match", vol1.copy(), vol1.copy()))
    
    # Case 2: No overlap
    vol2_pred = np.zeros((64, 64, 64))
    vol2_gt = np.zeros((64, 64, 64))
    vol2_pred[10:30, 10:30, 10:30] = 1
    vol2_gt[35:55, 35:55, 35:55] = 1
    test_cases.append(("No overlap", vol2_pred, vol2_gt))
    
    # Case 3: Partial overlap
    vol3_pred = np.zeros((64, 64, 64))
    vol3_gt = np.zeros((64, 64, 64))
    vol3_pred[20:40, 20:40, 20:40] = 1
    vol3_gt[30:50, 30:50, 30:50] = 1
    test_cases.append(("Partial overlap", vol3_pred, vol3_gt))
    
    # Case 4: Both empty (edge case)
    vol4 = np.zeros((64, 64, 64))
    test_cases.append(("Both empty", vol4.copy(), vol4.copy()))
    
    # Case 5: One empty (edge case)
    vol5_pred = np.zeros((64, 64, 64))
    vol5_gt = np.zeros((64, 64, 64))
    vol5_gt[20:30, 20:30, 20:30] = 1
    test_cases.append(("Prediction empty", vol5_pred, vol5_gt))
    
    return test_cases


def save_test_volume(volume, path):
    """Save volume as NIfTI file"""
    nii = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nii, path)


def compare_dice_implementations(pred_vol, gt_vol, temp_dir):
    """Compare DICE calculations between implementations"""
    results = {}
    
    # Save volumes as NIfTI files for file-based metrics
    pred_path = temp_dir / "pred.nii.gz"
    gt_path = temp_dir / "gt.nii.gz"
    save_test_volume(pred_vol, pred_path)
    save_test_volume(gt_vol, gt_path)
    
    # 1. Official file-based DICE
    try:
        dice_file = official_dice(str(pred_path), str(gt_path))
        results['official_file'] = dice_file * 100
    except Exception as e:
        results['official_file'] = f"Error: {e}"
    
    # 2. Official compute_dice with tensors
    try:
        pred_tensor = torch.tensor(pred_vol).long()
        gt_tensor = torch.tensor(gt_vol).long()
        dice_compute = compute_dice(pred_tensor, gt_tensor)
        results['official_compute'] = dice_compute * 100
    except Exception as e:
        results['official_compute'] = f"Error: {e}"
    
    # 3. Our integrated implementation
    try:
        evaluator = ModelEvaluator(log_to_file=False)
        dice_ours = evaluator.calculate_dice(pred_vol, gt_vol, use_official=True)
        results['ours_official'] = dice_ours
    except Exception as e:
        results['ours_official'] = f"Error: {e}"
    
    # 4. Our fallback implementation
    try:
        evaluator = ModelEvaluator(log_to_file=False)
        dice_fallback = evaluator.calculate_dice(pred_vol, gt_vol, use_official=False)
        results['ours_fallback'] = dice_fallback
    except Exception as e:
        results['ours_fallback'] = f"Error: {e}"
    
    return results


def compare_nsd_implementations(pred_vol, gt_vol, temp_dir, tolerance=1.0):
    """Compare NSD calculations between implementations"""
    results = {}
    spacing = (1.0, 1.0, 1.0)  # Default spacing
    
    # Save volumes as NIfTI files
    pred_path = temp_dir / "pred.nii.gz"
    gt_path = temp_dir / "gt.nii.gz"
    save_test_volume(pred_vol, pred_path)
    save_test_volume(gt_vol, gt_path)
    
    # 1. Official file-based NSD
    try:
        nsd_file = official_nsd(str(pred_path), str(gt_path), str(gt_path), tolerance=[tolerance])
        if isinstance(nsd_file, torch.Tensor):
            nsd_file = nsd_file.item()
        results['official_file'] = nsd_file * 100
    except Exception as e:
        results['official_file'] = f"Error: {e}"
    
    # 2. Our integrated implementation (official)
    try:
        evaluator = ModelEvaluator(log_to_file=False)
        nsd_ours = evaluator.calculate_nsd(pred_vol, gt_vol, spacing_mm=spacing, 
                                          tolerance=tolerance, use_official=True)
        results['ours_official'] = nsd_ours
    except Exception as e:
        results['ours_official'] = f"Error: {e}"
    
    # 3. Our fallback implementation
    try:
        evaluator = ModelEvaluator(log_to_file=False)
        nsd_fallback = evaluator.calculate_nsd(pred_vol, gt_vol, spacing_mm=spacing, 
                                              tolerance=tolerance, use_official=False)
        results['ours_fallback'] = nsd_fallback
    except Exception as e:
        results['ours_fallback'] = f"Error: {e}"
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare metric implementations")
    parser.add_argument('--tolerance', type=float, default=1.0,
                       help='NSD tolerance in mm (default: 1.0)')
    parser.add_argument('--test-files', nargs=2, metavar=('PRED', 'GT'),
                       help='Optional: Test with specific prediction and ground truth files')
    args = parser.parse_args()
    
    # Create temporary directory for test files
    temp_dir = Path("temp_metric_comparison")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("=" * 80)
    print("METRIC IMPLEMENTATION COMPARISON")
    print("=" * 80)
    print(f"NSD Tolerance: {args.tolerance}mm")
    print()
    
    if args.test_files:
        # Test with provided files
        pred_path, gt_path = args.test_files
        print(f"Testing with files:")
        print(f"  Prediction: {pred_path}")
        print(f"  Ground Truth: {gt_path}")
        print()
        
        # Load volumes
        pred_vol = nib.load(pred_path).get_fdata()
        gt_vol = nib.load(gt_path).get_fdata()
        
        # Compare DICE
        print("DICE Comparison:")
        dice_results = compare_dice_implementations(pred_vol, gt_vol, temp_dir)
        for key, value in dice_results.items():
            print(f"  {key:20s}: {value}")
        
        print()
        print("NSD Comparison:")
        nsd_results = compare_nsd_implementations(pred_vol, gt_vol, temp_dir, args.tolerance)
        for key, value in nsd_results.items():
            print(f"  {key:20s}: {value}")
    else:
        # Test with synthetic cases
        test_cases = create_test_volumes()
        
        # Collect all results
        dice_table = []
        nsd_table = []
        
        for case_name, pred_vol, gt_vol in test_cases:
            print(f"\nTest Case: {case_name}")
            print("-" * 40)
            
            # DICE comparison
            dice_results = compare_dice_implementations(pred_vol, gt_vol, temp_dir)
            dice_row = [case_name] + [f"{v:.2f}" if isinstance(v, (int, float)) else str(v) 
                                     for v in dice_results.values()]
            dice_table.append(dice_row)
            
            # NSD comparison
            nsd_results = compare_nsd_implementations(pred_vol, gt_vol, temp_dir, args.tolerance)
            nsd_row = [case_name] + [f"{v:.2f}" if isinstance(v, (int, float)) else str(v) 
                                    for v in nsd_results.values()]
            nsd_table.append(nsd_row)
        
        # Print summary tables
        print("\n" + "=" * 80)
        print("DICE COMPARISON SUMMARY")
        print("=" * 80)
        headers = ["Test Case", "Official File", "Official Compute", "Ours (Official)", "Ours (Fallback)"]
        print(tabulate(dice_table, headers=headers, tablefmt="grid"))
        
        print("\n" + "=" * 80)
        print(f"NSD COMPARISON SUMMARY (tolerance={args.tolerance}mm)")
        print("=" * 80)
        headers = ["Test Case", "Official File", "Ours (Official)", "Ours (Fallback)"]
        print(tabulate(nsd_table, headers=headers, tablefmt="grid"))
        
        # Check for discrepancies
        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        
        has_discrepancy = False
        for i, (case_name, _, _) in enumerate(test_cases):
            dice_vals = [v for v in dice_table[i][1:] if not isinstance(v, str) and 'Error' not in str(v)]
            nsd_vals = [v for v in nsd_table[i][1:] if not isinstance(v, str) and 'Error' not in str(v)]
            
            if dice_vals:
                dice_vals_float = [float(v) for v in dice_vals]
                dice_diff = max(dice_vals_float) - min(dice_vals_float)
                if dice_diff > 0.01:  # Allow 0.01% difference
                    print(f"⚠️  {case_name}: DICE discrepancy of {dice_diff:.4f}%")
                    has_discrepancy = True
            
            if nsd_vals:
                nsd_vals_float = [float(v) for v in nsd_vals]
                nsd_diff = max(nsd_vals_float) - min(nsd_vals_float)
                if nsd_diff > 0.01:  # Allow 0.01% difference
                    print(f"⚠️  {case_name}: NSD discrepancy of {nsd_diff:.4f}%")
                    has_discrepancy = True
        
        if not has_discrepancy:
            print("✅ All implementations are consistent!")
        else:
            print("\n⚠️  Some discrepancies detected. Please review the results above.")
    
    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 80)
    print("Comparison complete!")


if __name__ == "__main__":
    main()