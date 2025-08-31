#!/usr/bin/env python3
"""
Test script to verify the official metrics integration
"""

import os
import sys
import numpy as np
import nibabel as nib
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import the official metrics
from evaluation_metrics import dice, compute_dice, nsd

def test_metrics():
    """Test the official metrics with simple examples"""
    print("=" * 60)
    print("Testing Official Metrics Integration")
    print("=" * 60)
    
    # Create temporary test directory
    test_dir = Path("temp_test")
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Test 1: Perfect match
        print("\nTest 1: Perfect Match")
        print("-" * 30)
        vol1 = np.zeros((64, 64, 64))
        vol1[20:40, 20:40, 20:40] = 1
        
        # Save as NIfTI
        pred_path = test_dir / "test1_pred.nii.gz"
        gt_path = test_dir / "test1_gt.nii.gz"
        
        nii1 = nib.Nifti1Image(vol1, affine=np.eye(4))
        nib.save(nii1, pred_path)
        nib.save(nii1, gt_path)
        
        # Test DICE
        dice_score = dice(str(pred_path), str(gt_path))
        print(f"DICE (file-based): {dice_score:.4f} (expected: ~1.0)")
        
        # Test compute_dice
        vol1_tensor = torch.tensor(vol1).long()
        dice_compute = compute_dice(vol1_tensor, vol1_tensor)
        print(f"DICE (tensor-based): {dice_compute:.4f} (expected: ~1.0)")
        
        # Test NSD
        nsd_score = nsd(str(pred_path), str(gt_path), str(gt_path), tolerance=[1.0])
        if isinstance(nsd_score, torch.Tensor):
            nsd_score = nsd_score.item()
        print(f"NSD (1mm tolerance): {nsd_score:.4f} (expected: ~1.0)")
        
        # Test 2: No overlap
        print("\nTest 2: No Overlap")
        print("-" * 30)
        vol2_pred = np.zeros((64, 64, 64))
        vol2_gt = np.zeros((64, 64, 64))
        vol2_pred[10:20, 10:20, 10:20] = 1
        vol2_gt[40:50, 40:50, 40:50] = 1
        
        pred_path2 = test_dir / "test2_pred.nii.gz"
        gt_path2 = test_dir / "test2_gt.nii.gz"
        
        nib.save(nib.Nifti1Image(vol2_pred, affine=np.eye(4)), pred_path2)
        nib.save(nib.Nifti1Image(vol2_gt, affine=np.eye(4)), gt_path2)
        
        dice_score2 = dice(str(pred_path2), str(gt_path2))
        print(f"DICE: {dice_score2:.4f} (expected: 0.0)")
        
        nsd_score2 = nsd(str(pred_path2), str(gt_path2), str(gt_path2), tolerance=[1.0])
        if isinstance(nsd_score2, torch.Tensor):
            nsd_score2 = nsd_score2.item()
        print(f"NSD: {nsd_score2:.4f} (expected: 0.0)")
        
        # Test 3: Partial overlap
        print("\nTest 3: Partial Overlap")
        print("-" * 30)
        vol3_pred = np.zeros((64, 64, 64))
        vol3_gt = np.zeros((64, 64, 64))
        vol3_pred[20:40, 20:40, 20:40] = 1
        vol3_gt[30:50, 30:50, 30:50] = 1
        
        pred_path3 = test_dir / "test3_pred.nii.gz"
        gt_path3 = test_dir / "test3_gt.nii.gz"
        
        nib.save(nib.Nifti1Image(vol3_pred, affine=np.eye(4)), pred_path3)
        nib.save(nib.Nifti1Image(vol3_gt, affine=np.eye(4)), gt_path3)
        
        dice_score3 = dice(str(pred_path3), str(gt_path3))
        print(f"DICE: {dice_score3:.4f} (expected: ~0.286)")
        
        nsd_score3 = nsd(str(pred_path3), str(gt_path3), str(gt_path3), tolerance=[1.0])
        if isinstance(nsd_score3, torch.Tensor):
            nsd_score3 = nsd_score3.item()
        print(f"NSD: {nsd_score3:.4f}")
        
        # Test edge case handling
        print("\nTest 4: Edge Cases")
        print("-" * 30)
        
        # Both empty
        vol4 = np.zeros((64, 64, 64))
        pred_path4 = test_dir / "test4_pred.nii.gz"
        gt_path4 = test_dir / "test4_gt.nii.gz"
        
        nib.save(nib.Nifti1Image(vol4, affine=np.eye(4)), pred_path4)
        nib.save(nib.Nifti1Image(vol4, affine=np.eye(4)), gt_path4)
        
        dice_score4 = dice(str(pred_path4), str(gt_path4))
        print(f"DICE (both empty): {dice_score4:.4f} (expected: 1.0)")
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_evaluation_pipeline():
    """Test the full evaluation pipeline with official metrics"""
    print("\n" + "=" * 60)
    print("Testing Evaluation Pipeline Integration")
    print("=" * 60)
    
    from evaluation.evaluate_models import ModelEvaluator
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(log_to_file=False)
        print("✅ Evaluator initialized successfully")
        
        # Test metric calculations with dummy data
        pred = np.zeros((64, 64, 64))
        gt = np.zeros((64, 64, 64))
        pred[20:40, 20:40, 20:40] = 1
        gt[25:45, 25:45, 25:45] = 1
        
        # Test DICE calculation
        dice_official = evaluator.calculate_dice(pred, gt, use_official=True)
        dice_fallback = evaluator.calculate_dice(pred, gt, use_official=False)
        
        print(f"\nDICE Scores:")
        print(f"  Official: {dice_official:.2f}%")
        print(f"  Fallback: {dice_fallback:.2f}%")
        
        # Test NSD calculation
        nsd_official = evaluator.calculate_nsd(pred, gt, tolerance=1.0, use_official=True)
        nsd_fallback = evaluator.calculate_nsd(pred, gt, tolerance=1.0, use_official=False)
        
        print(f"\nNSD Scores (1mm tolerance):")
        print(f"  Official: {nsd_official:.2f}%")
        print(f"  Fallback: {nsd_fallback:.2f}%")
        
        print("\n✅ Evaluation pipeline integration successful!")
        
    except Exception as e:
        print(f"\n❌ Error in evaluation pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting Official Metrics Integration Tests\n")
    
    # Test the metrics directly
    test_metrics()
    
    # Test the evaluation pipeline integration
    test_evaluation_pipeline()
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)