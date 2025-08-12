#!/usr/bin/env python3
"""
Evaluation Pipeline for Segmentation Models
Computes DICE and NSD metrics as shown in the paper tables
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import nibabel as nib
from scipy.ndimage import distance_transform_edt
import torch
from monai.metrics import DiceMetric

# Add path for surface_distance library
sys.path.append('../../evaluation_pipeline/DiffTumor/STEP3.SegmentationModel/external/surface-distance')
try:
    from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
except ImportError:
    print("Warning: surface_distance library not found. NSD metrics may not work.")

class ModelEvaluator:
    def __init__(self, config_path="../configs/experiment_config.yaml"):
        """Initialize model evaluator with config"""
        # Get the evaluation_pipeline_v2 directory as base
        script_dir = Path(__file__).parent  # evaluation/
        self.base_dir = script_dir.parent   # evaluation_pipeline_v2/
        
        # Load config with proper path resolution
        config_full_path = self.base_dir / "configs" / "experiment_config.yaml"
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set directories relative to evaluation_pipeline_v2
        self.results_dir = self.base_dir / "evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📁 Results directory: {self.results_dir}")
        
    def calculate_dice(self, pred, gt):
        """Calculate DICE coefficient"""
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        
        if pred.sum() == 0 and gt.sum() == 0:
            return 1.0
        elif pred.sum() == 0 or gt.sum() == 0:
            return 0.0
            
        intersection = np.logical_and(pred, gt).sum()
        dice = 2.0 * intersection / (pred.sum() + gt.sum())
        return dice * 100  # Return as percentage
        
    def calculate_nsd(self, pred, gt, spacing_mm=(1, 1, 1), tolerance=2):
        """Calculate Normalized Surface Distance (NSD) at tolerance"""
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        
        # Handle edge cases
        if pred.sum() == 0 and gt.sum() == 0:
            return 100.0
        elif pred.sum() == 0 or gt.sum() == 0:
            return 0.0
            
        try:
            surface_distances = compute_surface_distances(gt, pred, spacing_mm=spacing_mm)
            nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance)
            return nsd * 100  # Return as percentage
        except Exception as e:
            print(f"Warning: NSD calculation failed: {e}")
            return 0.0
            
    def load_nifti(self, path):
        """Load NIfTI file"""
        nii = nib.load(path)
        return nii.get_fdata(), nii.header.get_zooms()[:3]
        
    def evaluate_single_case(self, pred_path, gt_path):
        """Evaluate a single prediction against ground truth"""
        # Load predictions and ground truth
        pred, spacing = self.load_nifti(pred_path)
        gt, _ = self.load_nifti(gt_path)
        
        # Calculate metrics
        dice = self.calculate_dice(pred, gt)
        nsd = self.calculate_nsd(pred, gt, spacing_mm=spacing, tolerance=self.config['evaluation']['nsd_tolerance'])
        
        return {
            'dice': dice,
            'nsd': nsd
        }
        
    def evaluate_model(self, dataset, method, model_type, seg_model):
        """Evaluate a trained model on test set"""
        print(f"\n📊 Evaluating {seg_model}")
        print(f"   Dataset: {dataset}")
        print(f"   Method: {method}")
        print(f"   Model Type: {model_type}")
        
        # Get model path
        model_dir = self.base_dir / "trained_models" / dataset / method / model_type / seg_model
        model_path = model_dir / "best_metric_model.pth"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return None
            
        # Get test data
        test_data_dir = Path(self.config['datasets'][dataset]['real_data_dir'])
        test_images_dir = test_data_dir / "imagesTs"
        test_labels_dir = test_data_dir / "labelsTs"
        
        if not test_images_dir.exists():
            print(f"❌ Test images not found: {test_images_dir}")
            return None
            
        # Run inference (simplified - in practice you'd load the model and run predictions)
        predictions_dir = self.results_dir / f"{dataset}_{method}_{model_type}_{seg_model}" / "predictions"
        os.makedirs(predictions_dir, exist_ok=True)
        
        # For now, we'll assume predictions are already generated
        # In practice, you'd run the model inference here
        
        # Collect all test cases
        test_cases = list(test_labels_dir.glob("*.nii.gz"))
        
        if len(test_cases) == 0:
            print(f"❌ No test cases found in {test_labels_dir}")
            return None
            
        # Evaluate each case
        results = []
        for gt_path in test_cases:
            case_name = gt_path.stem.replace(".nii", "")
            pred_path = predictions_dir / gt_path.name
            
            if pred_path.exists():
                metrics = self.evaluate_single_case(pred_path, gt_path)
                metrics['case'] = case_name
                results.append(metrics)
            else:
                print(f"⚠️ Prediction not found for {case_name}")
                
        if len(results) == 0:
            print(f"❌ No predictions found to evaluate")
            return None
            
        # Calculate statistics
        df = pd.DataFrame(results)
        
        stats = {
            'dataset': dataset,
            'method': method,
            'model_type': model_type,
            'seg_model': seg_model,
            'dice_mean': df['dice'].mean(),
            'dice_std': df['dice'].std(),
            'nsd_mean': df['nsd'].mean(),
            'nsd_std': df['nsd'].std(),
            'num_cases': len(df)
        }
        
        return stats
        
    def generate_paper_table(self, results, dataset="lidc"):
        """Generate evaluation table in paper format"""
        print(f"\n{'='*80}")
        
        if dataset == "lidc":
            print("Table 1: Downstream Lung Nodule Segmentation Dice (↑) and NSD (↑) on LIDC")
        else:
            print("Table 2: Downstream Cardiac Lesion Segmentation Dice (↑) on EMIDEC")
            
        print("P: real pathological cases. P'/N': synthetic pathological cases")
        print("Bold numbers indicate the best performance in each setting")
        print("="*80)
        
        # Create formatted table
        print(f"\n{'Methods':<30} {'Training':<15} {'nnU-Net (2021)':<30} {'SwinUNETR (2021)':<30}")
        print(f"{'':30} {'Setting':<15} {'Dice (↑)  NSD (↑)':<30} {'Dice (↑)  NSD (↑)':<30}")
        print("-"*105)
        
        # Group results by method
        for method in ['baseline', 'lefusion', 'lefusion_h', 'lefusion_h_diffmask']:
            method_results = [r for r in results if r['method'] == method]
            
            if len(method_results) == 0:
                continue
                
            # Format method name
            if method == 'baseline':
                method_name = "Baseline"
                training_setting = "P"
            elif method == 'lefusion':
                method_name = "LeFusion (Ours)"
                training_setting = "P+P'"
            elif method == 'lefusion_h':
                method_name = "LeFusion-H (Ours)"
                training_setting = "P+P'"
            else:
                method_name = "LeFusion-H+DiffMask (Ours)"
                training_setting = "P+N'"
                
            # Get results for each segmentation model
            nnunet_result = next((r for r in method_results if r['seg_model'] == 'nnunet'), None)
            swin_result = next((r for r in method_results if r['seg_model'] == 'swinunetr'), None)
            
            # Format metrics
            if nnunet_result:
                nnunet_str = f"{nnunet_result['dice_mean']:.2f}    {nnunet_result['nsd_mean']:.2f}"
            else:
                nnunet_str = "-         -"
                
            if swin_result:
                swin_str = f"{swin_result['dice_mean']:.2f}    {swin_result['nsd_mean']:.2f}"
            else:
                swin_str = "-         -"
                
            print(f"{method_name:<30} {training_setting:<15} {nnunet_str:<30} {swin_str:<30}")
            
        print("-"*105)
        
    def evaluate_all(self, dataset="lidc", methods=None, model_types=None, seg_models=None):
        """Evaluate all specified configurations"""
        
        # Default values
        if methods is None:
            methods = ["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"]
        if model_types is None:
            model_types = ["pretrained", "from_scratch"]
        if seg_models is None:
            seg_models = ["nnunet", "swinunetr"]
            
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION PIPELINE")
        print(f"Dataset: {dataset}")
        print(f"Methods: {methods}")
        print(f"Model Types: {model_types}")
        print(f"Segmentation Models: {seg_models}")
        print(f"{'='*60}")
        
        all_results = []
        
        for method in methods:
            for model_type in model_types:
                # Skip baseline for from_scratch
                if method == "baseline" and model_type == "from_scratch":
                    continue
                    
                for seg_model in seg_models:
                    print(f"\n{'='*50}")
                    print(f"Evaluating: {method} + {model_type} + {seg_model}")
                    print(f"{'='*50}")
                    
                    stats = self.evaluate_model(dataset, method, model_type, seg_model)
                    
                    if stats:
                        all_results.append(stats)
                        print(f"✅ DICE: {stats['dice_mean']:.2f} ± {stats['dice_std']:.2f}")
                        print(f"✅ NSD: {stats['nsd_mean']:.2f} ± {stats['nsd_std']:.2f}")
                    else:
                        print(f"❌ Evaluation failed")
                        
        # Save results to CSV
        if all_results:
            df = pd.DataFrame(all_results)
            output_file = self.results_dir / f"{dataset}_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_file, index=False)
            print(f"\n✅ Results saved to: {output_file}")
            
            # Generate paper-style table
            self.generate_paper_table(all_results, dataset)
            
        return all_results
        
    def compare_with_paper(self, results, paper_values):
        """Compare evaluation results with paper values"""
        print(f"\n{'='*60}")
        print("COMPARISON WITH PAPER")
        print(f"{'='*60}")
        
        for result in results:
            key = f"{result['method']}_{result['seg_model']}"
            if key in paper_values:
                paper = paper_values[key]
                dice_diff = result['dice_mean'] - paper['dice']
                nsd_diff = result['nsd_mean'] - paper['nsd']
                
                print(f"\n{key}:")
                print(f"  DICE: {result['dice_mean']:.2f} (paper: {paper['dice']:.2f}, diff: {dice_diff:+.2f})")
                print(f"  NSD: {result['nsd_mean']:.2f} (paper: {paper['nsd']:.2f}, diff: {nsd_diff:+.2f})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation models for LeFusion paper")
    parser.add_argument("--dataset", choices=["lidc", "emidec", "all"], default="lidc",
                        help="Dataset to evaluate on")
    parser.add_argument("--methods", nargs="+",
                        choices=["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"],
                        help="Methods to evaluate (default: all)")
    parser.add_argument("--model-types", nargs="+",
                        choices=["pretrained", "from_scratch"],
                        help="Model types to evaluate (default: all)")
    parser.add_argument("--seg-models", nargs="+",
                        choices=["nnunet", "swinunetr"],
                        help="Segmentation models to evaluate (default: all)")
    parser.add_argument("--config", default="../configs/experiment_config.yaml",
                        help="Path to config file")
    parser.add_argument("--compare-paper", action="store_true",
                        help="Compare with paper results")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.config)
    
    # Process datasets
    datasets = ["lidc", "emidec"] if args.dataset == "all" else [args.dataset]
    
    for dataset in datasets:
        results = evaluator.evaluate_all(
            dataset=dataset,
            methods=args.methods,
            model_types=args.model_types,
            seg_models=args.seg_models
        )
        
        # Compare with paper if requested
        if args.compare_paper and dataset == "lidc":
            # Paper values from Table 1
            paper_values = {
                'baseline_nnunet': {'dice': 78.26, 'nsd': 88.90},
                'baseline_swinunetr': {'dice': 78.38, 'nsd': 88.67},
                'lefusion_nnunet': {'dice': 78.77, 'nsd': 89.25},
                'lefusion_swinunetr': {'dice': 78.43, 'nsd': 88.54},
                'lefusion_h_nnunet': {'dice': 80.62, 'nsd': 90.90},
                'lefusion_h_swinunetr': {'dice': 80.95, 'nsd': 90.98},
                'lefusion_h_diffmask_nnunet': {'dice': 83.44, 'nsd': 93.35},
                'lefusion_h_diffmask_swinunetr': {'dice': 83.13, 'nsd': 93.20},
            }
            evaluator.compare_with_paper(results, paper_values)

if __name__ == "__main__":
    main() 