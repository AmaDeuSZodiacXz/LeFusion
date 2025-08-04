#!/usr/bin/env python3
"""
Comprehensive Paper Evaluation Pipeline for LeFusion
Reproduces the exact evaluation table from the paper with all methods
"""

import argparse
import subprocess
import os
import pandas as pd
import numpy as np
from datetime import datetime
import shutil

class PaperEvaluationPipeline:
    def __init__(self):
        self.base_dir = "evaluation_pipeline"
        self.experiments_dir = "paper_experiments"
        self.results_csv = "comprehensive_paper_results.csv"
        
        # Create main experiment directory
        os.makedirs(self.experiments_dir, exist_ok=True)
        
    def setup_experiment_structure(self):
        """Create organized directory structure for all experiments"""
        structure = {
            # Pretrained Models
            "pretrained": {
                "lefusion": "synthetic/pretrained/lefusion",
                "lefusion_h": "synthetic/pretrained/lefusion_h", 
                "lefusion_h_diffmask": "synthetic/pretrained/lefusion_h_diffmask",
                "baseline": "synthetic/pretrained/baseline"
            },
            # From Scratch Models
            "from_scratch": {
                "lefusion": "synthetic/from_scratch/lefusion",
                "lefusion_h": "synthetic/from_scratch/lefusion_h",
                "lefusion_h_diffmask": "synthetic/from_scratch/lefusion_h_diffmask",
                "baseline": "synthetic/from_scratch/baseline"
            },
            # Training Results
            "training": {
                "nnunet": "training/nnunet",
                "swinunetr": "training/swinunetr"
            },
            # Evaluation Results
            "evaluation": "evaluation_results"
        }
        
        for category, paths in structure.items():
            if isinstance(paths, dict):
                for name, path in paths.items():
                    full_path = os.path.join(self.experiments_dir, path)
                    os.makedirs(full_path, exist_ok=True)
                    print(f"Created: {full_path}")
            else:
                full_path = os.path.join(self.experiments_dir, paths)
                os.makedirs(full_path, exist_ok=True)
                print(f"Created: {full_path}")
    
    def generate_synthetic_data_pretrained(self, method):
        """Generate synthetic data using pretrained models"""
        print(f"\n{'='*60}")
        print(f"GENERATING SYNTHETIC DATA: {method.upper()} (PRETRAINED)")
        print(f"{'='*60}")
        
        output_dir = os.path.join(self.experiments_dir, f"synthetic/pretrained/{method}")
        os.makedirs(output_dir, exist_ok=True)
        
        if method == "lefusion":
            # LeFusion with pretrained model
            cmd = [
                "python", "../LeFusion/inference/inference.py",
                "data_type=lidc",
                "model_path=../LeFusion/LeFusion_Model/LIDC/lidc.pt",
                "dataset_root_dir=../data/LIDC/Normal/Image",
                "test_txt_dir=../data/LIDC/Pathological/test.txt",
                f"target_img_path={output_dir}/imagesTr",
                f"target_label_path={output_dir}/labelsTr",
                "batch_size=4",
                "types=3"
            ]
            
        elif method == "lefusion_h":
            # LeFusion-H with pretrained model
            cmd = [
                "python", "../LeFusion/inference/inference.py",
                "data_type=lidc",
                "model_path=../LeFusion/LeFusion_Model/LIDC/lidc.pt",
                "dataset_root_dir=../data/LIDC/Normal/Image",
                "test_txt_dir=../data/LIDC/Pathological/test.txt",
                f"target_img_path={output_dir}/imagesTr",
                f"target_label_path={output_dir}/labelsTr",
                "batch_size=4",
                "types=3"
            ]
            
        elif method == "lefusion_h_diffmask":
            # LeFusion-H + DiffMask with pretrained models
            # First generate LeFusion-H synthetic data
            lefusion_h_dir = os.path.join(self.experiments_dir, "synthetic/pretrained/lefusion_h")
            if not os.path.exists(lefusion_h_dir):
                self.generate_synthetic_data_pretrained("lefusion_h")
            
            # Then apply DiffMask
            cmd = [
                "python", "../DiffMask/inference/inference.py",
                "name=lidc_mask",
                "dataset_root_dir=../data/LIDC/Pathological/Image",
                "test_txt_path=../data/LIDC/Pathological/test.txt",
                f"gen_mask_path={output_dir}",
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "out_dim=1",
                "unet_num_channels=2",
                "model_path=../DiffMask/DiffMask_Model/diffmask.pt"
            ]
            
        elif method == "baseline":
            # Baseline - no synthetic data generation needed
            print("Baseline method - no synthetic data generation required")
            return True
            
        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=self.base_dir)
            print(f"✓ Synthetic data generated for {method} (pretrained)")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to generate synthetic data for {method}: {e}")
            return False
    
    def generate_synthetic_data_from_scratch(self, method):
        """Generate synthetic data using from-scratch models"""
        print(f"\n{'='*60}")
        print(f"GENERATING SYNTHETIC DATA: {method.upper()} (FROM SCRATCH)")
        print(f"{'='*60}")
        
        output_dir = os.path.join(self.experiments_dir, f"synthetic/from_scratch/{method}")
        os.makedirs(output_dir, exist_ok=True)
        
        if method == "lefusion":
            # LeFusion with from-scratch model
            cmd = [
                "python", "../LeFusion/inference/inference.py",
                "data_type=lidc",
                "model_path=../LeFusion/LeFusion_Model/LIDC/model-50.pt",
                "dataset_root_dir=../data/LIDC/Normal/Image",
                "test_txt_dir=../data/LIDC/Pathological/test.txt",
                f"target_img_path={output_dir}/imagesTr",
                f"target_label_path={output_dir}/labelsTr",
                "batch_size=4",
                "types=3"
            ]
            
        elif method == "lefusion_h":
            # LeFusion-H with from-scratch model
            cmd = [
                "python", "../LeFusion/inference/inference.py",
                "data_type=lidc",
                "model_path=../LeFusion/LeFusion_Model/LIDC/model-50.pt",
                "dataset_root_dir=../data/LIDC/Normal/Image",
                "test_txt_dir=../data/LIDC/Pathological/test.txt",
                f"target_img_path={output_dir}/imagesTr",
                f"target_label_path={output_dir}/labelsTr",
                "batch_size=4",
                "types=3"
            ]
            
        elif method == "lefusion_h_diffmask":
            # LeFusion-H + DiffMask with from-scratch models
            # First generate LeFusion-H synthetic data
            lefusion_h_dir = os.path.join(self.experiments_dir, "synthetic/from_scratch/lefusion_h")
            if not os.path.exists(lefusion_h_dir):
                self.generate_synthetic_data_from_scratch("lefusion_h")
            
            # Then apply DiffMask
            cmd = [
                "python", "../DiffMask/inference/inference.py",
                "name=lidc_mask",
                "dataset_root_dir=../data/LIDC/Pathological/Image",
                "test_txt_path=../data/LIDC/Pathological/test.txt",
                f"gen_mask_path={output_dir}",
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "out_dim=1",
                "unet_num_channels=2",
                "model_path=../DiffMask/DiffMask_Model/model-80.pt"
            ]
            
        elif method == "baseline":
            # Baseline - no synthetic data generation needed
            print("Baseline method - no synthetic data generation required")
            return True
            
        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=self.base_dir)
            print(f"✓ Synthetic data generated for {method} (from scratch)")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to generate synthetic data for {method}: {e}")
            return False
    
    def train_segmentation_model(self, method, model_type, segmentation_model):
        """Train segmentation model (nnU-Net or SwinUNETR)"""
        print(f"\n{'='*60}")
        print(f"TRAINING SEGMENTATION MODEL: {method.upper()} + {segmentation_model.upper()}")
        print(f"Model Type: {model_type}")
        print(f"{'='*60}")
        
        # Setup paths
        real_data_dir = "datasets/LIDC_real"
        synthetic_data_dir = os.path.join(self.experiments_dir, f"synthetic/{model_type}/{method}")
        training_output_dir = os.path.join(self.experiments_dir, f"training/{segmentation_model.lower()}/{method}_{model_type}")
        
        os.makedirs(training_output_dir, exist_ok=True)
        
        # Training command
        train_cmd = [
            "python", "run_segmentation_training.py",
            "--real_data_dir", real_data_dir,
            "--model_name", segmentation_model,
            "--output_model_dir", training_output_dir
        ]
        
        # Add synthetic data if not baseline
        if method != "baseline" and os.path.exists(synthetic_data_dir):
            train_cmd.extend(["--synthetic_data_dir", synthetic_data_dir])
        
        try:
            print(f"Running command: {' '.join(train_cmd)}")
            subprocess.run(train_cmd, check=True, cwd=self.base_dir)
            print(f"✓ Training completed for {method} + {segmentation_model}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Training failed for {method} + {segmentation_model}: {e}")
            return False
    
    def evaluate_model(self, method, model_type, segmentation_model):
        """Evaluate trained segmentation model"""
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {method.upper()} + {segmentation_model.upper()}")
        print(f"Model Type: {model_type}")
        print(f"{'='*60}")
        
        # Setup paths
        real_data_dir = "datasets/LIDC_real"
        training_output_dir = os.path.join(self.experiments_dir, f"training/{segmentation_model.lower()}/{method}_{model_type}")
        evaluation_output_dir = os.path.join(self.experiments_dir, f"evaluation_results/{method}_{model_type}_{segmentation_model.lower()}")
        
        os.makedirs(evaluation_output_dir, exist_ok=True)
        
        # Evaluation command
        eval_cmd = [
            "python", "run_segmentation_evaluation.py",
            "--test_data_dir", real_data_dir,
            "--gt_dir", f"{real_data_dir}/labelsTs",
            "--trained_model_path", f"{training_output_dir}/best_metric_model.pth",
            "--model_name", segmentation_model,
            "--output_pred_dir", evaluation_output_dir,
            "--results_csv", self.results_csv,
            "--experiment_name", f"{method}_{model_type}_{segmentation_model.lower()}"
        ]
        
        try:
            print(f"Running command: {' '.join(eval_cmd)}")
            subprocess.run(eval_cmd, check=True, cwd=self.base_dir)
            print(f"✓ Evaluation completed for {method} + {segmentation_model}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Evaluation failed for {method} + {segmentation_model}: {e}")
            return False
    
    def run_complete_pipeline(self, methods=None, model_types=None, segmentation_models=None):
        """Run complete evaluation pipeline"""
        if methods is None:
            methods = ["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"]
        if model_types is None:
            model_types = ["pretrained", "from_scratch"]
        if segmentation_models is None:
            segmentation_models = ["nnUNet", "SwinUNETR"]
        
        print("LeFusion Comprehensive Paper Evaluation Pipeline")
        print("=" * 80)
        print(f"Methods: {methods}")
        print(f"Model Types: {model_types}")
        print(f"Segmentation Models: {segmentation_models}")
        
        # Setup directory structure
        self.setup_experiment_structure()
        
        # Run experiments
        for method in methods:
            for model_type in model_types:
                # Generate synthetic data
                if model_type == "pretrained":
                    success = self.generate_synthetic_data_pretrained(method)
                else:
                    success = self.generate_synthetic_data_from_scratch(method)
                
                if not success:
                    print(f"Skipping {method} {model_type} due to synthetic data generation failure")
                    continue
                
                # Train and evaluate segmentation models
                for segmentation_model in segmentation_models:
                    # Train model
                    train_success = self.train_segmentation_model(method, model_type, segmentation_model)
                    if not train_success:
                        print(f"Skipping evaluation for {method} {model_type} {segmentation_model} due to training failure")
                        continue
                    
                    # Evaluate model
                    self.evaluate_model(method, model_type, segmentation_model)
        
        # Generate final results table
        self.generate_paper_results_table()
    
    def generate_paper_results_table(self):
        """Generate final results table in paper format"""
        if not os.path.exists(self.results_csv):
            print("No results file found!")
            return
        
        df = pd.read_csv(self.results_csv)
        
        print(f"\n{'='*100}")
        print("COMPREHENSIVE PAPER EVALUATION RESULTS")
        print(f"{'='*100}")
        
        # Group by method and model type
        results_summary = []
        
        for method in df['Experiment'].unique():
            exp_data = df[df['Experiment'] == method]
            
            # Parse experiment name to extract components
            parts = method.split('_')
            if len(parts) >= 3:
                method_name = parts[0]
                model_type = parts[1]
                seg_model = parts[2]
                
                # Calculate mean metrics
                dice_mean = exp_data['DICE_Mean'].mean()
                dice_std = exp_data['DICE_Mean'].std()
                nsd_mean = exp_data['NSD_Mean'].mean()
                nsd_std = exp_data['NSD_Mean'].std()
                
                results_summary.append({
                    'Method': method_name,
                    'Model_Type': model_type,
                    'Segmentation_Model': seg_model,
                    'DICE_Mean': dice_mean,
                    'DICE_Std': dice_std,
                    'NSD_Mean': nsd_mean,
                    'NSD_Std': nsd_std
                })
        
        # Create formatted table
        summary_df = pd.DataFrame(results_summary)
        
        print("\nQuantitative Results (Paper Format)")
        print("-" * 100)
        print(f"{'Method':<15} {'Model Type':<15} {'Seg Model':<12} {'DICE (%)':<20} {'NSD (%)':<20}")
        print("-" * 100)
        
        for _, row in summary_df.iterrows():
            dice_str = f"{row['DICE_Mean']:.2f} ± {row['DICE_Std']:.2f}"
            nsd_str = f"{row['NSD_Mean']:.2f} ± {row['NSD_Std']:.2f}"
            print(f"{row['Method']:<15} {row['Model_Type']:<15} {row['Segmentation_Model']:<12} {dice_str:<20} {nsd_str:<20}")
        
        print("-" * 100)
        
        # Save summary to file
        summary_file = f"paper_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
        
        return summary_df

def main():
    parser = argparse.ArgumentParser(description="Comprehensive paper evaluation pipeline for LeFusion")
    parser.add_argument('--methods', nargs='+', 
                       default=['baseline', 'lefusion', 'lefusion_h', 'lefusion_h_diffmask'],
                       help='Methods to evaluate')
    parser.add_argument('--model_types', nargs='+', 
                       default=['pretrained', 'from_scratch'],
                       help='Model types (pretrained/from_scratch)')
    parser.add_argument('--segmentation_models', nargs='+', 
                       default=['nnUNet', 'SwinUNETR'],
                       help='Segmentation models to evaluate')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing results')
    args = parser.parse_args()
    
    pipeline = PaperEvaluationPipeline()
    pipeline.run_complete_pipeline(
        methods=args.methods,
        model_types=args.model_types,
        segmentation_models=args.segmentation_models
    )

if __name__ == '__main__':
    main() 