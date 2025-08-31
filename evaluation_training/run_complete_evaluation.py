#!/usr/bin/env python3
"""
Master Script to Run Complete LeFusion Paper Evaluation
Orchestrates synthetic generation, training, and evaluation
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import time

class CompletePipeline:
    def __init__(self):
        self.base_dir = Path(".")
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def run_command(self, cmd, description):
        """Run a command with error handling"""
        print(f"\n{'='*60}")
        print(f"🔧 {description}")
        print(f"💻 Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {description} completed successfully")
                return True
            else:
                print(f"❌ {description} failed")
                print(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return False
            
    def run_synthetic_generation(self, dataset, model_type, methods):
        """Run synthetic data generation"""
        print(f"\n{'#'*70}")
        print(f"# PHASE 1: SYNTHETIC DATA GENERATION")
        print(f"# Dataset: {dataset}, Model Type: {model_type}")
        print(f"{'#'*70}")
        
        cmd = [
            "python", "synthetic_generation/generate_synthetic_data.py",
            "--dataset", dataset,
            "--model-type", model_type,
            "--resume"
        ]
        
        if methods:
            cmd.extend(["--methods"] + methods)
            
        return self.run_command(cmd, f"Synthetic generation for {dataset} ({model_type})")
        
    def run_training(self, dataset, methods, model_types, seg_models):
        """Run segmentation model training"""
        print(f"\n{'#'*70}")
        print(f"# PHASE 2: SEGMENTATION MODEL TRAINING")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*70}")
        
        cmd = [
            "python", "training/train_segmentation.py",
            "--dataset", dataset,
            "--resume"
        ]
        
        if methods:
            cmd.extend(["--methods"] + methods)
        if model_types:
            cmd.extend(["--model-types"] + model_types)
        if seg_models:
            cmd.extend(["--seg-models"] + seg_models)
            
        return self.run_command(cmd, f"Training segmentation models for {dataset}")
        
    def run_evaluation(self, dataset, methods, model_types, seg_models):
        """Run model evaluation"""
        print(f"\n{'#'*70}")
        print(f"# PHASE 3: MODEL EVALUATION")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*70}")
        
        cmd = [
            "python", "evaluation/evaluate_models.py",
            "--dataset", dataset,
            "--compare-paper"
        ]
        
        if methods:
            cmd.extend(["--methods"] + methods)
        if model_types:
            cmd.extend(["--model-types"] + model_types)
        if seg_models:
            cmd.extend(["--seg-models"] + seg_models)
            
        return self.run_command(cmd, f"Evaluating models for {dataset}")
        
    def run_complete_pipeline(self, dataset="lidc", methods=None, model_types=None, 
                            seg_models=None, skip_synthetic=False, skip_training=False):
        """Run the complete evaluation pipeline"""
        
        print(f"\n{'='*80}")
        print(f"🚀 LEFUSION COMPLETE EVALUATION PIPELINE")
        print(f"{'='*80}")
        print(f"📅 Timestamp: {self.timestamp}")
        print(f"📊 Dataset: {dataset}")
        print(f"🔬 Methods: {methods or 'all'}")
        print(f"🏷️ Model Types: {model_types or 'all'}")
        print(f"🤖 Segmentation Models: {seg_models or 'all'}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Default values
        if model_types is None:
            model_types = ["pretrained", "from_scratch"]
            
        # Phase 1: Synthetic Data Generation
        if not skip_synthetic:
            for model_type in model_types:
                success = self.run_synthetic_generation(dataset, model_type, methods)
                if not success:
                    print(f"⚠️ Synthetic generation failed for {model_type}, continuing...")
        else:
            print("\n⏭️ Skipping synthetic data generation")
            
        # Phase 2: Training
        if not skip_training:
            success = self.run_training(dataset, methods, model_types, seg_models)
            if not success:
                print(f"⚠️ Training failed, continuing to evaluation...")
        else:
            print("\n⏭️ Skipping training")
            
        # Phase 3: Evaluation
        success = self.run_evaluation(dataset, methods, model_types, seg_models)
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"🎉 PIPELINE COMPLETED")
        print(f"⏱️ Total Time: {elapsed_time/60:.2f} minutes")
        print(f"{'='*80}")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="Run complete LeFusion evaluation pipeline")
    
    # Dataset selection
    parser.add_argument("--dataset", choices=["lidc", "emidec", "all"], default="lidc",
                        help="Dataset to process")
    
    # Method selection
    parser.add_argument("--methods", nargs="+",
                        choices=["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"],
                        help="Methods to evaluate (default: all)")
    
    # Model type selection
    parser.add_argument("--model-types", nargs="+",
                        choices=["pretrained", "from_scratch"],
                        help="Model types to use (default: all)")
    
    # Segmentation model selection
    parser.add_argument("--seg-models", nargs="+",
                        choices=["nnunet", "swinunetr"],
                        help="Segmentation models to train/evaluate (default: all)")
    
    # Skip options
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Skip synthetic data generation")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training phase")
    
    # Quick test mode
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test with baseline only")
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.methods = ["baseline"]
        args.model_types = ["pretrained"]
        args.seg_models = ["nnunet"]
        
    pipeline = CompletePipeline()
    
    # Process datasets
    datasets = ["lidc", "emidec"] if args.dataset == "all" else [args.dataset]
    
    for dataset in datasets:
        pipeline.run_complete_pipeline(
            dataset=dataset,
            methods=args.methods,
            model_types=args.model_types,
            seg_models=args.seg_models,
            skip_synthetic=args.skip_synthetic,
            skip_training=args.skip_training
        )

if __name__ == "__main__":
    main() 