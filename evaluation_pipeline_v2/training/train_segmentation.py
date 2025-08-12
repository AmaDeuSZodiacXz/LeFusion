#!/usr/bin/env python3
"""
Training Pipeline for Segmentation Models
Supports: nnU-Net and SwinUNETR
Trains on combinations of real and synthetic data
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import time
import shutil

class SegmentationTrainer:
    def __init__(self, config_path="../configs/experiment_config.yaml"):
        """Initialize segmentation trainer with config"""
        # Get the evaluation_pipeline_v2 directory as base
        script_dir = Path(__file__).parent  # training/
        self.base_dir = script_dir.parent   # evaluation_pipeline_v2/
        
        # Load config with proper path resolution
        config_full_path = self.base_dir / "configs" / "experiment_config.yaml"
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set directories relative to evaluation_pipeline_v2
        self.output_dir = self.base_dir / "trained_models"
        self.checkpoint_file = self.output_dir / "training_checkpoint.json"
        
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        
    def load_checkpoint(self):
        """Load checkpoint for resume capability"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {}
        
    def save_checkpoint(self, checkpoint):
        """Save checkpoint for resume capability"""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
    def get_model_output_path(self, dataset, method, model_type, seg_model):
        """Get organized output path for trained models"""
        # Structure: trained_models/dataset/method/model_type/seg_model/
        path = self.output_dir / dataset / method / model_type / seg_model
        return path
        
    def prepare_training_data(self, dataset, method, model_type):
        """Prepare training data by combining real and synthetic data"""
        print(f"\n📁 Preparing training data")
        print(f"   Dataset: {dataset}")
        print(f"   Method: {method}")
        print(f"   Model Type: {model_type}")
        
        # Get real data path
        real_data_dir = Path(self.config['datasets'][dataset]['real_data_dir'])
        
        if method == "baseline":
            # Baseline uses only real data
            return real_data_dir
            
        # For other methods, combine real and synthetic data
        synthetic_base = self.base_dir / "synthetic_data" / dataset / model_type / method
        
        # Create combined data directory
        combined_dir = self.base_dir / "temp_training_data" / f"{dataset}_{method}_{model_type}"
        os.makedirs(combined_dir / "imagesTr", exist_ok=True)
        os.makedirs(combined_dir / "labelsTr", exist_ok=True)
        
        # Copy real data
        if (real_data_dir / "imagesTr").exists():
            for img in (real_data_dir / "imagesTr").glob("*"):
                shutil.copy2(img, combined_dir / "imagesTr")
        if (real_data_dir / "labelsTr").exists():
            for label in (real_data_dir / "labelsTr").glob("*"):
                shutil.copy2(label, combined_dir / "labelsTr")
                
        # Add synthetic data based on method
        if method == "lefusion":
            # Add P+P' data
            synthetic_dir = synthetic_base / "P_P_prime"
            if synthetic_dir.exists():
                self._add_synthetic_data(synthetic_dir, combined_dir)
                
        elif method == "lefusion_h":
            # Add P+P' and P+N' data
            for data_type in ["P_P_prime", "P_N_prime"]:
                synthetic_dir = synthetic_base / data_type
                if synthetic_dir.exists():
                    self._add_synthetic_data(synthetic_dir, combined_dir)
                    
        elif method == "lefusion_h_diffmask":
            # Add all combinations
            for data_type in ["P_N_prime", "P_N_double_prime", "P_P_prime_N_double_prime"]:
                synthetic_dir = synthetic_base / data_type
                if synthetic_dir.exists():
                    self._add_synthetic_data(synthetic_dir, combined_dir)
                    
        return combined_dir
        
    def _add_synthetic_data(self, source_dir, target_dir):
        """Add synthetic data to combined directory"""
        # Copy synthetic images
        if (source_dir / "imagesTr").exists():
            for img in (source_dir / "imagesTr").glob("*"):
                # Add prefix to avoid conflicts
                new_name = f"syn_{img.name}"
                shutil.copy2(img, target_dir / "imagesTr" / new_name)
                
        # Copy synthetic labels
        if (source_dir / "labelsTr").exists():
            for label in (source_dir / "labelsTr").glob("*"):
                # Add prefix to avoid conflicts
                new_name = f"syn_{label.name}"
                shutil.copy2(label, target_dir / "labelsTr" / new_name)
                
    def train_model(self, dataset, method, model_type, seg_model):
        """Train segmentation model"""
        print(f"\n🏋️ Training {seg_model}")
        print(f"   Dataset: {dataset}")
        print(f"   Method: {method}")
        print(f"   Model Type: {model_type}")
        
        # Prepare training data
        training_data_dir = self.prepare_training_data(dataset, method, model_type)
        
        # Get output path
        output_dir = self.get_model_output_path(dataset, method, model_type, seg_model)
        os.makedirs(output_dir, exist_ok=True)
        
        # Build training command
        cmd = [
            "python", "../../evaluation_pipeline/DiffTumor/STEP3.SegmentationModel/main.py",
            "--data_root", str(training_data_dir),
            "--logdir", str(output_dir),
            "--model_name", seg_model,
            "--max_epochs", str(self.config['training']['max_epochs']),
            "--batch_size", str(self.config['training']['batch_size']),
            "--optim_lr", str(self.config['training']['learning_rate']),
            "--optim_name", self.config['training']['optimizer'],
            "--workers", str(self.config['training']['workers']),
            "--cache_rate", str(self.config['training']['cache_rate']),
            "--val_every", str(self.config['training']['val_every']),
            "--save_checkpoint",
            "--datafold_dir", str(training_data_dir),
            "--tumor_type", "liver",  # Default for LIDC
            "--organ_type", "liver",
            "--fold", "0"
        ]
        
        # Add dataset-specific parameters
        if dataset == "emidec":
            cmd.extend(["--tumor_type", "cardiac", "--organ_type", "heart"])
            
        # Execute training
        try:
            print(f"💻 Running: {' '.join(cmd[:5])}...")  # Show abbreviated command
            
            # Run with timeout (4 hours for training)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
            
            if result.returncode == 0:
                print(f"✅ Training completed successfully")
                
                # Check if model was saved
                model_file = output_dir / "best_metric_model.pth"
                if model_file.exists():
                    print(f"✅ Model saved: {model_file}")
                    return True
                else:
                    print(f"⚠️ Model file not found")
                    return False
            else:
                print(f"❌ Training failed")
                print(f"Error: {result.stderr[-1000:]}")  # Show last 1000 chars of error
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Training timeout after 4 hours")
            return False
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
        finally:
            # Clean up temporary combined data
            if method != "baseline" and training_data_dir != Path(self.config['datasets'][dataset]['real_data_dir']):
                if training_data_dir.exists():
                    shutil.rmtree(training_data_dir)
                    
    def train_all(self, dataset="lidc", methods=None, model_types=None, seg_models=None, resume=False):
        """Train all specified configurations"""
        
        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else {}
        
        # Default values
        if methods is None:
            methods = ["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"]
        if model_types is None:
            model_types = ["pretrained", "from_scratch"]
        if seg_models is None:
            seg_models = ["nnunet", "swinunetr"]
            
        print(f"\n{'='*60}")
        print(f"SEGMENTATION MODEL TRAINING PIPELINE")
        print(f"Dataset: {dataset}")
        print(f"Methods: {methods}")
        print(f"Model Types: {model_types}")
        print(f"Segmentation Models: {seg_models}")
        print(f"Resume: {resume}")
        print(f"{'='*60}")
        
        results = {}
        total_configs = len(methods) * len(model_types) * len(seg_models)
        current = 0
        
        for method in methods:
            for model_type in model_types:
                # Skip baseline for from_scratch (doesn't make sense)
                if method == "baseline" and model_type == "from_scratch":
                    continue
                    
                for seg_model in seg_models:
                    current += 1
                    
                    # Check if already completed
                    checkpoint_key = f"{dataset}_{method}_{model_type}_{seg_model}"
                    if resume and checkpoint.get(checkpoint_key, {}).get('completed', False):
                        print(f"\n[{current}/{total_configs}] ✅ Skipping {checkpoint_key} - already completed")
                        continue
                        
                    print(f"\n[{current}/{total_configs}] {'='*50}")
                    print(f"Training: {method} + {model_type} + {seg_model}")
                    print(f"{'='*50}")
                    
                    start_time = time.time()
                    
                    # Check if synthetic data exists (if needed)
                    if method != "baseline":
                        synthetic_dir = self.base_dir / "synthetic_data" / dataset / model_type / method
                        if not synthetic_dir.exists():
                            print(f"⚠️ Synthetic data not found: {synthetic_dir}")
                            print(f"   Please run synthetic generation first")
                            success = False
                        else:
                            success = self.train_model(dataset, method, model_type, seg_model)
                    else:
                        success = self.train_model(dataset, method, model_type, seg_model)
                        
                    # Update checkpoint
                    elapsed_time = time.time() - start_time
                    checkpoint[checkpoint_key] = {
                        'completed': success,
                        'timestamp': datetime.now().isoformat(),
                        'elapsed_time': elapsed_time
                    }
                    self.save_checkpoint(checkpoint)
                    
                    results[checkpoint_key] = success
                    
                    if success:
                        print(f"✅ Training completed in {elapsed_time/60:.2f} minutes")
                    else:
                        print(f"❌ Training failed")
                        
        # Print summary
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*60}")
        
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        print(f"Completed: {success_count}/{total_count}")
        print(f"\nDetailed Results:")
        for key, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {key}")
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Train segmentation models for LeFusion evaluation")
    parser.add_argument("--dataset", choices=["lidc", "emidec", "all"], default="lidc",
                        help="Dataset to train on")
    parser.add_argument("--methods", nargs="+",
                        choices=["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"],
                        help="Methods to train (default: all)")
    parser.add_argument("--model-types", nargs="+",
                        choices=["pretrained", "from_scratch"],
                        help="Model types to use (default: all)")
    parser.add_argument("--seg-models", nargs="+",
                        choices=["nnunet", "swinunetr"],
                        help="Segmentation models to train (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--config", default="../configs/experiment_config.yaml",
                        help="Path to config file")
    
    args = parser.parse_args()
    
    trainer = SegmentationTrainer(args.config)
    
    # Process datasets
    datasets = ["lidc", "emidec"] if args.dataset == "all" else [args.dataset]
    
    for dataset in datasets:
        trainer.train_all(
            dataset=dataset,
            methods=args.methods,
            model_types=args.model_types,
            seg_models=args.seg_models,
            resume=args.resume
        )

if __name__ == "__main__":
    main() 