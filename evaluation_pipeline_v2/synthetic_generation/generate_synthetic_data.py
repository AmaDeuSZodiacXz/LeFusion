#!/usr/bin/env python3
"""
Synthetic Data Generation Pipeline for LeFusion Paper Evaluation
Supports: LeFusion, LeFusion-H, LeFusion-H+DiffMask
With resume capability and organized folder structure
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

class SyntheticDataGenerator:
    def __init__(self, config_path="configs/experiment_config.yaml"):
        """Initialize synthetic data generator with config"""
        # Get the evaluation_pipeline_v2 directory as base
        script_dir = Path(__file__).parent  # synthetic_generation/
        self.base_dir = script_dir.parent   # evaluation_pipeline_v2/
        
        # Load config with robust path resolution
        candidate_path = Path(config_path)
        if not candidate_path.is_absolute():
            candidate_path = (self.base_dir / candidate_path).resolve()
        if not candidate_path.exists():
            # Fallback to default inside evaluation_pipeline_v2/configs/
            fallback_path = (self.base_dir / "configs" / "experiment_config.yaml").resolve()
            candidate_path = fallback_path
        with open(candidate_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set output directory relative to evaluation_pipeline_v2
        self.output_dir = self.base_dir / "synthetic_data"
        self.checkpoint_file = self.output_dir / "generation_checkpoint.json"
        
        print(f"📁 Base directory: {self.base_dir}")
        print(f"🧩 Config file: {candidate_path}")
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
            
    def get_output_path(self, dataset, method, model_type, data_type):
        """Get organized output path for synthetic data"""
        # Structure: synthetic_data/dataset/model_type/method/data_type/
        path = self.output_dir / dataset / model_type / method / data_type
        return path
        
    def generate_lefusion(self, dataset, model_type, output_dir):
        """Generate synthetic data using LeFusion"""
        print(f"\n🎨 Generating LeFusion synthetic data")
        print(f"   Dataset: {dataset}")
        print(f"   Model Type: {model_type}")
        print(f"   Output: {output_dir}")
        
        # Get model path based on model type
        if model_type == "pretrained":
            model_path = self.config['model_weights']['pretrained']['lefusion'][dataset]
        else:
            model_path = self.config['model_weights']['from_scratch']['lefusion'][dataset]
            
        # Check if model exists
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return False
            
        # Create output directories
        os.makedirs(output_dir / "imagesTr", exist_ok=True)
        os.makedirs(output_dir / "labelsTr", exist_ok=True)
        
        # Build command based on dataset
        if dataset == "lidc":
            cmd = [
                "python", "../LeFusion/inference/inference.py",
                f"data_type={dataset}",
                f"model_path={model_path}",
                f"dataset_root_dir={self.config['datasets'][dataset]['normal_dir']}",
                f"test_txt_dir={self.config['datasets'][dataset]['test_file']}",
                f"target_img_path={output_dir}/imagesTr",
                f"target_label_path={output_dir}/labelsTr",
                "batch_size=1",
                "types=3",  # For LIDC
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=1",
                "cond_dim=16"
            ]
        elif dataset == "emidec":
            cmd = [
                "python", "../LeFusion/inference/inference.py",
                f"data_type={dataset}",
                f"model_path={model_path}",
                f"dataset_root_dir={self.config['datasets'][dataset]['normal_dir']}",
                f"target_img_path={output_dir}/imagesTr",
                f"target_label_path={output_dir}/labelsTr",
                "batch_size=1",
                "types=1",  # For EMIDEC
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=1",
                "cond_dim=16"
            ]
        else:
            print(f"❌ Unknown dataset: {dataset}")
            return False
            
        # Execute command
        try:
            print(f"💻 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"✅ LeFusion generation completed successfully")
                return True
            else:
                print(f"❌ LeFusion generation failed")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after 1 hour")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
            
    def generate_lefusion_h(self, dataset, model_type, output_dir):
        """Generate synthetic data using LeFusion-H (histogram conditioned)"""
        print(f"\n🎨 Generating LeFusion-H synthetic data")
        print(f"   Dataset: {dataset}")
        print(f"   Model Type: {model_type}")
        print(f"   Output: {output_dir}")
        
        # LeFusion-H uses the same model but with histogram conditioning
        # This requires the histogram clusters file
        if model_type == "pretrained":
            model_path = self.config['model_weights']['pretrained']['lefusion'][dataset]
        else:
            model_path = self.config['model_weights']['from_scratch']['lefusion'][dataset]
            
        # Check if model exists
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return False
            
        # Create output directories
        os.makedirs(output_dir / "imagesTr", exist_ok=True)
        os.makedirs(output_dir / "labelsTr", exist_ok=True)
        
        # Build command with histogram conditioning
        if dataset == "lidc":
            cmd = [
                "python", "../LeFusion/inference/inference.py",
                f"data_type={dataset}_hist",  # Use histogram version
                f"model_path={model_path}",
                f"dataset_root_dir={self.config['datasets'][dataset]['normal_dir']}",
                f"test_txt_dir={self.config['datasets'][dataset]['test_file']}",
                f"target_img_path={output_dir}/imagesTr",
                f"target_label_path={output_dir}/labelsTr",
                "batch_size=1",
                "types=3",
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=1",
                "cond_dim=16",
                "use_histogram=True"  # Enable histogram conditioning
            ]
        elif dataset == "emidec":
            cmd = [
                "python", "../LeFusion/inference/inference.py",
                f"data_type={dataset}_hist",
                f"model_path={model_path}",
                f"dataset_root_dir={self.config['datasets'][dataset]['normal_dir']}",
                f"target_img_path={output_dir}/imagesTr",
                f"target_label_path={output_dir}/labelsTr",
                "batch_size=1",
                "types=1",
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=1",
                "cond_dim=16",
                "use_histogram=True"
            ]
        else:
            print(f"❌ Unknown dataset: {dataset}")
            return False
            
        # Execute command
        try:
            print(f"💻 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"✅ LeFusion-H generation completed successfully")
                return True
            else:
                print(f"❌ LeFusion-H generation failed")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after 1 hour")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
            
    def generate_diffmask_enhancement(self, dataset, model_type, input_dir, output_dir):
        """Enhance synthetic data with DiffMask"""
        print(f"\n🎨 Enhancing with DiffMask")
        print(f"   Dataset: {dataset}")
        print(f"   Model Type: {model_type}")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        
        # Get DiffMask model path
        if model_type == "pretrained":
            model_path = self.config['model_weights']['pretrained']['diffmask'][dataset]
        else:
            model_path = self.config['model_weights']['from_scratch']['diffmask'][dataset]
            
        # Check if model exists
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return False
            
        # Create output directories
        os.makedirs(output_dir / "imagesTr", exist_ok=True)
        os.makedirs(output_dir / "labelsTr", exist_ok=True)
        
        # Build DiffMask command
        cmd = [
            "python", "../DiffMask/inference/inference.py",
            f"model_path={model_path}",
            f"input_img_path={input_dir}/imagesTr",
            f"input_label_path={input_dir}/labelsTr",
            f"target_img_path={output_dir}/imagesTr",
            f"target_label_path={output_dir}/labelsTr",
            "batch_size=1",
            "out_dim=1",
            "unet_num_channels=2"
        ]
        
        # Execute command
        try:
            print(f"💻 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"✅ DiffMask enhancement completed successfully")
                return True
            else:
                print(f"❌ DiffMask enhancement failed")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout after 1 hour")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
            
    def generate_all(self, dataset="lidc", model_type="pretrained", methods=None, resume=False):
        """Generate all synthetic data for specified configuration"""
        
        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else {}
        
        # Default to all methods if not specified
        if methods is None:
            methods = ["lefusion", "lefusion_h", "lefusion_h_diffmask"]
            
        print(f"\n{'='*60}")
        print(f"SYNTHETIC DATA GENERATION PIPELINE")
        print(f"Dataset: {dataset}")
        print(f"Model Type: {model_type}")
        print(f"Methods: {methods}")
        print(f"Resume: {resume}")
        print(f"{'='*60}")
        
        results = {}
        
        for method in methods:
            # Check if already completed
            checkpoint_key = f"{dataset}_{model_type}_{method}"
            if resume and checkpoint.get(checkpoint_key, {}).get('completed', False):
                print(f"\n✅ Skipping {method} - already completed")
                continue
                
            print(f"\n{'='*50}")
            print(f"Processing: {method}")
            print(f"{'='*50}")
            
            start_time = time.time()
            
            if method == "lefusion":
                # Generate P+P' data
                output_dir = self.get_output_path(dataset, method, model_type, "P_P_prime")
                success = self.generate_lefusion(dataset, model_type, output_dir)
                
            elif method == "lefusion_h":
                # Generate P+P' and P+N' data
                output_dir_pp = self.get_output_path(dataset, method, model_type, "P_P_prime")
                success_pp = self.generate_lefusion_h(dataset, model_type, output_dir_pp)
                
                output_dir_pn = self.get_output_path(dataset, method, model_type, "P_N_prime")
                success_pn = self.generate_lefusion_h(dataset, model_type, output_dir_pn)
                
                success = success_pp and success_pn
                
            elif method == "lefusion_h_diffmask":
                # First generate LeFusion-H data
                temp_dir = self.get_output_path(dataset, "lefusion_h_temp", model_type, "temp")
                success_h = self.generate_lefusion_h(dataset, model_type, temp_dir)
                
                if success_h:
                    # Then enhance with DiffMask for different combinations
                    output_dir_pn = self.get_output_path(dataset, method, model_type, "P_N_prime")
                    success_pn = self.generate_diffmask_enhancement(dataset, model_type, temp_dir, output_dir_pn)
                    
                    output_dir_pn2 = self.get_output_path(dataset, method, model_type, "P_N_double_prime")
                    success_pn2 = self.generate_diffmask_enhancement(dataset, model_type, temp_dir, output_dir_pn2)
                    
                    output_dir_all = self.get_output_path(dataset, method, model_type, "P_P_prime_N_double_prime")
                    success_all = self.generate_diffmask_enhancement(dataset, model_type, temp_dir, output_dir_all)
                    
                    success = success_pn and success_pn2 and success_all
                else:
                    success = False
                    
            else:
                print(f"❌ Unknown method: {method}")
                success = False
                
            # Update checkpoint
            elapsed_time = time.time() - start_time
            checkpoint[checkpoint_key] = {
                'completed': success,
                'timestamp': datetime.now().isoformat(),
                'elapsed_time': elapsed_time
            }
            self.save_checkpoint(checkpoint)
            
            results[method] = success
            
            if success:
                print(f"✅ {method} completed in {elapsed_time:.2f} seconds")
            else:
                print(f"❌ {method} failed")
                
        # Print summary
        print(f"\n{'='*60}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*60}")
        for method, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"{method:30} {status}")
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for LeFusion paper evaluation")
    parser.add_argument("--dataset", choices=["lidc", "emidec", "all"], default="lidc",
                        help="Dataset to generate synthetic data for")
    parser.add_argument("--model-type", choices=["pretrained", "from_scratch", "all"], default="pretrained",
                        help="Model type to use")
    parser.add_argument("--methods", nargs="+", 
                        choices=["lefusion", "lefusion_h", "lefusion_h_diffmask"],
                        help="Methods to generate (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--config", default="configs/experiment_config.yaml",
                        help="Path to config file")
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(args.config)
    
    # Process datasets
    datasets = ["lidc", "emidec"] if args.dataset == "all" else [args.dataset]
    model_types = ["pretrained", "from_scratch"] if args.model_type == "all" else [args.model_type]
    
    for dataset in datasets:
        for model_type in model_types:
            generator.generate_all(
                dataset=dataset,
                model_type=model_type,
                methods=args.methods,
                resume=args.resume
            )

if __name__ == "__main__":
    main() 