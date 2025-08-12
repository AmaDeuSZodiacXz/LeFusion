#!/usr/bin/env python3
"""
LeFusion Comprehensive Paper Evaluation Pipeline
Reproducing the exact evaluation table from the paper
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
from datetime import datetime
import time
import signal

# Global variable to track if we should stop
should_stop = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global should_stop
    print("\n🛑 Received interrupt signal. Stopping gracefully...")
    should_stop = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def check_gpu_status():
    """Check GPU status and memory"""
    try:
        # Check if CUDA is available
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"🔍 GPU Status: {gpu_count} GPU(s) available")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                print(f"   GPU {i}: {props.name}")
                print(f"     Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        else:
            print("⚠️ CUDA not available - using CPU")
            
    except Exception as e:
        print(f"❌ Error checking GPU status: {e}")

def check_system_resources():
    """Check system memory and CPU usage"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        print(f"🔍 System Resources:")
        print(f"   Memory: {memory.percent}% used ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
        print(f"   CPU: {cpu_percent}% used")
        
    except ImportError:
        print("⚠️ psutil not available - cannot check system resources")
    except Exception as e:
        print(f"❌ Error checking system resources: {e}")

def check_dataset_loading_complete(timeout=60):
    """Check if dataset loading is complete by monitoring output"""
    print("🔍 Checking dataset loading status...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Check if loading is complete by looking for specific patterns
        # This is a simple heuristic - in practice you might want more sophisticated checks
        time.sleep(2)
        print("⏳ Still waiting for dataset loading...")

def execute_command_with_realtime_output(cmd, description, timeout=1800):  # เพิ่ม timeout เป็น 30 นาที
    """Execute command with real-time output and timeout protection"""
    print(f"🔧 {description}")
    print(f"💻 Command: {' '.join(cmd)}")
    
    try:
        # Execute command with real-time output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            bufsize=1,
            universal_newlines=True
        )
        
        # Log output in real-time with timeout
        stdout_lines = []
        stderr_lines = []
        start_time = time.time()
        last_output_time = time.time()
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"⏰ Timeout after {timeout} seconds for {description}")
                process.terminate()
                return False, ''.join(stdout_lines), f"Timeout after {timeout} seconds"
            
            # Check if no output for too long (potential hang)
            if time.time() - last_output_time > 300:  # 5 minutes without output
                print(f"⚠️ No output for 5 minutes, checking process status...")
                if process.poll() is None:
                    print(f"❌ Process appears to be hanging, terminating...")
                    process.terminate()
                    return False, ''.join(stdout_lines), "Process hanging - no output for 5 minutes"
            
            # Try to read with timeout
            try:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()
                
                if stdout_line:
                    print(f"📤 {stdout_line.strip()}")
                    stdout_lines.append(stdout_line)
                    last_output_time = time.time()
                
                if stderr_line:
                    print(f"⚠️ {stderr_line.strip()}")
                    stderr_lines.append(stderr_line)
                    last_output_time = time.time()
                
                # Check if process has finished
                if process.poll() is not None:
                    break
                
                # Add progress indicator every 2 minutes  
                elapsed = time.time() - start_time
                if elapsed > 0 and elapsed % 120 < 0.1:  # Every ~2 minutes
                    print(f"⏱️ {description} running for {elapsed:.1f} seconds...")
                    
            except Exception as e:
                print(f"❌ Error reading output: {e}")
                break
        
        # Get return code
        return_code = process.wait()
        
        if return_code == 0:
            print(f"✅ {description} completed successfully")
            return True, ''.join(stdout_lines), ''.join(stderr_lines)
        else:
            print(f"❌ {description} failed with return code {return_code}")
            return False, ''.join(stdout_lines), ''.join(stderr_lines)
            
    except Exception as e:
        print(f"❌ Exception during {description}: {e}")
        return False, "", str(e)

class PaperEvaluationPipeline:
    def __init__(self):
        self.base_dir = "."
        self.experiments_dir = "paper_experiments"
        self.results_csv = "comprehensive_paper_results.csv"
        
        # Available methods and model types
        self.methods = ["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"]
        self.model_types = ["pretrained", "from_scratch"]
        self.segmentation_models = ["nnunet", "swinunetr"]
    
    def create_experiment_logger(self, method, model_type, segmentation_model):
        """Create a simple print-based logger for an experiment"""
        experiment_name = f"{method}_{model_type}_{segmentation_model}"
        
        print(f"🔬 EXPERIMENT: {experiment_name}")
        print("=" * 60)
        
        return None, None  # No logger object, no log file path
    
    def setup_experiment_structure(self):
        """Create directory structure for experiments"""
        print("Setting up experiment directory structure...")
        
        # Create synthetic data directories
        for model_type in self.model_types:
            for method in self.methods:
                dir_path = os.path.join(self.experiments_dir, f"synthetic/{model_type}/{method}")
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created: {dir_path}")
        
        # Create training directories
        for seg_model in self.segmentation_models:
            dir_path = os.path.join(self.experiments_dir, f"training/{seg_model}")
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created: {dir_path}")
        
        # Create evaluation results directory
        os.makedirs(os.path.join(self.experiments_dir, "evaluation_results"), exist_ok=True)
        print(f"Created: {self.experiments_dir}/evaluation_results")
    
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
                "batch_size=1",  # เปลี่ยนจาก 4 เป็น 1
                "types=3",
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=1",
                "cond_dim=16"
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
                "batch_size=1",  # เปลี่ยนจาก 4 เป็น 1
                "types=3",
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=1",
                "cond_dim=16"
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
        
        else:
            print(f"❌ Unknown method: {method}")
            return False
        
        success, stdout, stderr = execute_command_with_realtime_output(
            cmd, f"Generate synthetic data for {method} (pretrained)"
        )
        
        if success:
            print(f"✅ Synthetic data generated successfully for {method}")
            return True
        else:
            print(f"❌ Failed to generate synthetic data for {method}")
            print(f"Error: {stderr}")
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
                "model_path=../LeFusion/LeFusion_Model/LIDC/lidc_from_scratch.pt",
                "dataset_root_dir=../data/LIDC/Normal/Image",
                "test_txt_dir=../data/LIDC/Pathological/test.txt",
                f"target_img_path={output_dir}/imagesTr",
                f"target_label_path={output_dir}/labelsTr",
                "batch_size=1",
                "types=3",
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=1",
                "cond_dim=16"
            ]
            
        elif method == "lefusion_h":
            # LeFusion-H with from-scratch model
            cmd = [
                "python", "../LeFusion/inference/inference.py",
                "data_type=lidc",
                "model_path=../LeFusion/LeFusion_Model/LIDC/lidc_from_scratch.pt",
                "dataset_root_dir=../data/LIDC/Normal/Image",
                "test_txt_dir=../data/LIDC/Pathological/test.txt",
                f"target_img_path={output_dir}/imagesTr",
                f"target_label_path={output_dir}/labelsTr",
                "batch_size=1",
                "types=3",
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=1",
                "cond_dim=16"
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
                "model_path=../DiffMask/DiffMask_Model/diffmask_from_scratch.pt"
            ]
            
        elif method == "baseline":
            # Baseline - no synthetic data generation needed
            print("Baseline method - no synthetic data generation required")
            return True
        
        else:
            print(f"❌ Unknown method: {method}")
            return False
        
        success, stdout, stderr = execute_command_with_realtime_output(
            cmd, f"Generate synthetic data for {method} (from scratch)"
        )
        
        if success:
            print(f"✅ Synthetic data generated successfully for {method}")
            return True
        else:
            print(f"❌ Failed to generate synthetic data for {method}")
            print(f"Error: {stderr}")
            return False

    def train_segmentation_model(self, method, model_type, segmentation_model):
        """Train segmentation model with combined real and synthetic data"""
        print(f"\n{'='*60}")
        print(f"TRAINING SEGMENTATION MODEL: {segmentation_model.upper()}")
        print(f"Method: {method}, Model Type: {model_type}")
        print(f"{'='*60}")
        
        # Determine data paths
        real_data_dir = "datasets/LIDC_real"
        synthetic_data_dir = None
        
        if method != "baseline":
            synthetic_data_dir = os.path.join(self.experiments_dir, f"synthetic/{model_type}/{method}")
            if not os.path.exists(synthetic_data_dir):
                print(f"❌ Synthetic data directory not found: {synthetic_data_dir}")
                return False
        
        # Setup training output directory
        training_output_dir = os.path.join(self.experiments_dir, f"training/{segmentation_model}/{method}_{model_type}")
        os.makedirs(training_output_dir, exist_ok=True)
        
        # Prepare training command
        cmd = [
            "python", "run_segmentation_training.py",
            "--real_data_dir", real_data_dir,
            "--model_name", segmentation_model,
            "--output_model_dir", training_output_dir
        ]
        
        if synthetic_data_dir:
            cmd.extend(["--synthetic_data_dir", synthetic_data_dir])
        
        success, stdout, stderr = execute_command_with_realtime_output(
            cmd, f"Train {segmentation_model} for {method} ({model_type})"
        )
        
        if success:
            print(f"✅ Training completed successfully for {segmentation_model}")
            return True
        else:
            print(f"❌ Training failed for {segmentation_model}")
            print(f"Error: {stderr}")
            return False

    def evaluate_model(self, method, model_type, segmentation_model):
        """Evaluate trained segmentation model"""
        print(f"\n{'='*60}")
        print(f"EVALUATING SEGMENTATION MODEL: {segmentation_model.upper()}")
        print(f"Method: {method}, Model Type: {model_type}")
        print(f"{'='*60}")
        
        # Setup paths
        test_data_dir = "datasets/LIDC_real"
        gt_dir = "datasets/LIDC_real/labelsTs"
        trained_model_path = os.path.join(self.experiments_dir, f"training/{segmentation_model}/{method}_{model_type}")
        output_pred_dir = os.path.join(self.experiments_dir, f"evaluation_results/{method}_{model_type}_{segmentation_model}")
        results_csv = self.results_csv
        experiment_name = f"{method}_{model_type}_{segmentation_model}"
        
        # Prepare evaluation command
        cmd = [
            "python", "run_segmentation_evaluation.py",
            "--test_data_dir", test_data_dir,
            "--gt_dir", gt_dir,
            "--trained_model_path", trained_model_path,
            "--model_name", segmentation_model,
            "--output_pred_dir", output_pred_dir,
            "--results_csv", results_csv,
            "--experiment_name", experiment_name
        ]
        
        success, stdout, stderr = execute_command_with_realtime_output(
            cmd, f"Evaluate {segmentation_model} for {method} ({model_type})"
        )
        
        if success:
            print(f"✅ Evaluation completed successfully for {segmentation_model}")
            return True
        else:
            print(f"❌ Evaluation failed for {segmentation_model}")
            print(f"Error: {stderr}")
            return False

    def run_complete_pipeline(self, methods=None, model_types=None, segmentation_models=None):
        """Run the complete paper evaluation pipeline with real-time output"""
        if methods is None:
            methods = self.methods
        if model_types is None:
            model_types = self.model_types
        if segmentation_models is None:
            segmentation_models = self.segmentation_models
            
        print("🚀 STARTING COMPREHENSIVE PAPER EVALUATION PIPELINE")
        print("=" * 80)
        print(f"📋 Methods: {methods}")
        print(f"📋 Model Types: {model_types}")
        print(f"📋 Segmentation Models: {segmentation_models}")
        print("=" * 80)
        
        # Check system resources at start
        print("🔍 Checking system resources...")
        check_gpu_status()
        check_system_resources()
        
        # Setup experiment structure
        print("📁 Setting up experiment directory structure...")
        self.setup_experiment_structure()
        
        total_experiments = len(methods) * len(model_types) * len(segmentation_models)
        current_experiment = 0
        
        for method in methods:
            for model_type in model_types:
                for segmentation_model in segmentation_models:
                    # Check for interrupt
                    if should_stop:
                        print("🛑 Pipeline stopped by user request")
                        return
                    
                    current_experiment += 1
                    print(f"🔄 PROGRESS: {current_experiment}/{total_experiments}")
                    print(f"🔬 Running: {method} + {model_type} + {segmentation_model}")
                    print("-" * 60)
                    
                    # Create experiment-specific logger
                    _, _ = self.create_experiment_logger(method, model_type, segmentation_model)
                    
                    # Check resources before starting experiment
                    print("🔍 Checking resources before experiment...")
                    check_gpu_status()
                    check_system_resources()
                    
                    start_time = time.time()
                    
                    # Generate synthetic data
                    if method != "baseline":
                        print(f"🎨 Generating synthetic data for {method} ({model_type})...")
                        if model_type == "pretrained":
                            success = self.generate_synthetic_data_pretrained(method)
                        else:
                            success = self.generate_synthetic_data_from_scratch(method)
                        
                        if not success:
                            print(f"❌ Failed to generate synthetic data for {method}")
                            continue
                        print(f"✅ Synthetic data generated for {method}")
                    
                    # Train segmentation model
                    print(f"🏋️ Training {segmentation_model} for {method} ({model_type})...")
                    print("⏳ Waiting for dataset loading to complete...")
                    time.sleep(5)  # รอ 5 วินาทีให้ dataset load เสร็จ
                    success = self.train_segmentation_model(method, model_type, segmentation_model)
                    if not success:
                        print(f"❌ Failed to train {segmentation_model} for {method}")
                        continue
                    print(f"✅ Training completed for {segmentation_model}")
                    
                    # Evaluate model
                    print(f"📊 Evaluating {segmentation_model} for {method} ({model_type})...")
                    success = self.evaluate_model(method, model_type, segmentation_model)
                    if not success:
                        print(f"❌ Failed to evaluate {segmentation_model} for {method}")
                        continue
                    print(f"✅ Evaluation completed for {segmentation_model}")
                    
                    elapsed_time = time.time() - start_time
                    print(f"⏱️ Time taken: {elapsed_time:.2f} seconds")
                    print("=" * 60)
                    print(f"🎉 EXPERIMENT COMPLETED: {method}_{model_type}_{segmentation_model}")
                    print("=" * 60)
        
        # Generate final results table
        print("📊 Generating final results table...")
        self.generate_paper_results_table()
        
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved to: {self.results_csv}")
    
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
        print(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="LeFusion Comprehensive Paper Evaluation Pipeline")
    parser.add_argument('--methods', nargs='+', 
                       default=['baseline', 'lefusion', 'lefusion_h', 'lefusion_h_diffmask'],
                       help='Methods to evaluate')
    parser.add_argument('--model_types', nargs='+', 
                       default=['pretrained', 'from_scratch'],
                       help='Model types (pretrained/from_scratch)')
    parser.add_argument('--segmentation_models', nargs='+', 
                       default=['nnunet', 'swinunetr'],
                       help='Segmentation models to evaluate')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing progress')
    args = parser.parse_args()
    
    pipeline = PaperEvaluationPipeline()
    pipeline.run_complete_pipeline(
        methods=args.methods,
        model_types=args.model_types,
        segmentation_models=args.segmentation_models
    )

if __name__ == '__main__':
    main() 