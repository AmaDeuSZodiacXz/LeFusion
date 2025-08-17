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

import re

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
            
        # Set directories relative to evaluation_pipeline_v2 using config
        output_cfg = (self.config or {}).get("output", {})
        trained_models_dir_name = output_cfg.get("trained_models", "trained_models")
        self.output_dir = self.base_dir / trained_models_dir_name
        self.checkpoint_file = self.output_dir / "training_checkpoint.json"
        
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📁 Output directory (trained models): {self.output_dir}")
        
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
    
    def _get_latest_checkpoint_info(self, output_dir: Path):
        """Return (checkpoint_path, epoch_idx) if found, else (None, -1).
        Prefers epoch_XXXX.pt, falls back to model_final.pt if present.
        """
        latest_epoch = -1
        latest_file = None
        if output_dir.exists():
            for f in output_dir.glob('epoch_*.pt'):
                m = re.match(r'^epoch_(\d+)\.pt$', f.name)
                if m:
                    e = int(m.group(1))
                    if e > latest_epoch:
                        latest_epoch = e
                        latest_file = f
        if latest_file is not None:
            return latest_file, latest_epoch
        # Fallback to rolling checkpoint
        model_final = output_dir / 'model_final.pt'
        if model_final.exists():
            try:
                import torch
                ckp = torch.load(model_final, map_location='cpu')
                e = ckp.get('epoch', -1)
                if isinstance(e, (int, float)):
                    return model_final, int(e)
            except Exception:
                return model_final, -1
        return None, -1
        
    def prepare_training_data(self, dataset, method, model_type):
        """Prepare training data by combining real and synthetic data"""
        print(f"\n📁 Preparing training data")
        print(f"   Dataset: {dataset}")
        print(f"   Method: {method}")
        print(f"   Model Type: {model_type}")
        
        # Get real data path
        real_data_dir = Path(self.config['datasets'][dataset]['real_data_dir'])
        if not real_data_dir.exists():
            # Fallback to legacy evaluation_pipeline datasets
            fallback = self.base_dir.parent / "evaluation_pipeline" / "datasets" / ("LIDC_real" if dataset == "lidc" else "EMIDEC_real")
            if fallback.exists():
                print(f"⚠️ real_data_dir not found at {real_data_dir}. Falling back to {fallback}")
                real_data_dir = fallback
                
        if method == "baseline":
            # Baseline uses only real data
            self._ensure_split_files(real_data_dir)
            return real_data_dir.resolve()
            
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
                    
        # Ensure split files exist in combined dir
        self._ensure_split_files(combined_dir)
        return combined_dir.resolve()

    def _ensure_split_files(self, data_root: Path):
        """Create train/val split files required by DiffTumor if missing.
        Format: each line 'imagesTr/xxx.nii.gz labelsTr/xxx.nii.gz' (relative paths).
        """
        train_txt = data_root / "real_liver_train_0.txt"
        val_txt = data_root / "real_liver_val_0.txt"
        images_dir = data_root / "imagesTr"
        labels_dir = data_root / "labelsTr"
        if not images_dir.exists() or not labels_dir.exists():
            return
        if train_txt.exists() and val_txt.exists():
            return
        # Build pairs by iterating labels and finding best-matching image
        label_files = sorted([p for p in labels_dir.rglob("*.nii.gz") if p.is_file()])
        pairs = []
        for lbl in label_files:
            lbl_name = lbl.name
            has_syn = lbl_name.startswith("syn_")
            core = lbl_name[4:] if has_syn else lbl_name
            candidates = set()
            # Common mappings
            if "GenMask_" in core:
                candidates.add(core.replace("GenMask_", "CVol_"))
                candidates.add(core.replace("GenMask_", "Vol_"))
                candidates.add(core.replace("GenMask_", ""))
            if "_Mask_" in core:
                candidates.add(core.replace("_Mask_", "_CVol_"))
                candidates.add(core.replace("_Mask_", "_Vol_"))
                candidates.add(core.replace("_Mask_", ""))
            # Fallback to same basename
            candidates.add(core)
            # Reapply syn_ prefix for image search if label had it
            images_to_try = [ ("syn_" + c) if has_syn else c for c in candidates ]
            img_path = None
            for cand in images_to_try:
                test_path = images_dir / cand
                if test_path.exists():
                    img_path = test_path
                    break
            if img_path is not None:
                pairs.append((f"/imagesTr/{img_path.name}", f"/labelsTr/{lbl.name}"))
        if not pairs:
            return
        split_idx = max(1, int(0.8 * len(pairs)))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        with open(train_txt, "w") as f:
            for a, b in train_pairs:
                f.write(f"{a} {b}\n")
        with open(val_txt, "w") as f:
            for a, b in val_pairs:
                f.write(f"{a} {b}\n")
                
    def _add_synthetic_data(self, source_dir, target_dir):
        """Add synthetic data to combined directory"""
        # Copy synthetic images (recursively, files only)
        images_root = source_dir / "imagesTr"
        if images_root.exists():
            for img in images_root.rglob("*.nii.gz"):
                if img.is_file():
                    new_name = f"syn_{img.name}"
                    shutil.copy2(img, target_dir / "imagesTr" / new_name)
                
        # Copy synthetic labels (recursively, files only)
        labels_root = source_dir / "labelsTr"
        if labels_root.exists():
            for label in labels_root.rglob("*.nii.gz"):
                if label.is_file():
                    new_name = f"syn_{label.name}"
                    shutil.copy2(label, target_dir / "labelsTr" / new_name)
                
    def _run_streaming(self, cmd, cwd: str | None, timeout: int | None, log_path: Path | None) -> int:
        """Run a subprocess, streaming stdout/stderr in real-time and tee to log file if provided."""
        import subprocess
        import threading
        from queue import Queue, Empty
        
        def enqueue_output(stream, queue):
            for line in iter(stream.readline, ''):
                queue.put(line)
            stream.close()
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
            q: Queue[str] = Queue()
            t = threading.Thread(target=enqueue_output, args=(process.stdout, q))
            t.daemon = True
            t.start()
            log_f = None
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_f = open(log_path, 'a', buffering=1)
            start_time = time.time()
            while True:
                if timeout and (time.time() - start_time) > timeout:
                    process.kill()
                    if log_f:
                        log_f.write("\n⏰ Timeout reached. Process killed.\n")
                        log_f.close()
                    return -1
                try:
                    line = q.get_nowait()
                except Empty:
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue
                print(line, end='')
                if log_f:
                    log_f.write(line)
            ret = process.wait()
            if log_f:
                log_f.flush()
                log_f.close()
            return ret
        except Exception as e:
            print(f"❌ Error running command: {e}")
            return -1

    def train_model(self, dataset, method, model_type, seg_model, checkpoint_path: Path | None = None):
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
        
        # Build training command (absolute path to avoid CWD issues)
        diff_tumor_main = (self.base_dir.parent / "evaluation_pipeline" / "DiffTumor" / "STEP3.SegmentationModel" / "main.py").resolve()
        cmd = [
            "python", str(diff_tumor_main),
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
        # Pass checkpoint to resume if provided
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            cmd.extend(["--checkpoint", str(Path(checkpoint_path).resolve())])
            
        # Add dataset-specific parameters
        if dataset == "emidec":
            cmd.extend(["--tumor_type", "cardiac", "--organ_type", "heart"])
            
        # Execute training
        try:
            print(f"💻 Running: {' '.join(cmd[:5])}...")
            # Stream logs to console and file in output_dir
            log_path = output_dir / "training_log.txt"
            rc = self._run_streaming(
                cmd=cmd,
                cwd=str((self.base_dir.parent / "evaluation_pipeline").resolve()),
                timeout=14400,
                log_path=log_path,
            )
            if rc == 0:
                print(f"✅ Training completed successfully")
                # Consider training successful; completion will be decided in caller by epoch count
                model_file = output_dir / "best_metric_model.pth"
                if model_file.exists():
                    print(f"✅ Model saved: {model_file}")
                else:
                    print(f"⚠️ Model file not found (continuing)")
                return True
            else:
                print(f"❌ Training failed (exit {rc})")
                print(f"📄 See log: {log_path}")
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
                    
                    # Determine output dir and checkpoint status
                    output_dir = self.get_model_output_path(dataset, method, model_type, seg_model)
                    checkpoint_path, last_epoch = self._get_latest_checkpoint_info(output_dir)
                    max_epochs = int(self.config['training']['max_epochs'])

                    if resume:
                        # If already finished, mark completed and skip
                        if last_epoch >= max_epochs - 1:
                            print(f"\n[{current}/{total_configs}] ✅ Skipping {checkpoint_key} - already reached {last_epoch+1}/{max_epochs} epochs")
                            checkpoint[checkpoint_key] = {
                                'completed': True,
                                'timestamp': datetime.now().isoformat(),
                                'elapsed_time': checkpoint.get(checkpoint_key, {}).get('elapsed_time', 0)
                            }
                            self.save_checkpoint(checkpoint)
                            continue

                    # Check if synthetic data exists (if needed)
                    if method != "baseline":
                        synthetic_dir = self.base_dir / "synthetic_data" / dataset / model_type / method
                        if not synthetic_dir.exists():
                            print(f"⚠️ Synthetic data not found: {synthetic_dir}")
                            print(f"   Please run synthetic generation first")
                            success = False
                        else:
                            success = self.train_model(dataset, method, model_type, seg_model, checkpoint_path=checkpoint_path if resume else None)
                    else:
                        success = self.train_model(dataset, method, model_type, seg_model, checkpoint_path=checkpoint_path if resume else None)
                        
                    # Update checkpoint
                    elapsed_time = time.time() - start_time
                    # Recompute progress after this run
                    _, updated_epoch = self._get_latest_checkpoint_info(output_dir)
                    finished = updated_epoch >= max_epochs - 1
                    checkpoint[checkpoint_key] = {
                        'completed': bool(success and finished),
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
                        choices=["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask", "all"],
                        default=["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"],
                        help="Methods to train (default: all)")
    parser.add_argument("--model-types", nargs="+",
                        choices=["pretrained", "from_scratch", "all"],
                        default=["pretrained", "from_scratch"],
                        help="Model types to use (default: all)")
    parser.add_argument("--seg-models", nargs="+",
                        choices=["nnunet", "swinunetr", "all"],
                        default=["nnunet", "swinunetr"],
                        help="Segmentation models to train (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--config", default="../configs/experiment_config.yaml",
                        help="Path to config file")
    
    args = parser.parse_args()
    
    trainer = SegmentationTrainer(args.config)
    
    # Process datasets
    datasets = ["lidc", "emidec"] if args.dataset == "all" else [args.dataset]
    
    # Expand 'all' selections
    all_methods = ["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"]
    all_model_types = ["pretrained", "from_scratch"]
    all_seg_models = ["nnunet", "swinunetr"]
    
    methods = all_methods if (not args.methods or "all" in args.methods) else args.methods
    model_types = all_model_types if (not args.model_types or "all" in args.model_types) else args.model_types
    seg_models = all_seg_models if (not args.seg_models or "all" in args.seg_models) else args.seg_models
    
    for dataset in datasets:
        trainer.train_all(
            dataset=dataset,
            methods=methods,
            model_types=model_types,
            seg_models=seg_models,
            resume=args.resume
        )

if __name__ == "__main__":
    main() 