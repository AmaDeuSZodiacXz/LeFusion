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
from scipy.ndimage import distance_transform_edt, zoom
import torch
from monai.metrics import DiceMetric
import re
import subprocess
import shutil
import csv

# Add path for surface_distance library
sys.path.append('../../evaluation_pipeline/DiffTumor/STEP3.SegmentationModel/external/surface-distance')
try:
    from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
except ImportError:
    print("Warning: surface_distance library not found. NSD metrics may not work.")

class ModelEvaluator:
    def __init__(self, config_path="../configs/experiment_config.yaml", log_to_file=True):
        """Initialize model evaluator with config"""
        # Get the evaluation_pipeline_v2 directory as base
        script_dir = Path(__file__).parent  # evaluation/
        self.base_dir = script_dir.parent   # evaluation_pipeline_v2/
        
        # Load config with proper path resolution
        config_full_path = self.base_dir / "configs" / "experiment_config.yaml"
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set directories relative to evaluation_pipeline_v2 using config
        output_cfg = (self.config or {}).get("output", {})
        evaluation_results_dir_name = output_cfg.get("evaluation_results", "evaluation_results")
        self.results_dir = self.base_dir / evaluation_results_dir_name
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup logging
        self.log_to_file = log_to_file
        self.log_file = None
        self.log_file_handle = None
        if self.log_to_file:
            self._setup_logging()
        
        self._print(f"📁 Base directory: {self.base_dir}")
    
    def _setup_logging(self):
        """Setup log file for this evaluation run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.results_dir / "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = log_dir / f"evaluation_log_{timestamp}.txt"
        self.log_file_handle = open(self.log_file, 'w', buffering=1)
        self._print(f"📝 Logging to: {self.log_file}")
        self._print(f"Evaluation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._print("="*80)
    
    def _print(self, *args, **kwargs):
        """Print to both console and log file"""
        # Print to console
        print(*args, **kwargs)
        # Write to log file if enabled
        if self.log_file_handle:
            # Convert args to string
            output = ' '.join(str(arg) for arg in args)
            self.log_file_handle.write(output + '\n')
            self.log_file_handle.flush()
    
    def __del__(self):
        """Cleanup log file handle"""
        if self.log_file_handle:
            self._print("="*80)
            self._print(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_file_handle.close()
        print(f"📁 Results directory: {self.results_dir}")
    
    def _resolve_path(self, maybe_rel_path: str) -> Path:
        p = Path(maybe_rel_path)
        if not p.is_absolute():
            p = (self.base_dir / p).resolve()
        return p
    
    def _get_checkpoint_epoch(self, ckpt_path: Path) -> int:
        try:
            ckp = torch.load(str(ckpt_path), map_location='cpu')
            e = ckp.get('epoch', -1)
            if isinstance(e, (int, float)):
                return int(e)
        except Exception:
            pass
        # Try parse from filename like epoch_XXXX.pt
        m = re.search(r'epoch_(\d+)\.pt$', ckpt_path.name)
        if m:
            return int(m.group(1))
        return -1

    def _select_real_data_dir(self, dataset: str) -> Path:
        """Pick an existing real_data_dir for dataset, trying config then known fallbacks."""
        cfg_path = self._resolve_path(self.config['datasets'][dataset]['real_data_dir'])
        if cfg_path.exists():
            return cfg_path
        # Known fallback under the legacy evaluation_pipeline
        real_name = 'LIDC_real' if dataset == 'lidc' else 'EMIDEC_real'
        candidates = [
            (self.base_dir.parent / 'evaluation_pipeline' / 'datasets' / real_name).resolve(),
            (self.base_dir / 'datasets' / real_name).resolve(),
        ]
        for cand in candidates:
            if cand.exists():
                print(f"ℹ️  Using fallback real_data_dir: {cand}")
                return cand
        # Return configured path even if missing; caller will error clearly
        return cfg_path
    
    def _stage_test_set_from_txt(self, dataset: str) -> Path | None:
        """Create a staging test set (imagesTs/labelsTs) based on test.txt for LIDC.
        Prefers sourcing from LIDC_real if available; falls back to original Pathological folders.
        Returns the staging directory path or None if failed.
        """
        if dataset != 'lidc':
            return None
        test_txt_conf = self.config['datasets'][dataset].get('test_file')
        if not test_txt_conf:
            return None
        test_txt = self._resolve_path(test_txt_conf)
        if not test_txt.exists():
            print(f"❌ test.txt not found: {test_txt}")
            return None
        # Determine source directories
        lidc_real = (self.base_dir.parent / 'evaluation_pipeline' / 'datasets' / 'LIDC_real').resolve()
        if (lidc_real / 'imagesTr').exists() and (lidc_real / 'labelsTr').exists():
            src_img = lidc_real / 'imagesTr'
            src_lbl = lidc_real / 'labelsTr'
        else:
            # Fallback to original dataset
            patho_base = self._resolve_path(self.config['datasets'][dataset]['pathological_dir'])
            src_img = patho_base / 'Image'
            src_lbl = patho_base / 'Mask'
        # Prepare staging dir
        staging_root = self.results_dir / f"{dataset}_staging_test"
        imagesTs = staging_root / 'imagesTs'
        labelsTs = staging_root / 'labelsTs'
        os.makedirs(imagesTs, exist_ok=True)
        os.makedirs(labelsTs, exist_ok=True)
        # Build file pairs from test.txt
        copied = 0
        with open(test_txt, 'r') as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                # Construct candidate filenames
                img_cands = [f"{name.replace('Vol_', 'CVol_')}.nii.gz", f"{name}.nii.gz"]
                lbl_name = f"{name.replace('Vol_', 'Mask_')}.nii.gz"
                img_src = None
                for cand in img_cands:
                    p = src_img / cand
                    if p.exists():
                        img_src = p
                        break
                lbl_src = src_lbl / lbl_name
                if img_src is None or not lbl_src.exists():
                    continue
                # Copy if not already staged
                img_dst = imagesTs / img_src.name
                lbl_dst = labelsTs / lbl_src.name
                if not img_dst.exists():
                    try:
                        shutil.copy2(img_src, img_dst)
                    except Exception:
                        pass
                if not lbl_dst.exists():
                    try:
                        shutil.copy2(lbl_src, lbl_dst)
                    except Exception:
                        pass
                copied += 1
        if copied == 0:
            print(f"❌ No test pairs could be staged from {test_txt}")
            return None
        print(f"✅ Staged {copied} test pairs to {staging_root}")
        return staging_root
    
    def _ensure_modelpt(self, model_dir: Path, checkpoint_epoch: int = None, use_best: bool = False) -> str:
        """Ensure model checkpoint exists and return the checkpoint name to use.
        
        Args:
            model_dir: Directory containing model checkpoints
            checkpoint_epoch: Specific epoch to load (e.g., 50, 100, 150)
            use_best: Whether to use best_metric_model.pt if available
            
        Returns:
            Name of checkpoint file to use
        """
        model_pt = model_dir / 'model.pt'
        
        # Priority 1: Use specific epoch if requested
        if checkpoint_epoch is not None:
            # Try different naming conventions (including zero-padded)
            epoch_candidates = [
                f'epoch_{checkpoint_epoch:04d}.pt',  # epoch_0010.pt (zero-padded)
                f'epoch_{checkpoint_epoch}.pt',       # epoch_10.pt
                f'model_epoch_{checkpoint_epoch}.pt',
                f'checkpoint_epoch_{checkpoint_epoch}.pt',
                f'model_{checkpoint_epoch}.pt',
                f'epoch_{checkpoint_epoch:03d}.pt',  # epoch_010.pt (3-digit padding)
            ]
            
            for cand_name in epoch_candidates:
                epoch_checkpoint = model_dir / cand_name
                if epoch_checkpoint.exists():
                    # Copy to model.pt for compatibility
                    try:
                        shutil.copy2(epoch_checkpoint, model_pt)
                        self._print(f"✅ Using checkpoint from epoch {checkpoint_epoch}: {cand_name}")
                        if checkpoint_epoch < 200:
                            self._print(f"⚠️  Warning: Training incomplete (epoch {checkpoint_epoch}/200) - results may not be optimal")
                        return cand_name
                    except Exception as e:
                        print(f"⚠️  Could not copy checkpoint: {e}")
            
            self._print(f"⚠️  Checkpoint for epoch {checkpoint_epoch} not found, falling back to default")
        
        # Priority 2: Use best checkpoint if requested
        if use_best:
            best_candidates = ['best_metric_model.pt', 'best_metric_model.pth']
            for cand in best_candidates:
                src = model_dir / cand
                if src.exists():
                    try:
                        shutil.copy2(src, model_pt)
                        print(f"✅ Using best checkpoint: {cand}")
                        return cand
                    except Exception as e:
                        print(f"⚠️  Could not copy best checkpoint: {e}")
        
        # Priority 3: Use existing model.pt if available
        if model_pt.exists():
            print(f"ℹ️  Using existing model.pt")
            return 'model.pt'
        
        # Priority 4: Try to find any available checkpoint
        for cand in ['model_final.pt', 'model.pt', 'best_metric_model.pth']:
            src = model_dir / cand
            if src.exists():
                try:
                    shutil.copy2(src, model_pt)
                    print(f"ℹ️  Copied {src.name} -> model.pt for validation")
                    return src.name
                except Exception as e:
                    print(f"⚠️  Could not prepare model.pt: {e}")
        
        # Last resort: Find any .pt file
        import glob
        checkpoints = list(model_dir.glob('*.pt'))
        if checkpoints:
            # Sort by modification time and use the latest
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest = checkpoints[0]
            try:
                shutil.copy2(latest, model_pt)
                print(f"ℹ️  Using latest checkpoint found: {latest.name}")
                
                # Check if training was incomplete
                if 'epoch' in latest.name:
                    import re
                    match = re.search(r'epoch[_]?(\d+)', latest.name)
                    if match:
                        epoch_num = int(match.group(1))
                        if epoch_num < 200:
                            print(f"⚠️  Warning: Training incomplete (epoch {epoch_num}/200) - results may not be optimal")
                
                return latest.name
            except Exception as e:
                print(f"⚠️  Could not copy checkpoint: {e}")
        
        self._print(f"❌ No checkpoint found in {model_dir}")
        return None
    
    def _ensure_val_pseudo_labels(self, data_root: Path, organ_type: str, tumor_type: str) -> None:
        """Create missing organ pseudo labels required by STEP3 validation by copying val labels."""
        try:
            val_list = data_root / f"real_{tumor_type}_val_0.txt"
            if not val_list.exists():
                return
            step3_root = (self.base_dir.parent / "evaluation_pipeline" / "DiffTumor" / "STEP3.SegmentationModel").resolve()
            pseudo_dir = step3_root / "organ_pseudo_swin_new" / organ_type
            pseudo_dir.mkdir(parents=True, exist_ok=True)
            with open(val_list, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    lbl_rel = parts[1].lstrip('/')
                    lbl_abs = (data_root / lbl_rel).resolve()
                    if not lbl_abs.exists():
                        continue
                    dest = pseudo_dir / os.path.basename(lbl_abs)
                    if not dest.exists():
                        try:
                            shutil.copy2(lbl_abs, dest)
                        except Exception:
                            pass
        except Exception:
            pass
    
    def _run_validation_generate_predictions(self, dataset: str, method: str, model_type: str, seg_model: str, data_root: Path, model_dir: Path, save_parent: Path, val_overlap: float = 0.75, checkpoint_epoch: int = None, use_best_checkpoint: bool = False) -> Path:
        """Invoke STEP3 validation.py to generate predictions and return the predictions directory path."""
        checkpoint_name = self._ensure_modelpt(model_dir, checkpoint_epoch=checkpoint_epoch, use_best=use_best_checkpoint)
        if checkpoint_name is None:
            print(f"❌ No checkpoint found for evaluation")
            return None
        # Ensure pseudo labels (names are only used for legacy val split)
        tumor_type = 'lung' if dataset == 'lidc' else 'cardiac'
        organ_type = 'lung' if dataset == 'lidc' else 'heart'
        self._ensure_val_pseudo_labels(data_root, organ_type=organ_type, tumor_type=tumor_type)
        # Build command
        validation_py = (self.base_dir.parent / 'evaluation_pipeline' / 'DiffTumor' / 'STEP3.SegmentationModel' / 'validation.py').resolve()
        cmd = [
            sys.executable, str(validation_py),
            '--data_root', str(data_root),
            '--datafold_dir', str(data_root),
            '--tumor_type', tumor_type,
            '--organ_type', organ_type,
            '--fold', '0',
            '--save_dir', str(save_parent),
            '--model', seg_model,
            '--val_overlap', str(val_overlap),
            '--checkpoint',
            '--log_dir', str(model_dir),
            '--use_test_set',
            '--disable_organ_override',
        ]
        # For LIDC, check if checkpoint is 3-channel or 2-channel
        if dataset == 'lidc':
            # Check the checkpoint architecture
            import torch
            checkpoint_path = model_dir / 'model.pt'
            if checkpoint_path.exists():
                try:
                    ckpt = torch.load(str(checkpoint_path), map_location='cpu')
                    # Check output channels in the checkpoint
                    is_3channel = False
                    for key, val in ckpt.get('state_dict', {}).items():
                        if 'output_block.conv.conv.weight' in key:
                            if val.shape[0] == 3:
                                is_3channel = True
                                self._print(f"⚠️  Detected 3-channel checkpoint (legacy). Using compatibility mode.")
                            break
                    
                    if not is_3channel:
                        # New 2-channel model
                        cmd.extend(['--num_classes', '2'])
                    else:
                        # Old 3-channel model - use default (3 classes)
                        self._print(f"ℹ️  Using 3-channel evaluation for legacy checkpoint")
                except Exception as e:
                    self._print(f"⚠️  Could not determine checkpoint architecture: {e}")
                    # Default to 2-channel for LIDC
                    cmd.extend(['--num_classes', '2'])
            else:
                # No checkpoint yet, assume 2-channel for LIDC
                cmd.extend(['--num_classes', '2'])
        # For LIDC 2-channel model: lesions are always in channel 1 (no longer need lesion_class_index)
        self._print(f"🧪 Generating predictions via validation.py: {' '.join(cmd[:6])} ...")
        step3_cwd = str((self.base_dir.parent / 'evaluation_pipeline' / 'DiffTumor' / 'STEP3.SegmentationModel').resolve())
        try:
            subprocess.run(cmd, cwd=step3_cwd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ validation.py failed: {e}")
        # Validation saves to save_dir/<model>/<val_overlap>/pred
        pred_dir = save_parent / seg_model / str(val_overlap) / 'pred'
        return pred_dir

    def _read_validation_metrics_csv(self, save_parent: Path, seg_model: str, val_overlap: float) -> dict | None:
        """Read tumor Dice/NSD from STEP3 validation metrics.csv; returns stats dict or None."""
        metrics_path = save_parent / seg_model / str(val_overlap) / 'metrics.csv'
        if not metrics_path.exists():
            return None
        tumor_dice_vals = []
        tumor_nsd_vals = []
        try:
            with open(metrics_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                # Expect header: name, organ_dice, organ_nsd, tumor_dice, tumor_nsd
                for row in reader:
                    if not row:
                        continue
                    if row[0].strip().lower() == 'average':
                        # skip the summary row; we recompute robustly
                        continue
                    # Robust parse of columns 3 and 4 as floats (they are decimals 0..1)
                    try:
                        td = float(row[3]) * 100.0
                        tn = float(row[4]) * 100.0
                        tumor_dice_vals.append(td)
                        tumor_nsd_vals.append(tn)
                    except Exception:
                        continue
        except Exception:
            return None
        if not tumor_dice_vals:
            return None
        import numpy as np
        return {
            'dice_mean': float(np.mean(tumor_dice_vals)),
            'dice_std': float(np.std(tumor_dice_vals)),
            'nsd_mean': float(np.mean(tumor_nsd_vals)) if tumor_nsd_vals else 0.0,
            'nsd_std': float(np.std(tumor_nsd_vals)) if tumor_nsd_vals else 0.0,
            'num_cases': len(tumor_dice_vals),
        }
        
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
        pred, _ = self.load_nifti(pred_path)
        gt, gt_spacing = self.load_nifti(gt_path)
        
        # If multi-class mask, reduce to tumor channel (value==2)
        if pred.max() > 1.5:
            pred = (pred == 2).astype(np.uint8)
        if gt.max() > 1.5:
            gt = (gt == 2).astype(np.uint8)
        
        # Align shapes by resampling prediction to GT shape (nearest neighbor)
        if pred.shape != gt.shape:
            scale = (
                gt.shape[0] / max(pred.shape[0], 1e-6),
                gt.shape[1] / max(pred.shape[1], 1e-6),
                gt.shape[2] / max(pred.shape[2], 1e-6),
            )
            pred = zoom(pred.astype(float), zoom=scale, order=0)  # nearest
        
        pred = (pred > 0.5).astype(np.uint8)
        gt = (gt > 0.5).astype(np.uint8)
         
        # Skip if GT has no tumor voxels
        if gt.sum() == 0:
            return None
        # Calculate metrics
        dice = self.calculate_dice(pred, gt)
        nsd = self.calculate_nsd(pred, gt, spacing_mm=gt_spacing, tolerance=self.config['evaluation']['nsd_tolerance'])
        
        return {
            'dice': dice,
            'nsd': nsd
        }
        
    def evaluate_model(self, dataset, method, model_type, seg_model, checkpoint_epoch=None, use_best_checkpoint=False):
        """Evaluate a trained model on test set
        
        Args:
            dataset: Dataset name (lidc or emidec)
            method: Method name (baseline, lefusion, etc.)
            model_type: Model type (pretrained or from_scratch)
            seg_model: Segmentation model (nnunet or swinunetr)
            checkpoint_epoch: Specific epoch to evaluate (e.g., 50, 100, 150)
            use_best_checkpoint: Whether to use best_metric_model.pt if available
        """
        print(f"\n📊 Evaluating {seg_model}")
        print(f"   Dataset: {dataset}")
        print(f"   Method: {method}")
        print(f"   Model Type: {model_type}")
        
        # Get model path
        model_dir = self.base_dir / "trained_models" / dataset / method / model_type / seg_model
        # Try multiple known filenames
        candidates = [
            model_dir / "best_metric_model.pth",
            model_dir / "model.pt",
            model_dir / "model_final.pt",
        ]
        model_path = next((p for p in candidates if p.exists()), None)
        # If still not found, pick the latest epoch_XXXX.pt
        if model_path is None:
            epoch_ckpts = sorted(model_dir.glob('epoch_*.pt'))
            if epoch_ckpts:
                # sort by numeric epoch
                def epoch_num(p):
                    name = p.stem  # epoch_XXXX
                    try:
                        return int(name.split('_')[-1])
                    except Exception:
                        return -1
                epoch_ckpts.sort(key=epoch_num)
                model_path = epoch_ckpts[-1]
        if model_path is None:
            print(f"⚠️ Model checkpoint not found in {model_dir}. Proceeding to look for predictions only.")
        else:
            print(f"📦 Using checkpoint: {model_path}")
        
        # Get test data
        test_data_dir = self._select_real_data_dir(dataset)
        # If LIDC and test.txt is available, stage test set into imagesTs/labelsTs
        if dataset == 'lidc':
            staged = self._stage_test_set_from_txt(dataset)
            if staged is not None:
                test_data_dir = staged
        test_images_dir = test_data_dir / "imagesTs"
        test_labels_dir = test_data_dir / "labelsTs"
        
        if not test_images_dir.exists():
            print(f"❌ Test images not found: {test_images_dir}")
            return None
            
        # Run inference (simplified - in practice you'd load the model and run predictions)
        predictions_root = self.results_dir / f"{dataset}_{method}_{model_type}_{seg_model}"
        predictions_dir = predictions_root / "predictions"
        os.makedirs(predictions_dir, exist_ok=True)
        
        # For now, we'll assume predictions are already generated
        # In practice, you'd run the model inference here
        
        # Collect all test cases
        test_cases = list(test_labels_dir.glob("*.nii.gz"))
        
        if len(test_cases) == 0:
            print(f"❌ No test cases found in {test_labels_dir}. Please provide the paper test set under imagesTs/labelsTs or a valid test.txt.")
            return None
            
        # Evaluate each case
        results = []
        for gt_path in test_cases:
            case_name = gt_path.stem.replace(".nii", "")
            pred_path = predictions_dir / gt_path.name
            
            if pred_path.exists():
                metrics = self.evaluate_single_case(pred_path, gt_path)
                if metrics is not None:
                    metrics['case'] = case_name
                    results.append(metrics)
            else:
                print(f"⚠️ Prediction not found for {case_name}")
                
        if len(results) == 0:
            # Try to auto-generate predictions using validation.py
            model_dir = self.base_dir / 'trained_models' / dataset / method / model_type / seg_model
            gen_pred_dir = self._run_validation_generate_predictions(
                dataset=dataset, method=method, model_type=model_type, seg_model=seg_model,
                data_root=test_data_dir, model_dir=model_dir, save_parent=predictions_root,
                val_overlap=0.75,
                checkpoint_epoch=checkpoint_epoch,
                use_best_checkpoint=use_best_checkpoint,
            )
            if gen_pred_dir is not None and gen_pred_dir.exists():
                # Prefer reading validator's metrics.csv for consistency with printed per-case values
                csv_stats = self._read_validation_metrics_csv(save_parent=predictions_root, seg_model=seg_model, val_overlap=0.75)
                if csv_stats is not None:
                    stats = {
                        'dataset': dataset,
                        'method': method,
                        'model_type': model_type,
                        'seg_model': seg_model,
                        'dice_mean': csv_stats['dice_mean'],
                        'dice_std': csv_stats['dice_std'],
                        'nsd_mean': csv_stats['nsd_mean'],
                        'nsd_std': csv_stats['nsd_std'],
                        'num_cases': csv_stats['num_cases'],
                    }
                    return stats
                # Fallback: compute metrics from saved masks if CSV not available
                predictions_dir = gen_pred_dir
                results = []
                for gt_path in test_cases:
                    case_name = gt_path.stem.replace(".nii", "")
                    pred_path = predictions_dir / f"{case_name}.nii.gz"
                    if pred_path.exists():
                        metrics = self.evaluate_single_case(pred_path, gt_path)
                        if metrics is not None:
                            metrics['case'] = case_name
                            results.append(metrics)
                if len(results) == 0:
                    print(f"❌ No predictions found to evaluate in {predictions_dir}")
                    return None
            else:
                print(f"❌ No predictions found to evaluate in {predictions_dir}")
                return None
        else:
            # No checkpoint found or predictions could not be generated
            print(f"❌ Cannot evaluate {method}+{model_type}+{seg_model}: No checkpoint available")
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
        self._print(f"\n{'='*80}")
        
        if dataset == "lidc":
            self._print("Table 1: Downstream Lung Nodule Segmentation Dice (↑) and NSD (↑) on LIDC")
        else:
            self._print("Table 2: Downstream Cardiac Lesion Segmentation Dice (↑) on EMIDEC")
            
        self._print("P: real pathological cases. P'/N': synthetic pathological cases")
        self._print("Bold numbers indicate the best performance in each setting")
        self._print("="*80)
        
        # Create formatted table
        self._print(f"\n{'Methods':<30} {'Training':<15} {'nnU-Net (2021)':<30} {'SwinUNETR (2021)':<30}")
        self._print(f"{'':30} {'Setting':<15} {'Dice (↑)  NSD (↑)':<30} {'Dice (↑)  NSD (↑)':<30}")
        self._print("-"*105)
        
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
                
            self._print(f"{method_name:<30} {training_setting:<15} {nnunet_str:<30} {swin_str:<30}")
            
        self._print("-"*105)
        
    def evaluate_all(self, dataset="lidc", methods=None, model_types=None, seg_models=None, checkpoint_epoch=None, use_best_checkpoint=False):
        """Evaluate all specified configurations
        
        Args:
            dataset: Dataset name
            methods: List of methods to evaluate
            model_types: List of model types
            seg_models: List of segmentation models
            checkpoint_epoch: Specific epoch to evaluate
            use_best_checkpoint: Whether to use best checkpoint
        """
        
        # Default values
        if methods is None:
            methods = ["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask"]
        if model_types is None:
            model_types = ["pretrained", "from_scratch"]
        if seg_models is None:
            seg_models = ["nnunet", "swinunetr"]
            
        self._print(f"\n{'='*60}")
        self._print(f"MODEL EVALUATION PIPELINE")
        self._print(f"Dataset: {dataset}")
        self._print(f"Methods: {methods}")
        self._print(f"Model Types: {model_types}")
        self._print(f"Segmentation Models: {seg_models}")
        self._print(f"{'='*60}")
        
        all_results = []
        
        for method in methods:
            for model_type in model_types:
                # Skip baseline for from_scratch
                if method == "baseline" and model_type == "from_scratch":
                    continue
                    
                for seg_model in seg_models:
                    self._print(f"\n{'='*50}")
                    self._print(f"Evaluating: {method} + {model_type} + {seg_model}")
                    self._print(f"{'='*50}")
                    
                    # Locate checkpoint and skip if not finished
                    model_dir = self.base_dir / "trained_models" / dataset / method / model_type / seg_model
                    # Try preferred file order
                    ckpt = None
                    for cand in ["model_final.pt", "model.pt", "best_metric_model.pth"]:
                        p = model_dir / cand
                        if p.exists():
                            ckpt = p
                            break
                    if ckpt is None:
                        # Try latest epoch_*.pt
                        epoch_ckpts = sorted(model_dir.glob('epoch_*.pt'))
                        if epoch_ckpts:
                            def epoch_num(p):
                                try:
                                    return int(p.stem.split('_')[-1])
                                except Exception:
                                    return -1
                            epoch_ckpts.sort(key=epoch_num)
                            ckpt = epoch_ckpts[-1]
                    # If specific epoch requested, let evaluate_model handle it
                    if checkpoint_epoch is not None:
                        self._print(f"🎯 Evaluating at specified epoch {checkpoint_epoch}")
                    elif use_best_checkpoint:
                        self._print(f"🎯 Using best checkpoint if available")
                    elif ckpt is not None:
                        last_epoch = self._get_checkpoint_epoch(ckpt)
                        max_epochs = int(self.config['training']['max_epochs'])
                        if last_epoch < max_epochs - 1:
                            self._print(f"⚠️  Warning: Training incomplete ({method}+{model_type}+{seg_model} at epoch {last_epoch+1}/{max_epochs})")
                            self._print(f"   Continuing with evaluation at epoch {last_epoch+1}...")
                    else:
                        self._print(f"⏭️  Skipping (no checkpoint): {method}+{model_type}+{seg_model}")
                        continue

                    stats = self.evaluate_model(dataset, method, model_type, seg_model, 
                                               checkpoint_epoch=checkpoint_epoch, 
                                               use_best_checkpoint=use_best_checkpoint)
                    
                    if stats:
                        all_results.append(stats)
                        self._print(f"✅ DICE: {stats['dice_mean']:.2f} ± {stats['dice_std']:.2f}")
                        self._print(f"✅ NSD: {stats['nsd_mean']:.2f} ± {stats['nsd_std']:.2f}")
                    else:
                        self._print(f"❌ Evaluation failed")
                        
        # Save results to CSV
        if all_results:
            df = pd.DataFrame(all_results)
            output_file = self.results_dir / f"{dataset}_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_file, index=False)
            self._print(f"\n✅ Results saved to: {output_file}")
            
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
                        choices=["baseline", "lefusion", "lefusion_h", "lefusion_h_diffmask", "all"],
                        help="Methods to evaluate (default: all)")
    parser.add_argument("--model-types", nargs="+",
                        choices=["pretrained", "from_scratch", "all"],
                        help="Model types to evaluate (default: all)")
    parser.add_argument("--seg-models", nargs="+",
                        choices=["nnunet", "swinunetr", "all"],
                        help="Segmentation models to evaluate (default: all)")
    parser.add_argument("--config", default="../configs/experiment_config.yaml",
                        help="Path to config file")
    parser.add_argument("--compare-paper", action="store_true",
                        help="Compare with paper results")
    parser.add_argument("--checkpoint-epoch", type=int, default=None,
                        help="Specific epoch checkpoint to evaluate (e.g., 50, 100, 150)")
    parser.add_argument("--use-best-checkpoint", action="store_true",
                        help="Use best_metric_model.pt if available instead of final checkpoint")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.config)
    
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
        results = evaluator.evaluate_all(
            dataset=dataset,
            methods=methods,
            model_types=model_types,
            seg_models=seg_models,
            checkpoint_epoch=args.checkpoint_epoch,
            use_best_checkpoint=args.use_best_checkpoint
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