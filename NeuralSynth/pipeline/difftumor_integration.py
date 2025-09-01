"""
DiffTumor Framework Integration for NeuralSynth
================================================
Integrates with LeFusion's DiffTumor segmentation training framework.
Handles both nnU-Net and SwinUNETR architectures.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess

# Add DiffTumor to path
DIFFTUMOR_PATH = Path(__file__).parent.parent.parent / "utility_training_resources" / "DiffTumor" / "STEP3.SegmentationModel"
sys.path.insert(0, str(DIFFTUMOR_PATH))

class DiffTumorIntegration:
    """
    Integrates NeuralSynth with DiffTumor segmentation framework.
    Follows LeFusion's evaluation pipeline structure.
    """
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f) if config_path.endswith('.json') else yaml.safe_load(f)
        
        self.difftumor_path = DIFFTUMOR_PATH
        self.setup_paths()
    
    def setup_paths(self):
        """Setup paths for DiffTumor integration."""
        self.base_dir = Path(self.config.get('base_dir', '/Users/skb/Documents/LeFusion'))
        self.neuralsynth_dir = self.base_dir / "NeuralSynth"
        self.synthetic_dir = self.neuralsynth_dir / "synthetic_data"
        self.segmentation_dir = self.neuralsynth_dir / "segmentation_models"
        self.results_dir = self.neuralsynth_dir / "evaluation_results"
        
        # Create directories
        for dir_path in [self.synthetic_dir, self.segmentation_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def prepare_difftumor_data(self, dataset: str, method: str) -> Dict[str, Path]:
        """
        Prepare data in DiffTumor format.
        
        Compatible with LeFusion's data structure:
        - P (pathological only)
        - P_P_prime (pathological + synthetic from pathological)
        - P_N_prime (pathological + synthetic from normal)
        - P_P_prime_N_double_prime (all combined)
        """
        data_paths = {}
        
        # Base paths
        real_data = self.base_dir / "data" / dataset.upper()
        synthetic_base = self.synthetic_dir / dataset / method
        
        # Prepare different combinations
        combinations = {
            'P': self._prepare_P_only(real_data, dataset),
            'P_P_prime': self._prepare_P_P_prime(real_data, synthetic_base, dataset),
            'P_N_prime': self._prepare_P_N_prime(real_data, synthetic_base, dataset),
            'P_P_prime_N_double_prime': self._prepare_all_combined(real_data, synthetic_base, dataset)
        }
        
        for combo_name, combo_path in combinations.items():
            if combo_path.exists():
                data_paths[combo_name] = combo_path
                print(f"✓ Prepared {combo_name}: {combo_path}")
        
        return data_paths
    
    def _prepare_P_only(self, real_data: Path, dataset: str) -> Path:
        """Prepare pathological only data."""
        output_dir = self.segmentation_dir / dataset / "P_only"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy real pathological data
        pathological_dir = real_data / "pathological"
        if pathological_dir.exists():
            for file in pathological_dir.glob("*.npz"):
                shutil.copy2(file, output_dir / file.name)
        
        # Create data list for DiffTumor
        self._create_data_list(output_dir, dataset)
        return output_dir
    
    def _prepare_P_P_prime(self, real_data: Path, synthetic_base: Path, dataset: str) -> Path:
        """Prepare pathological + synthetic from pathological."""
        output_dir = self.segmentation_dir / dataset / "P_P_prime"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy real pathological
        pathological_dir = real_data / "pathological"
        if pathological_dir.exists():
            for file in pathological_dir.glob("*.npz"):
                shutil.copy2(file, output_dir / f"real_{file.name}")
        
        # Copy synthetic from pathological (P_P_prime)
        synthetic_p = synthetic_base / "P_P_prime"
        if synthetic_p.exists():
            for file in synthetic_p.glob("*.npz"):
                shutil.copy2(file, output_dir / f"synth_p_{file.name}")
        
        self._create_data_list(output_dir, dataset)
        return output_dir
    
    def _prepare_P_N_prime(self, real_data: Path, synthetic_base: Path, dataset: str) -> Path:
        """Prepare pathological + synthetic from normal."""
        output_dir = self.segmentation_dir / dataset / "P_N_prime"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy real pathological
        pathological_dir = real_data / "pathological"
        if pathological_dir.exists():
            for file in pathological_dir.glob("*.npz"):
                shutil.copy2(file, output_dir / f"real_{file.name}")
        
        # Copy synthetic from normal (P_N_prime) - NeuralSynth's main output
        synthetic_n = synthetic_base / "P_N_prime"
        if synthetic_n.exists():
            for file in synthetic_n.glob("*.npz"):
                shutil.copy2(file, output_dir / f"synth_n_{file.name}")
        
        self._create_data_list(output_dir, dataset)
        return output_dir
    
    def _prepare_all_combined(self, real_data: Path, synthetic_base: Path, dataset: str) -> Path:
        """Prepare all data combined."""
        output_dir = self.segmentation_dir / dataset / "P_P_prime_N_double_prime"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy real pathological
        pathological_dir = real_data / "pathological"
        if pathological_dir.exists():
            for file in pathological_dir.glob("*.npz"):
                shutil.copy2(file, output_dir / f"real_{file.name}")
        
        # Copy all synthetic variations
        for synth_type in ["P_P_prime", "P_N_prime", "P_N_double_prime"]:
            synthetic_dir = synthetic_base / synth_type
            if synthetic_dir.exists():
                for file in synthetic_dir.glob("*.npz"):
                    shutil.copy2(file, output_dir / f"synth_{synth_type}_{file.name}")
        
        self._create_data_list(output_dir, dataset)
        return output_dir
    
    def _create_data_list(self, data_dir: Path, dataset: str):
        """Create train/val splits for DiffTumor."""
        all_files = sorted(data_dir.glob("*.npz"))
        n_files = len(all_files)
        
        # 80/20 train/val split
        n_train = int(0.8 * n_files)
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:]
        
        # Write train list
        train_list = data_dir / "train_files.txt"
        with open(train_list, 'w') as f:
            for file in train_files:
                f.write(f"{file.stem}\n")
        
        # Write val list
        val_list = data_dir / "val_files.txt"
        with open(val_list, 'w') as f:
            for file in val_files:
                f.write(f"{file.stem}\n")
        
        print(f"  Created splits: {n_train} train, {len(val_files)} val")
    
    def run_difftumor_training(
        self,
        dataset: str,
        method: str,
        combination: str,
        seg_model: str = "nnunet",
        epochs: int = 200
    ) -> Path:
        """
        Run DiffTumor segmentation training.
        
        Args:
            dataset: 'lidc' or 'emidec'
            method: 'neuralsynth', 'lefusion', 'lefusion_h', 'lefusion_h_diffmask'
            combination: 'P', 'P_P_prime', 'P_N_prime', 'P_P_prime_N_double_prime'
            seg_model: 'nnunet' or 'swinunetr'
            epochs: Number of training epochs
        """
        # Prepare data
        data_paths = self.prepare_difftumor_data(dataset, method)
        
        if combination not in data_paths:
            raise ValueError(f"Combination {combination} not available")
        
        data_path = data_paths[combination]
        
        # Prepare DiffTumor config
        difftumor_config = self._create_difftumor_config(
            dataset, method, combination, seg_model, data_path, epochs
        )
        
        config_path = self.segmentation_dir / f"difftumor_config_{dataset}_{method}_{combination}.json"
        with open(config_path, 'w') as f:
            json.dump(difftumor_config, f, indent=2)
        
        # Run DiffTumor training
        model_save_path = self.segmentation_dir / dataset / method / combination / seg_model
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Command to run DiffTumor
        cmd = [
            "python", str(self.difftumor_path / "main.py"),
            "--config", str(config_path),
            "--output", str(model_save_path)
        ]
        
        print(f"\n🎯 Running DiffTumor training:")
        print(f"   Dataset: {dataset}")
        print(f"   Method: {method}")
        print(f"   Combination: {combination}")
        print(f"   Model: {seg_model}")
        print(f"   Command: {' '.join(cmd)}")
        
        # Run training
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Training completed: {model_save_path}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Training failed: {e}")
            return None
        
        return model_save_path
    
    def _create_difftumor_config(
        self,
        dataset: str,
        method: str,
        combination: str,
        seg_model: str,
        data_path: Path,
        epochs: int
    ) -> Dict:
        """Create DiffTumor configuration."""
        config = {
            "dataset": dataset.upper(),
            "method": method,
            "combination": combination,
            "model": seg_model,
            "data_path": str(data_path),
            "train_list": str(data_path / "train_files.txt"),
            "val_list": str(data_path / "val_files.txt"),
            "epochs": epochs,
            "batch_size": 4 if dataset == "lidc" else 2,
            "learning_rate": 1e-4,
            "optimizer": "AdamW",
            "loss": "DiceCELoss",
            "num_classes": 2,  # background + lesion
            "patch_size": [64, 64, 32] if dataset == "lidc" else [72, 72, 10],
            "spacing": [1.0, 1.0, 1.0],
            "cache_rate": 0.5,
            "num_workers": 4,
            "val_interval": 5,
            "save_checkpoint": True,
            "deterministic": True,
            "seed": 42
        }
        
        # Model-specific settings
        if seg_model == "nnunet":
            config.update({
                "model_name": "nnUNet",
                "in_channels": 1,
                "out_channels": 2,
                "feature_size": 48,
                "num_levels": 5,
                "max_features": 320,
                "norm_name": "instance"
            })
        elif seg_model == "swinunetr":
            config.update({
                "model_name": "SwinUNETR",
                "in_channels": 1,
                "out_channels": 2,
                "feature_size": 48,
                "depths": [2, 2, 2, 2],
                "num_heads": [3, 6, 12, 24],
                "norm_name": "instance",
                "drop_rate": 0.0,
                "attn_drop_rate": 0.0,
                "dropout_path_rate": 0.0,
                "use_checkpoint": True
            })
        
        return config
    
    def evaluate_with_difftumor(
        self,
        dataset: str,
        method: str,
        combination: str,
        seg_model: str,
        checkpoint: Optional[str] = None
    ) -> Dict:
        """
        Evaluate segmentation model using DiffTumor metrics.
        Returns DICE and NSD scores matching LeFusion paper.
        """
        model_path = self.segmentation_dir / dataset / method / combination / seg_model
        
        if checkpoint:
            model_file = model_path / checkpoint
        else:
            # Use best checkpoint
            model_file = model_path / "best_model.pt"
        
        if not model_file.exists():
            print(f"✗ Model not found: {model_file}")
            return {}
        
        # Prepare test data
        test_data = self._prepare_test_data(dataset)
        
        # Run evaluation
        results = self._run_evaluation(model_file, test_data, dataset, seg_model)
        
        # Save results
        results_file = self.results_dir / f"{dataset}_{method}_{combination}_{seg_model}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Evaluation Results:")
        print(f"   DICE: {results.get('dice', 0):.4f}")
        print(f"   NSD: {results.get('nsd', 0):.4f}")
        print(f"   Results saved: {results_file}")
        
        return results
    
    def _prepare_test_data(self, dataset: str) -> Path:
        """Prepare test data for evaluation."""
        # Use real test split from utility_training_resources
        test_split_path = self.base_dir / "utility_training_resources" / "datasets" / f"{dataset.upper()}_real"
        
        if dataset == "lidc":
            test_list = test_split_path / "real_lung_val_0.txt"
        else:
            test_list = test_split_path / "real_cardiac_val_0.txt"
        
        return test_list
    
    def _run_evaluation(
        self,
        model_file: Path,
        test_data: Path,
        dataset: str,
        seg_model: str
    ) -> Dict:
        """Run evaluation and compute metrics."""
        # This would integrate with DiffTumor's evaluation script
        # For now, return placeholder results
        results = {
            "dataset": dataset,
            "model": seg_model,
            "dice": 0.0,
            "nsd": 0.0,
            "hd95": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0
        }
        
        # In real implementation, would call DiffTumor's evaluation
        # cmd = ["python", str(self.difftumor_path / "evaluate.py"), ...]
        
        return results
    
    def compare_with_lefusion(self, dataset: str, seg_model: str = "nnunet") -> Dict:
        """
        Compare NeuralSynth results with LeFusion paper benchmarks.
        """
        # LeFusion paper results (Table 1 for LIDC, Table 2 for EMIDEC)
        lefusion_benchmarks = {
            "lidc": {
                "nnunet": {
                    "baseline": {"dice": 78.26, "nsd": 88.90},
                    "lefusion": {"dice": 78.77, "nsd": 89.25},
                    "lefusion_h": {"dice": 80.62, "nsd": 90.90},
                    "lefusion_h_diffmask": {"dice": 83.44, "nsd": 93.35}
                },
                "swinunetr": {
                    "baseline": {"dice": 78.38, "nsd": 88.67},
                    "lefusion": {"dice": 78.43, "nsd": 88.54},
                    "lefusion_h": {"dice": 80.95, "nsd": 90.98},
                    "lefusion_h_diffmask": {"dice": 83.13, "nsd": 93.20}
                }
            },
            "emidec": {
                "nnunet": {
                    "baseline": {"mi_dice": 68.61, "pmo_dice": 36.32},
                    "lefusion": {"mi_dice": 69.88, "pmo_dice": 34.79},
                    "lefusion_h": {"mi_dice": 69.95, "pmo_dice": 38.01},
                    "lefusion_h_diffmask": {"mi_dice": 71.28, "pmo_dice": 43.41}
                },
                "swinunetr": {
                    "baseline": {"mi_dice": 57.79, "pmo_dice": 35.76},
                    "lefusion": {"mi_dice": 57.85, "pmo_dice": 35.63},
                    "lefusion_h": {"mi_dice": 59.61, "pmo_dice": 37.99},
                    "lefusion_h_diffmask": {"mi_dice": 59.30, "pmo_dice": 42.49}
                }
            }
        }
        
        # Get NeuralSynth results
        neuralsynth_results = {}
        for combination in ["P", "P_P_prime", "P_N_prime", "P_P_prime_N_double_prime"]:
            results = self.evaluate_with_difftumor(
                dataset, "neuralsynth", combination, seg_model
            )
            neuralsynth_results[combination] = results
        
        # Compare
        comparison = {
            "dataset": dataset,
            "model": seg_model,
            "lefusion_benchmarks": lefusion_benchmarks[dataset][seg_model],
            "neuralsynth_results": neuralsynth_results,
            "improvements": {}
        }
        
        # Calculate improvements
        best_lefusion = lefusion_benchmarks[dataset][seg_model]["lefusion_h_diffmask"]
        best_neuralsynth = neuralsynth_results.get("P_P_prime_N_double_prime", {})
        
        if best_neuralsynth:
            if dataset == "lidc":
                comparison["improvements"]["dice"] = best_neuralsynth.get("dice", 0) - best_lefusion["dice"]
                comparison["improvements"]["nsd"] = best_neuralsynth.get("nsd", 0) - best_lefusion["nsd"]
            else:
                comparison["improvements"]["mi_dice"] = best_neuralsynth.get("mi_dice", 0) - best_lefusion["mi_dice"]
                comparison["improvements"]["pmo_dice"] = best_neuralsynth.get("pmo_dice", 0) - best_lefusion["pmo_dice"]
        
        # Save comparison
        comparison_file = self.results_dir / f"comparison_{dataset}_{seg_model}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n📈 Comparison with LeFusion:")
        print(f"   Dataset: {dataset}")
        print(f"   Model: {seg_model}")
        print(f"   LeFusion best: {best_lefusion}")
        print(f"   NeuralSynth best: {best_neuralsynth}")
        print(f"   Improvements: {comparison['improvements']}")
        
        return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DiffTumor Integration for NeuralSynth")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument("--dataset", type=str, choices=["lidc", "emidec"], required=True)
    parser.add_argument("--method", type=str, default="neuralsynth")
    parser.add_argument("--combination", type=str, default="P_P_prime_N_double_prime")
    parser.add_argument("--seg-model", type=str, choices=["nnunet", "swinunetr"], default="nnunet")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--compare", action="store_true", help="Compare with LeFusion")
    parser.add_argument("--epochs", type=int, default=200)
    
    args = parser.parse_args()
    
    # Initialize integration
    integration = DiffTumorIntegration(args.config)
    
    # Run requested operations
    if args.train:
        print(f"\n🚀 Starting DiffTumor training integration...")
        integration.run_difftumor_training(
            args.dataset,
            args.method,
            args.combination,
            args.seg_model,
            args.epochs
        )
    
    if args.evaluate:
        print(f"\n📊 Running evaluation...")
        integration.evaluate_with_difftumor(
            args.dataset,
            args.method,
            args.combination,
            args.seg_model
        )
    
    if args.compare:
        print(f"\n📈 Comparing with LeFusion benchmarks...")
        integration.compare_with_lefusion(args.dataset, args.seg_model)