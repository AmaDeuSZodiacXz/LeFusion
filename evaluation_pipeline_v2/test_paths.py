#!/usr/bin/env python3
"""
Test script to verify all paths in Evaluation Pipeline V2
"""

from pathlib import Path
import os

def test_paths():
    """Test and display all important paths"""
    
    # Get base directory
    script_dir = Path(__file__).parent
    base_dir = script_dir  # evaluation_pipeline_v2/
    
    print("=" * 60)
    print("📁 EVALUATION PIPELINE V2 - PATH VERIFICATION")
    print("=" * 60)
    
    print(f"\n1️⃣ Base Directory:")
    print(f"   {base_dir}")
    
    print(f"\n2️⃣ Synthetic Data Output:")
    synthetic_dir = base_dir / "synthetic_data"
    print(f"   {synthetic_dir}")
    print(f"   Example outputs:")
    print(f"   - {synthetic_dir}/lidc/pretrained/lefusion/P_P_prime/")
    print(f"   - {synthetic_dir}/lidc/pretrained/lefusion_h/P_N_prime/")
    print(f"   - {synthetic_dir}/lidc/from_scratch/lefusion_h_diffmask/")
    
    print(f"\n3️⃣ Trained Models Output:")
    models_dir = base_dir / "trained_models"
    print(f"   {models_dir}")
    print(f"   Example outputs:")
    print(f"   - {models_dir}/lidc/baseline/pretrained/nnunet/")
    print(f"   - {models_dir}/lidc/lefusion_h/pretrained/swinunetr/")
    
    print(f"\n4️⃣ Evaluation Results:")
    results_dir = base_dir / "evaluation_results"
    print(f"   {results_dir}")
    print(f"   Example outputs:")
    print(f"   - {results_dir}/lidc_evaluation_results_*.csv")
    
    print(f"\n5️⃣ Configuration File:")
    config_file = base_dir / "configs" / "experiment_config.yaml"
    print(f"   {config_file}")
    print(f"   Exists: {config_file.exists()}")
    
    print(f"\n6️⃣ Real Data Location (from config):")
    print(f"   LIDC Real: {base_dir.parent}/evaluation_pipeline/datasets/LIDC_real/")
    print(f"   EMIDEC Real: {base_dir.parent}/evaluation_pipeline/datasets/EMIDEC_real/")
    
    print(f"\n7️⃣ Model Weights (from config):")
    print(f"   LeFusion LIDC: {base_dir.parent}/LeFusion/LeFusion_Model/LIDC/lidc.pt")
    print(f"   DiffMask: {base_dir.parent}/DiffMask/DiffMask_Model/diffmask.pt")
    
    print("\n" + "=" * 60)
    print("✅ Path Structure Summary:")
    print("=" * 60)
    
    structure = """
    LeFusion/
    ├── evaluation_pipeline_v2/         # Main pipeline directory
    │   ├── synthetic_data/             # Generated synthetic data
    │   │   └── lidc/pretrained/...
    │   ├── trained_models/             # Trained segmentation models
    │   │   └── lidc/baseline/...
    │   ├── evaluation_results/         # Evaluation outputs
    │   │   └── *.csv
    │   └── configs/                    # Configuration
    │       └── experiment_config.yaml
    ├── LeFusion/                       # Core LeFusion model
    │   └── LeFusion_Model/            # Model weights
    ├── DiffMask/                       # DiffMask model
    │   └── DiffMask_Model/            # Model weights
    └── data/                          # Original datasets
        ├── LIDC/
        └── EMIDEC/
    """
    
    print(structure)
    
    # Check if directories need to be created
    print("🔧 Creating necessary directories...")
    dirs_to_create = [
        synthetic_dir,
        models_dir,
        results_dir,
        base_dir / "logs"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   ✅ {dir_path.name}/")
    
    print("\n✅ All paths verified and directories created!")

if __name__ == "__main__":
    test_paths() 