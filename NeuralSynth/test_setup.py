#!/usr/bin/env python3
"""
Test NeuralSynth setup and verify integration with main repository
"""

import os
import sys
from pathlib import Path
import json

def test_directory_structure():
    """Test if all required directories exist."""
    print("Testing directory structure...")
    
    # Use relative paths from current file location
    neuralsynth_dir = Path(__file__).parent
    base_dir = neuralsynth_dir.parent
    
    required_dirs = [
        neuralsynth_dir / 'configs',
        neuralsynth_dir / 'models',
        neuralsynth_dir / 'synthetic_generation',
        neuralsynth_dir / 'training',
        neuralsynth_dir / 'evaluation',
        neuralsynth_dir / 'scripts',
        neuralsynth_dir / 'utils'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"  ✓ {dir_path.relative_to(base_dir)}")
        else:
            print(f"  ✗ {dir_path.relative_to(base_dir)} - NOT FOUND")
            all_exist = False
    
    return all_exist

def test_data_access():
    """Test access to main repository data."""
    print("\nTesting data access...")
    
    # Use relative path to data directory
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    datasets = {
        'LIDC': {
            'Normal/Image': 'Normal images',
            'Pathological/Image': 'Pathological images',
            'Pathological/Mask': 'Pathological masks'
        },
        'EMIDEC': {
            'Normal': 'Normal scans',
            'Pathological/images': 'Pathological images',
            'Pathological/labels': 'Pathological labels'
        }
    }
    
    all_exist = True
    for dataset, paths in datasets.items():
        dataset_dir = data_dir / dataset
        if dataset_dir.exists():
            print(f"  {dataset}:")
            for subpath, desc in paths.items():
                full_path = dataset_dir / subpath
                if full_path.exists():
                    # Count files
                    if full_path.is_dir():
                        count = len(list(full_path.glob('*.nii.gz')))
                        print(f"    ✓ {subpath}: {count} files")
                    else:
                        print(f"    ✓ {subpath}")
                else:
                    print(f"    ✗ {subpath} - NOT FOUND")
                    all_exist = False
        else:
            print(f"  ✗ {dataset} dataset not found")
            all_exist = False
    
    return all_exist

def test_evaluation_training_integration():
    """Test integration with evaluation_training pipeline."""
    print("\nTesting evaluation_training integration...")
    
    # Use relative path to evaluation_training
    base_dir = Path(__file__).parent.parent
    eval_training_dir = base_dir / 'evaluation_training'
    
    if eval_training_dir.exists():
        print(f"  ✓ evaluation_training directory found")
        
        # Check for key modules
        key_modules = [
            'training/train_models.py',
            'evaluation/evaluate.py',
            'synthetic_generation/generate_synthetic_data.py',
            'configs/training_config.py'
        ]
        
        for module in key_modules:
            module_path = eval_training_dir / module
            if module_path.exists():
                print(f"    ✓ {module}")
            else:
                print(f"    ⚠ {module} - Not found (may be named differently)")
        
        return True
    else:
        print(f"  ✗ evaluation_training directory not found")
        return False

def test_difftumor_integration():
    """Test DiffTumor framework availability."""
    print("\nTesting DiffTumor integration...")
    
    # Use relative path to DiffTumor
    base_dir = Path(__file__).parent.parent
    difftumor_path = base_dir / 'utility_training_resources' / 'DiffTumor' / 'STEP3.SegmentationModel'
    
    if difftumor_path.exists():
        print(f"  ✓ DiffTumor framework found")
        return True
    else:
        print(f"  ✗ DiffTumor framework not found")
        print(f"    Expected at: {difftumor_path}")
        return False

def test_config_files():
    """Test configuration files."""
    print("\nTesting configuration files...")
    
    # Use relative path to config
    neuralsynth_dir = Path(__file__).parent
    config_file = neuralsynth_dir / 'configs' / 'training_config.yaml'
    
    if config_file.exists():
        print(f"  ✓ training_config.yaml found")
        
        # Try to load it
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"    ✓ Configuration loaded successfully")
            print(f"    - Datasets: {list(config.get('datasets', {}).keys())}")
            print(f"    - Models: {list(config.get('models', {}).keys())}")
            return True
        except Exception as e:
            print(f"    ✗ Error loading config: {e}")
            return False
    else:
        print(f"  ✗ training_config.yaml not found")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("NeuralSynth Setup Test")
    print("="*60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Data Access", test_data_access),
        ("Evaluation Training Integration", test_evaluation_training_integration),
        ("DiffTumor Integration", test_difftumor_integration),
        ("Configuration Files", test_config_files)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! NeuralSynth is ready to use.")
        print("\nNext steps:")
        print("1. Generate synthetic data:")
        print("   cd NeuralSynth")
        print("   python synthetic_generation/generate_synthetic.py --dataset lidc")
        print("\n2. Train segmentation models:")
        print("   python training/train_segmentation.py --dataset lidc --combination P_N_prime")
        print("\n3. Or run complete pipeline:")
        print("   bash scripts/run_complete_pipeline.sh --dataset lidc")
    else:
        print("✗ Some tests failed. Please check the setup.")
    print("="*60)

if __name__ == '__main__':
    main()