"""
Path utilities for NeuralSynth
Provides consistent relative path handling across all modules
"""

import sys
from pathlib import Path

def setup_paths():
    """Setup Python paths for imports using relative paths."""
    # Get the NeuralSynth directory (parent of utils)
    neuralsynth_dir = Path(__file__).parent.parent
    
    # Get the project root (LeFusion)
    project_root = neuralsynth_dir.parent
    
    # Add necessary paths to sys.path
    paths_to_add = [
        str(project_root),
        str(project_root / 'evaluation_training'),
        str(project_root / 'utility_training_resources'),
        str(project_root / 'utility_training_resources' / 'DiffTumor' / 'STEP3.SegmentationModel'),
        str(neuralsynth_dir),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return {
        'neuralsynth_dir': neuralsynth_dir,
        'project_root': project_root,
        'data_dir': project_root / 'data',
        'eval_training_dir': project_root / 'evaluation_training',
        'difftumor_dir': project_root / 'utility_training_resources' / 'DiffTumor' / 'STEP3.SegmentationModel',
    }

def get_data_path(dataset: str = 'lidc') -> Path:
    """Get the data path for a specific dataset."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / 'data' / dataset.upper()

def get_synthetic_data_path(dataset: str = 'lidc', method: str = 'neuralsynth') -> Path:
    """Get the synthetic data path."""
    neuralsynth_dir = Path(__file__).parent.parent
    return neuralsynth_dir / 'synthetic_data' / dataset / method

def get_model_path(dataset: str, method: str, combination: str, seg_model: str) -> Path:
    """Get the model save path."""
    neuralsynth_dir = Path(__file__).parent.parent
    model_name = f"{dataset}_{method}_{combination}_{seg_model}"
    return neuralsynth_dir / 'trained_models' / dataset / model_name

def get_results_path(dataset: str) -> Path:
    """Get the results path."""
    neuralsynth_dir = Path(__file__).parent.parent
    return neuralsynth_dir / 'evaluation_results' / f"{dataset}_results.json"

def resolve_config_path(path_str: str, base_dir: Path = None) -> Path:
    """
    Resolve a path string from config file.
    Handles both relative and absolute paths.
    """
    path = Path(path_str)
    
    # If path is already absolute, return it
    if path.is_absolute():
        return path
    
    # If base_dir not provided, use NeuralSynth directory
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    # Handle special cases
    if path_str.startswith('..'):
        # Path relative to parent
        return (base_dir / path).resolve()
    elif path_str.startswith('.'):
        # Path relative to current
        return (base_dir / path).resolve()
    else:
        # Assume relative to base_dir
        return (base_dir / path).resolve()