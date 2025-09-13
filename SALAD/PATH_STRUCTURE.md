# NeuralSynth Path Structure Documentation

## 📁 Relative Path System

All paths in NeuralSynth now use **relative paths** instead of absolute paths. This makes the project portable and easier to share across different systems.

## 🗂️ Directory Structure

```
LeFusion/                        # Project root (../)
├── NeuralSynth/                # Current directory (./)
│   ├── configs/                # ./configs/
│   ├── models/                 # ./models/
│   ├── training/               # ./training/
│   ├── evaluation/             # ./evaluation/
│   ├── synthetic_data/         # ./synthetic_data/
│   ├── trained_models/         # ./trained_models/
│   └── evaluation_results/     # ./evaluation_results/
├── data/                       # ../data/
│   ├── LIDC/                   # ../data/LIDC/
│   └── EMIDEC/                 # ../data/EMIDEC/
├── evaluation_training/        # ../evaluation_training/
└── utility_training_resources/ # ../utility_training_resources/
    └── DiffTumor/              # ../utility_training_resources/DiffTumor/
```

## 🔧 Path Configuration

### In Python Scripts

```python
from pathlib import Path

# Get NeuralSynth directory
neuralsynth_dir = Path(__file__).parent.parent  # If in subdirectory
# or
neuralsynth_dir = Path(__file__).parent  # If in root of NeuralSynth

# Get project root (LeFusion)
project_root = neuralsynth_dir.parent

# Access data
data_dir = project_root / 'data'
lidc_data = data_dir / 'LIDC'
emidec_data = data_dir / 'EMIDEC'

# Access other modules
eval_training = project_root / 'evaluation_training'
difftumor = project_root / 'utility_training_resources' / 'DiffTumor'
```

### In YAML Configuration

```yaml
# configs/training_config.yaml
datasets:
  lidc:
    data_path: "../data/LIDC"  # Relative to NeuralSynth
  emidec:
    data_path: "../data/EMIDEC"

paths:
  base_dir: ".."               # Parent directory (LeFusion)
  neuralsynth_dir: "."         # Current directory
  data_dir: "../data"
  synthetic_data_dir: "./synthetic_data"
  trained_models_dir: "./trained_models"
  difftumor_path: "../utility_training_resources/DiffTumor/STEP3.SegmentationModel"
```

### In Shell Scripts

```bash
# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to NeuralSynth root
NEURALSYNTH_DIR="$(dirname "$SCRIPT_DIR")"

# Get project root
PROJECT_ROOT="$(dirname "$NEURALSYNTH_DIR")"

# Access data
DATA_DIR="$PROJECT_ROOT/data"
```

## 📦 Path Utilities

Use the provided utilities for consistent path handling:

```python
from utils.path_utils import (
    setup_paths,
    get_data_path,
    get_synthetic_data_path,
    get_model_path,
    get_results_path
)

# Setup all paths
paths = setup_paths()

# Get specific paths
lidc_data = get_data_path('lidc')
synthetic_path = get_synthetic_data_path('lidc', 'neuralsynth')
model_path = get_model_path('lidc', 'neuralsynth', 'P_N_prime', 'nnunet')
results = get_results_path('lidc')
```

## 🚀 Running the Pipeline

All commands should be run from the NeuralSynth directory:

```bash
# Navigate to NeuralSynth
cd /path/to/LeFusion/NeuralSynth

# Run complete pipeline
bash scripts/run_complete_pipeline.sh --dataset lidc

# Or individual steps
python training/train_segmentation.py --dataset lidc --combination P_N_prime
python evaluation/evaluate_models.py --dataset lidc
```

## ⚠️ Important Notes

1. **Working Directory**: Always run scripts from the NeuralSynth directory
2. **Import Paths**: Python modules automatically add parent directories to sys.path
3. **Data Access**: All data paths are relative to `../data/`
4. **Model Checkpoints**: Saved in `./trained_models/`
5. **Results**: Stored in `./evaluation_results/`

## 🔍 Troubleshooting

If you encounter path-related errors:

1. **Check working directory**:
   ```bash
   pwd  # Should show /path/to/LeFusion/NeuralSynth
   ```

2. **Test setup**:
   ```bash
   python test_setup.py
   ```

3. **Use path utilities**:
   ```python
   from utils.path_utils import setup_paths
   paths = setup_paths()
   print(paths)  # Shows all resolved paths
   ```

## 📋 Path Reference Table

| Resource | Relative Path from NeuralSynth | Example Usage |
|----------|--------------------------------|---------------|
| LIDC Data | `../data/LIDC/` | `data_dir = Path("../data/LIDC")` |
| EMIDEC Data | `../data/EMIDEC/` | `data_dir = Path("../data/EMIDEC")` |
| Evaluation Training | `../evaluation_training/` | `sys.path.append("../evaluation_training")` |
| DiffTumor | `../utility_training_resources/DiffTumor/` | `difftumor = Path("../utility_training_resources/DiffTumor")` |
| Synthetic Data | `./synthetic_data/` | `synth = Path("./synthetic_data")` |
| Trained Models | `./trained_models/` | `models = Path("./trained_models")` |
| Results | `./evaluation_results/` | `results = Path("./evaluation_results")` |

## ✅ Benefits of Relative Paths

1. **Portability**: Works on any system without modification
2. **Collaboration**: Easy to share and clone
3. **Docker/Container Ready**: No hardcoded paths
4. **Version Control**: No user-specific paths in code
5. **Testing**: Can run tests from any location