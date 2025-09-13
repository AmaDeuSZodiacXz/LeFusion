# SALAD Clean Directory Structure

## Core Files
- `inference_pipeline.py` - Main inference pipeline for synthetic data generation
- `run_inference.sh` - Bash script for easy inference execution
- `quick_inference.py` - Minimal script for quick testing

## Training Scripts
- `train_lidc.sh` - Train on LIDC dataset
- `train_emidec.sh` - Train on EMIDEC dataset
- `train_lidc_stable.sh` - Stable training configuration

## Directory Structure
```
SALAD/
├── configs/              # Configuration files
│   └── salad_config.yaml
├── checkpoints/          # Model checkpoints
├── data/                 # Dataset directories
│   ├── LIDC/
│   └── EMIDEC/
├── models/               # Core model implementations
│   ├── salad_core.py     # Main SALAD model
│   ├── advanced_losses.py # Loss functions
│   └── optimized_inference.py # Fast inference
├── evaluation/           # Evaluation metrics
├── results/              # Output directory
│   └── synthesis/        # Generated samples
└── logs/                 # Training logs
```

## Usage
```bash
# Run inference
./run_inference.sh checkpoints/salad_best.pt

# Or with Python
python inference_pipeline.py --checkpoint checkpoints/salad_best.pt
```
