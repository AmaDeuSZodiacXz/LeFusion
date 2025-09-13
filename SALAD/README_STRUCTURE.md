# SALAD Project Structure

## 📁 Directory Organization

```
SALAD/
├── inference/              # Inference pipeline and scripts
│   ├── inference_pipeline.py
│   ├── quick_inference.py
│   ├── scripts/           # Shell scripts for inference
│   ├── configs/           # Inference configurations
│   └── examples/          # Example usage scripts
│
├── training/              # Training pipeline and scripts
│   ├── scripts/           # Training shell scripts
│   ├── configs/           # Training configurations
│   └── datasets/          # Dataset-specific code
│
├── models/                # Core model implementations
│   ├── salad_core.py
│   ├── advanced_losses.py
│   └── optimized_inference.py
│
├── evaluation/            # Evaluation metrics and tools
│   ├── advanced_metrics.py
│   └── comparative_framework.py
│
├── documentation/         # All documentation
│   ├── architecture/      # Architecture descriptions
│   ├── methods/           # Method explanations
│   └── branding/          # Branding and naming
│
├── experiments/           # Experimental pipelines
│   ├── evaluation_pipeline/
│   ├── segmentation_training/
│   └── synthetic_generation/
│
├── configs/               # Global configurations
│   └── main_config.yaml
│
├── data/                  # Dataset storage
│   ├── LIDC/
│   └── EMIDEC/
│
├── checkpoints/           # Model checkpoints
├── results/               # Output results
├── assets/                # Images and visual assets
├── paper/                 # Research paper
├── tests/                 # Unit tests
└── utils/                 # Utility functions
```

## 🚀 Quick Start

### Inference
```bash
cd inference
python inference_pipeline.py --checkpoint ../checkpoints/salad_best.pt
# or
./scripts/run_inference.sh
```

### Training
```bash
cd training
./scripts/train_lidc.sh
# or
./scripts/train_emidec.sh
```

## 📝 Main Components

- **Inference Pipeline**: Streamlined synthetic data generation
- **Training Scripts**: Dataset-specific training configurations
- **Models**: Core SALAD implementation with adaptive noise scheduling
- **Documentation**: Comprehensive architecture and method descriptions
