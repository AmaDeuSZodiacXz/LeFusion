# SALAD Organized Folder Structure

## 📂 Complete Directory Layout

```
SALAD/
│
├── 🚀 inference/                    # Inference Pipeline
│   ├── inference_pipeline.py        # Main inference script
│   ├── quick_inference.py          # Quick test script
│   ├── configs/
│   │   └── inference_config.yaml   # Inference configuration
│   ├── scripts/
│   │   └── run_inference.sh        # Bash runner
│   └── examples/                   # Usage examples
│
├── 🎯 training/                     # Training Pipeline
│   ├── train_emidec.py             # EMIDEC training
│   ├── train_lidc.py               # LIDC training
│   ├── train_lidc_stable.py        # Stable training
│   ├── train_lidc_steps.py         # Step-wise training
│   ├── train_emidec_steps.py       # EMIDEC steps
│   ├── README.md                   # Training guide
│   ├── configs/                    # Training configs
│   ├── scripts/
│   │   ├── train_lidc.sh
│   │   ├── train_emidec.sh
│   │   └── train_lidc_stable.sh
│   └── datasets/                   # Dataset handlers
│
├── 🧠 models/                       # Core Models
│   ├── salad_core.py              # Main SALAD model
│   ├── advanced_losses.py         # Loss functions
│   ├── optimized_inference.py     # Fast inference
│   └── tiny_lesion_adapter.py     # Lesion adapter
│
├── 📊 evaluation/                   # Evaluation Metrics
│   ├── advanced_metrics.py        # Advanced metrics
│   ├── comparative_framework.py   # Comparison tools
│   └── paper_metrics.py          # Paper metrics
│
├── 📚 documentation/                # Documentation
│   ├── architecture/
│   │   ├── SALAD_ARCHITECTURE.md
│   │   ├── SALAD_WORKING_PRINCIPLES.md
│   │   └── DEEP_ARCHITECTURE_EXPLANATION.md
│   ├── methods/
│   │   ├── METHODS_DETAILED.md
│   │   ├── TECHNIQUE_DETAILS.md
│   │   └── TECHNIQUE_NAMES.md
│   └── branding/
│       ├── SALAD_BRANDING.md
│       ├── VISUAL_IDENTITY.md
│       └── NAME_SCORING_MATRIX.md
│
├── 🧪 experiments/                  # Experimental Pipelines
│   ├── evaluation_pipeline/       # Evaluation experiments
│   ├── segmentation_training/     # Segmentation experiments
│   └── synthetic_generation/      # Generation experiments
│
├── ⚙️ configs/                      # Global Configurations
│   ├── main_config.yaml          # Main configuration
│   └── salad_config.yaml         # SALAD specific config
│
├── 💾 data/                         # Datasets
│   ├── LIDC/                     # LIDC dataset
│   └── EMIDEC/                   # EMIDEC dataset
│
├── 🎨 assets/                       # Visual Assets
│   ├── salad_architecture.png    # Architecture diagram
│   └── salad_diffusion.png       # Diffusion diagram
│
├── 📝 paper/                        # Research Paper
│   ├── neuralsynth_paper.tex     # Main paper
│   ├── main.tex                  # LaTeX main
│   └── references.bib            # Bibliography
│
├── ✅ tests/                        # Unit Tests
│   └── test_pipeline_integration.py
│
├── 🛠️ utils/                        # Utilities
│   └── path_utils.py
│
├── 💪 checkpoints/                  # Model Checkpoints
├── 📈 results/                      # Output Results
│   └── synthesis/                # Generated samples
│
├── 🏃 run_salad.sh                 # Main launcher script
├── README.md                      # Project README
├── README_STRUCTURE.md            # Structure documentation
├── requirements.txt               # Python dependencies
└── environment.yml                # Conda environment
```

## 🎯 Quick Usage

### 1️⃣ **Run Inference**
```bash
# Method 1: Using launcher
./run_salad.sh inference --checkpoint checkpoints/salad_best.pt

# Method 2: Direct inference
cd inference
python inference_pipeline.py --checkpoint ../checkpoints/salad_best.pt

# Method 3: Quick test
./run_salad.sh quick checkpoints/salad_best.pt
```

### 2️⃣ **Train Model**
```bash
# Train on LIDC
./run_salad.sh train lidc

# Train on EMIDEC
./run_salad.sh train emidec

# Or directly
cd training
./scripts/train_lidc.sh
```

### 3️⃣ **Evaluate Model**
```bash
cd evaluation
python advanced_metrics.py --checkpoint ../checkpoints/salad_best.pt
```

## 📋 Directory Purposes

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| **inference/** | Synthetic data generation | `inference_pipeline.py` |
| **training/** | Model training scripts | `train_*.py`, `train_*.sh` |
| **models/** | Core SALAD implementation | `salad_core.py` |
| **evaluation/** | Performance metrics | `advanced_metrics.py` |
| **documentation/** | All docs organized by topic | Architecture, methods, branding |
| **experiments/** | Experimental pipelines | Various test pipelines |
| **configs/** | Configuration files | `main_config.yaml` |
| **data/** | Dataset storage | LIDC, EMIDEC |
| **checkpoints/** | Saved model weights | `.pt` files |
| **results/** | Generated outputs | Synthetic images |

## 🔧 Configuration Files

- `configs/main_config.yaml` - Global project configuration
- `inference/configs/inference_config.yaml` - Inference settings
- `training/configs/` - Training configurations

## 🚦 Entry Points

1. **Main Launcher**: `./run_salad.sh`
2. **Inference**: `inference/inference_pipeline.py`
3. **Training**: `training/scripts/train_*.sh`
4. **Quick Test**: `inference/quick_inference.py`

## 📦 Dependencies

- See `requirements.txt` for Python packages
- See `environment.yml` for Conda environment

---

**Note**: The structure is designed for easy navigation and clear separation of concerns. Each major component has its own directory with relevant subdirectories for organization.