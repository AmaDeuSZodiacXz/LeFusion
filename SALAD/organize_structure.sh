#!/bin/bash

# SALAD Folder Structure Organization Script
# Creates a clean, organized directory structure

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Organizing SALAD Folder Structure    ${NC}"
echo -e "${BLUE}========================================${NC}"

# Create organized directory structure
echo -e "${YELLOW}Creating organized directory structure...${NC}"

# Main directories
mkdir -p inference
mkdir -p training
mkdir -p documentation
mkdir -p assets
mkdir -p experiments

# Inference subdirectories
mkdir -p inference/scripts
mkdir -p inference/configs
mkdir -p inference/examples

# Training subdirectories  
mkdir -p training/scripts
mkdir -p training/configs
mkdir -p training/datasets

# Documentation subdirectories
mkdir -p documentation/architecture
mkdir -p documentation/methods
mkdir -p documentation/branding

echo -e "${GREEN}✓ Created directory structure${NC}"

# Move inference-related files
echo -e "${YELLOW}Organizing inference files...${NC}"
mv inference_pipeline.py inference/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved inference_pipeline.py"
mv quick_inference.py inference/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved quick_inference.py"
mv run_inference.sh inference/scripts/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved run_inference.sh"

# Move training scripts
echo -e "${YELLOW}Organizing training files...${NC}"
mv train_*.sh training/scripts/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved training scripts"
mv synthetic_training/* training/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved synthetic training files"
rmdir synthetic_training 2>/dev/null

# Move documentation files
echo -e "${YELLOW}Organizing documentation...${NC}"
mv SALAD_ARCHITECTURE.md documentation/architecture/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved SALAD_ARCHITECTURE.md"
mv SALAD_WORKING_PRINCIPLES.md documentation/architecture/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved SALAD_WORKING_PRINCIPLES.md"
mv DEEP_ARCHITECTURE_EXPLANATION.md documentation/architecture/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved architecture docs"

mv METHODS_DETAILED.md documentation/methods/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved METHODS_DETAILED.md"
mv TECHNIQUE_*.md documentation/methods/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved technique docs"

mv SALAD_BRANDING.md documentation/branding/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved SALAD_BRANDING.md"
mv VISUAL_IDENTITY.md documentation/branding/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved VISUAL_IDENTITY.md"
mv NAME_SCORING_MATRIX.md documentation/branding/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved naming docs"

# Move assets
echo -e "${YELLOW}Organizing assets...${NC}"
mv *.png assets/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved image files"

# Move experiment-related directories
echo -e "${YELLOW}Organizing experiments...${NC}"
mv evaluation_pipeline experiments/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved evaluation_pipeline"
mv segmentation_training experiments/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved segmentation_training"
mv synthetic_generation experiments/ 2>/dev/null && echo -e "  ${GREEN}✓${NC} Moved synthetic_generation"

# Clean up empty directories
echo -e "${YELLOW}Cleaning up empty directories...${NC}"
rmdir synthetic_data 2>/dev/null
rmdir logs 2>/dev/null

# Create main configuration file
echo -e "${YELLOW}Creating main configuration...${NC}"
cat > configs/main_config.yaml << 'EOF'
# SALAD Main Configuration
project:
  name: SALAD
  version: 1.0.0
  description: Stochastic Augmentation with Lesion-Aware Diffusion

paths:
  inference: ./inference
  training: ./training
  models: ./models
  data: ./data
  results: ./results
  checkpoints: ./checkpoints

inference:
  default_config: ./inference/configs/inference_config.yaml
  scripts: ./inference/scripts
  
training:
  default_config: ./training/configs/training_config.yaml
  scripts: ./training/scripts
EOF
echo -e "  ${GREEN}✓${NC} Created main_config.yaml"

# Create inference configuration
cat > inference/configs/inference_config.yaml << 'EOF'
# SALAD Inference Configuration
model:
  checkpoint: ../checkpoints/salad_best.pt
  device: cuda
  
sampling:
  method: DDIM
  steps: 50
  batch_size: 8
  guidance_scale: 1.0
  
output:
  format: nifti
  directory: ../results/synthesis
EOF
echo -e "  ${GREEN}✓${NC} Created inference_config.yaml"

# Create organized README
cat > README_STRUCTURE.md << 'EOF'
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
EOF
echo -e "  ${GREEN}✓${NC} Created README_STRUCTURE.md"

# Create simple launcher script
cat > run_salad.sh << 'EOF'
#!/bin/bash

# SALAD Main Launcher Script

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}           SALAD Launcher               ${NC}"
echo -e "${BLUE}========================================${NC}"

if [ "$1" == "inference" ]; then
    echo -e "${YELLOW}Running inference pipeline...${NC}"
    cd inference
    python inference_pipeline.py "${@:2}"
elif [ "$1" == "train" ]; then
    echo -e "${YELLOW}Starting training...${NC}"
    cd training
    if [ "$2" == "lidc" ]; then
        ./scripts/train_lidc.sh
    elif [ "$2" == "emidec" ]; then
        ./scripts/train_emidec.sh
    else
        echo -e "${RED}Please specify dataset: lidc or emidec${NC}"
    fi
elif [ "$1" == "quick" ]; then
    echo -e "${YELLOW}Running quick inference...${NC}"
    cd inference
    python quick_inference.py "${@:2}"
else
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./run_salad.sh inference [options]  - Run inference"
    echo "  ./run_salad.sh train [lidc|emidec]  - Train model"
    echo "  ./run_salad.sh quick [checkpoint]   - Quick test"
fi
EOF
chmod +x run_salad.sh
echo -e "  ${GREEN}✓${NC} Created run_salad.sh launcher"

# Final summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    Organization Complete!              ${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "${YELLOW}New Structure:${NC}"
echo "  📁 inference/     - All inference-related files"
echo "  📁 training/      - Training scripts and configs"
echo "  📁 documentation/ - Organized documentation"
echo "  📁 experiments/   - Experimental pipelines"
echo "  📁 assets/        - Images and visual files"

echo -e "${YELLOW}\nMain Entry Points:${NC}"
echo "  • ./run_salad.sh inference  - Run inference"
echo "  • ./run_salad.sh train      - Train model"
echo "  • ./run_salad.sh quick      - Quick test"

echo -e "${YELLOW}\nSee README_STRUCTURE.md for detailed information${NC}"