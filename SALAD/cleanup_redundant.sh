#!/bin/bash

# SALAD Directory Cleanup Script
# Removes redundant files and organizes the structure

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    SALAD Directory Cleanup Script     ${NC}"
echo -e "${BLUE}========================================${NC}"

# Create backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo -e "${YELLOW}Creating backup directory: $BACKUP_DIR${NC}"
mkdir -p "$BACKUP_DIR"

# Files to remove (redundant/old versions)
REDUNDANT_FILES=(
    "run_neuralsynth.sh"  # Old naming convention
    "fix_imports.py"      # One-time fix script
    "test_training.py"    # Test file
    "train_simple.py"     # Simplified version, use main training scripts
    "setup_conda_env.sh"  # One-time setup
    "install_salad.sh"    # One-time setup
    "ARCHITECTURE_VISUAL.py"  # Documentation file
)

# Move redundant files to backup
echo -e "${YELLOW}Moving redundant files to backup...${NC}"
for file in "${REDUNDANT_FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "$BACKUP_DIR/"
        echo -e "  ${GREEN}✓${NC} Moved $file"
    fi
done

# Consolidate pipeline files
echo -e "${YELLOW}Checking pipeline directory for consolidation...${NC}"

# Remove old/redundant pipeline files
if [ -f "pipeline/difftumor_integration.py" ]; then
    mv "pipeline/difftumor_integration.py" "$BACKUP_DIR/"
    echo -e "  ${GREEN}✓${NC} Moved difftumor_integration.py (redundant with full_pipeline)"
fi

if [ -f "pipeline/segmentation_training.py" ]; then
    mv "pipeline/segmentation_training.py" "$BACKUP_DIR/"
    echo -e "  ${GREEN}✓${NC} Moved segmentation_training.py (integrated in main pipeline)"
fi

if [ -f "pipeline/normal_to_pathological.py" ]; then
    mv "pipeline/normal_to_pathological.py" "$BACKUP_DIR/"
    echo -e "  ${GREEN}✓${NC} Moved normal_to_pathological.py (functionality in inference_pipeline)"
fi

if [ -f "pipeline/full_pipeline.py" ]; then
    mv "pipeline/full_pipeline.py" "$BACKUP_DIR/"
    echo -e "  ${GREEN}✓${NC} Moved full_pipeline.py (replaced by inference_pipeline.py)"
fi

# Clean up old model files if they exist
echo -e "${YELLOW}Checking for old model files...${NC}"
OLD_MODEL_FILES=(
    "models/neuralsynth_core.py"
    "models/neuralsynth_unet.py"
    "models/lesion_adapter.py"
)

for file in "${OLD_MODEL_FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "$BACKUP_DIR/"
        echo -e "  ${GREEN}✓${NC} Moved $file"
    fi
done

# Create clean directory structure
echo -e "${YELLOW}Creating clean directory structure...${NC}"

# Ensure main directories exist
mkdir -p configs
mkdir -p checkpoints
mkdir -p results/synthesis
mkdir -p data/LIDC
mkdir -p data/EMIDEC
mkdir -p logs

# Create main config file if it doesn't exist
if [ ! -f "configs/salad_config.yaml" ]; then
    cat > configs/salad_config.yaml << 'EOF'
# SALAD Configuration File
model:
  name: SALAD
  channels: 128
  attention_resolutions: [16, 8]
  num_res_blocks: 3
  dropout: 0.1
  
inference:
  ddim_steps: 50
  batch_size: 8
  guidance_scale: 1.0
  
training:
  learning_rate: 1e-4
  batch_size: 32
  num_epochs: 100
  
data:
  image_size: [256, 256]
  num_classes: 5
EOF
    echo -e "  ${GREEN}✓${NC} Created configs/salad_config.yaml"
fi

# Create README for clean structure
cat > SALAD_STRUCTURE.md << 'EOF'
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
EOF

echo -e "${GREEN}✓ Created SALAD_STRUCTURE.md${NC}"

# Count cleaned files
CLEANED_COUNT=$(ls -1 "$BACKUP_DIR" 2>/dev/null | wc -l)

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cleanup Complete!${NC}"
echo -e "${GREEN}  - Moved $CLEANED_COUNT redundant files to $BACKUP_DIR${NC}"
echo -e "${GREEN}  - Consolidated pipeline functionality${NC}"
echo -e "${GREEN}  - Created clean directory structure${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "${YELLOW}Main inference files:${NC}"
echo "  • inference_pipeline.py - Main pipeline"
echo "  • run_inference.sh - Easy runner"
echo "  • quick_inference.py - Quick test"

echo -e "${YELLOW}To restore backed up files:${NC}"
echo "  mv $BACKUP_DIR/* ."