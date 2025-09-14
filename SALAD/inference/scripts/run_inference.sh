#!/bin/bash

# SALAD Inference Script
# Generates synthetic pathological images from normal images

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}        SALAD Inference Pipeline        ${NC}"
echo -e "${BLUE}========================================${NC}"

# Default values
CHECKPOINT="${1:-../checkpoints/lidc_steps/checkpoint_step_50000.pth}"
NORMAL_DIR="${2:-/content/LeFusion/data/LIDC/Normal/Image}"
OUTPUT_DIR="${3:-../results/synthesis}"
DDIM_STEPS="${4:-50}"
DEVICE="${5:-cuda}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${YELLOW}Warning: Checkpoint not found at $CHECKPOINT${NC}"
fi

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Checkpoint: $CHECKPOINT"
echo "  Normal images: $NORMAL_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  DDIM steps: $DDIM_STEPS"
echo "  Device: $DEVICE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
echo -e "${GREEN}Starting synthesis...${NC}"

cd ..  # Go to SALAD directory
python inference/inference.py \
    --checkpoint "$CHECKPOINT" \
    --normal_dir "$NORMAL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ddim_steps "$DDIM_STEPS" \
    --device "$DEVICE"

# Check results
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Synthesis completed successfully!${NC}"
    
    # Count generated files
    if [ -d "$OUTPUT_DIR" ]; then
        NUM_FILES=$(ls -1 "$OUTPUT_DIR"/*.nii.gz 2>/dev/null | wc -l)
        echo -e "${GREEN}Generated $NUM_FILES synthetic images${NC}"
    fi
else
    echo -e "${RED}✗ Synthesis failed!${NC}"
    exit 1
fi