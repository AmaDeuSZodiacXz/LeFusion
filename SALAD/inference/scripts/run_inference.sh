#!/bin/bash

# SALAD Inference Script - Generate Synthetic Pathological Images
# Clean and simple pipeline for synthetic data generation

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    SALAD Synthetic Data Generation     ${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
CHECKPOINT_PATH="${1:-checkpoints/salad_best.pt}"
OUTPUT_DIR="${2:-results/synthesis}"
NUM_SAMPLES="${3:-100}"
BATCH_SIZE="${4:-8}"
DDIM_STEPS="${5:-50}"
DEVICE="${6:-cuda}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}Error: Checkpoint not found at $CHECKPOINT_PATH${NC}"
    echo -e "${YELLOW}Please provide a valid checkpoint path as first argument${NC}"
    echo "Usage: ./run_inference.sh <checkpoint_path> [output_dir] [num_samples] [batch_size] [ddim_steps] [device]"
    exit 1
fi

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Number of Samples: $NUM_SAMPLES"
echo "  Batch Size: $BATCH_SIZE"
echo "  DDIM Steps: $DDIM_STEPS (50=fast, 1000=quality)"
echo "  Device: $DEVICE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
echo -e "${GREEN}Starting synthetic data generation...${NC}"

python inference_pipeline.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --ddim_steps "$DDIM_STEPS" \
    --device "$DEVICE" \
    --seed 42 \
    --guidance_scale 1.0 \
    --image_size 256 256 \
    --save_format nifti

# Check if successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Synthesis completed successfully!${NC}"
    echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
    
    # Count generated files
    NUM_FILES=$(ls -1 "$OUTPUT_DIR"/*.nii.gz 2>/dev/null | wc -l)
    echo -e "${GREEN}Generated $NUM_FILES synthetic images${NC}"
else
    echo -e "${RED}✗ Synthesis failed!${NC}"
    exit 1
fi