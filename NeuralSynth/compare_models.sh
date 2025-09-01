#!/bin/bash

echo "Running Model Comparison Framework"
echo "=================================="

NEURALSYNTH_CHECKPOINT="./checkpoints/neuralsynth_lidc/best_model.pt"
LEFUSION_CHECKPOINT="/Users/skb/Documents/LeFusion/LeFusion/checkpoints/best_model.pt"
SCAR_CHECKPOINT="/Users/skb/Documents/LeFusion/CLAIM-Scar-Synthesis/checkpoints/best_model.pt"
OUTPUT_DIR="./comparison_results"

mkdir -p $OUTPUT_DIR

python -u evaluation/run_comparison.py \
    --neuralsynth_checkpoint $NEURALSYNTH_CHECKPOINT \
    --lefusion_checkpoint $LEFUSION_CHECKPOINT \
    --scar_checkpoint $SCAR_CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --dataset "lidc" \
    --num_samples 100 \
    --metrics "all" \
    --generate_plots \
    --generate_report \
    2>&1 | tee "${OUTPUT_DIR}/comparison_$(date +%Y%m%d_%H%M%S).log"

echo "Comparison completed! Results saved to $OUTPUT_DIR"