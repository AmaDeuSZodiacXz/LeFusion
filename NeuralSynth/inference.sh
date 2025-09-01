#!/bin/bash

echo "Running NeuralSynth Inference"
echo "============================="

MODEL_TYPE=${1:-"lidc"}
CHECKPOINT_PATH=${2:-"./checkpoints/neuralsynth_${MODEL_TYPE}/best_model.pt"}
OUTPUT_DIR=${3:-"./outputs/neuralsynth_${MODEL_TYPE}"}

mkdir -p $OUTPUT_DIR

python -u inference.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --num_samples 50 \
    --batch_size 4 \
    --use_ddim \
    --ddim_steps 50 \
    --use_cache \
    --device cuda \
    2>&1 | tee "${OUTPUT_DIR}/inference_$(date +%Y%m%d_%H%M%S).log"

echo "Inference completed! Results saved to $OUTPUT_DIR"