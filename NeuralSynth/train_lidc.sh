#!/bin/bash

echo "Starting NeuralSynth training on LIDC dataset"
echo "============================================"

export CUDA_VISIBLE_DEVICES=0

DATA_DIR="/Users/skb/Documents/LeFusion/data/LIDC"
CHECKPOINT_DIR="./checkpoints/neuralsynth_lidc"
LOG_DIR="./logs/neuralsynth_lidc"

mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

python -u training/train_lidc.py \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_dir $LOG_DIR \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --image_size 256 \
    --num_workers 4 \
    --use_fp16 \
    --use_adaptive_noise \
    --use_multi_scale \
    --use_lesion_attention \
    --num_timesteps 1000 \
    --save_every 10 \
    --validate_every 5 \
    2>&1 | tee "${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

echo "Training completed!"