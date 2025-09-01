#!/bin/bash

echo "Starting NeuralSynth training on EMIDEC dataset"
echo "==============================================="

export CUDA_VISIBLE_DEVICES=0

DATA_DIR="/Users/skb/Documents/LeFusion/data/EMIDEC"
CHECKPOINT_DIR="./checkpoints/neuralsynth_emidec"
LOG_DIR="./logs/neuralsynth_emidec"

mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

python -u training/train_emidec.py \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_dir $LOG_DIR \
    --batch_size 6 \
    --num_epochs 150 \
    --learning_rate 2e-4 \
    --image_size 256 \
    --num_workers 4 \
    --use_fp16 \
    --use_adaptive_noise \
    --use_multi_scale \
    --use_lesion_attention \
    --num_timesteps 1000 \
    --augment \
    --save_every 10 \
    --validate_every 5 \
    2>&1 | tee "${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

echo "Training completed!"