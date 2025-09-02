#!/bin/bash

# Training script with stable hyperparameters
cd synthetic_training

echo "Starting NeuralSynth training with stable configuration..."
echo "================================================"

python train_lidc.py \
    --data_dir ../../data/LIDC/Pathological \
    --output_dir ../checkpoints/lidc_stable \
    --epochs 100 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --use_adaptive_noise \
    --use_lesion_attention \
    --use_multi_scale \
    --val_interval 5 \
    --save_interval 10 \
    --num_workers 2

echo "Training complete!"