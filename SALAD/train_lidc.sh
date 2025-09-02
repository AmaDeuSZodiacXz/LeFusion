#!/bin/bash

# Train NeuralSynth on LIDC dataset (step-based like LeFusion)
echo "================================================"
echo "Training NeuralSynth on LIDC Dataset"
echo "Using step-based training (like LeFusion)"
echo "================================================"

cd synthetic_training

python train_lidc_steps.py \
    --data_dir ../../data/LIDC/Pathological \
    --output_dir ../checkpoints/lidc_steps \
    --train_num_steps 50001 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --save_every 5000 \
    --use_adaptive_noise \
    --use_lesion_attention \
    --use_multi_scale \
    --num_workers 4 \
    --device cuda:0

echo "Training complete!"