#!/bin/bash

# Train NeuralSynth on EMIDEC dataset (step-based like LeFusion)
echo "================================================"
echo "Training NeuralSynth on EMIDEC Dataset"
echo "Using step-based training (like LeFusion)"
echo "================================================"

cd synthetic_training

python train_emidec_steps.py \
    --data_dir ../../data/EMIDEC \
    --output_dir ../checkpoints/emidec_steps \
    --train_num_steps 50001 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --save_every 5000 \
    --image_size 256 \
    --model_channels 128 \
    --use_adaptive_noise \
    --use_lesion_attention \
    --use_multi_scale \
    --num_workers 4 \
    --device cuda:0

echo "Training complete!"