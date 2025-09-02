# STEP 1: Train Synthetic Model (NeuralSynth Technique)

## Overview

This is the core innovation of NeuralSynth - training a diffusion model with:
- **Adaptive Noise Scheduling**: Learnable beta parameters
- **Lesion-Aware Attention**: Focus on lesion boundaries
- **Multi-Scale Features**: Handle all lesion sizes
- **7-Component Loss System**: Comprehensive quality control

## Training Scripts

### 1. Train on LIDC Dataset

```bash
python train_lidc.py \
    --data_dir ../../data/LIDC/Pathological \
    --output_dir ../checkpoints/lidc \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --num_timesteps 1000 \
    --use_adaptive_noise \
    --use_lesion_attention \
    --use_multi_scale \
    --ddim_steps 50
```

### 2. Train on EMIDEC Dataset

```bash
python train_emidec.py \
    --data_dir ../../data/EMIDEC/Pathological \
    --output_dir ../checkpoints/emidec \
    --epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --num_classes 2 \  # MI and PMO
    --use_adaptive_noise \
    --use_lesion_attention \
    --use_multi_scale
```

## Model Architecture

```python
NeuralSynthDiffusion(
    in_channels=1,
    out_channels=1,
    base_channels=128,  # 2x LeFusion
    attention_resolutions=[16, 8],
    num_res_blocks=3,
    use_adaptive_noise=True,
    use_lesion_attention=True,
    use_multi_scale=True
)
```

## Key Components

### Adaptive Noise Scheduler
- Learns optimal beta schedule during training
- Adapts to dataset characteristics
- Faster convergence than fixed schedule

### Lesion-Aware Attention
- Spatial attention mechanism
- Bias towards lesion regions
- Better boundary preservation

### Multi-Scale Feature Extraction
- Parallel processing at scales: [1.0, 0.5, 0.25]
- Captures lesions of all sizes
- Learned fusion of features

### Advanced Loss System
```python
total_loss = (
    1.0 * diffusion_loss +
    0.1 * perceptual_loss +
    0.05 * ssim_loss +
    0.02 * frequency_loss +
    0.02 * edge_loss +
    0.1 * lesion_consistency_loss +
    0.01 * adversarial_loss
)
```

## Training Configuration

### LIDC Configuration (`configs/lidc_config.yaml`)
```yaml
data:
  dataset: "LIDC"
  image_size: [64, 64, 32]
  num_classes: 1  # Binary lesion

model:
  base_channels: 128
  channel_mult: [1, 2, 4, 8]
  attention_resolutions: [16, 8]
  
training:
  epochs: 50
  batch_size: 4
  learning_rate: 1e-4
  gradient_clip: 1.0
  ema_decay: 0.9999
  
diffusion:
  timesteps: 1000
  beta_schedule: "adaptive"
  sampling_method: "DDIM"
  ddim_steps: 50
```

### EMIDEC Configuration (`configs/emidec_config.yaml`)
```yaml
data:
  dataset: "EMIDEC"
  image_size: [72, 72, 10]
  num_classes: 2  # MI and PMO

model:
  base_channels: 128
  channel_mult: [1, 2, 4]
  attention_resolutions: [18, 9]
  
training:
  epochs: 50
  batch_size: 2
  learning_rate: 1e-4
  gradient_clip: 1.0
  
diffusion:
  timesteps: 1000
  beta_schedule: "adaptive"
  multi_class: true
```

## Expected Training Time

| Dataset | GPU | Batch Size | Training Time |
|---------|-----|------------|---------------|
| LIDC | A100 40GB | 4 | ~36 hours |
| LIDC | V100 32GB | 2 | ~48 hours |
| EMIDEC | A100 40GB | 2 | ~24 hours |
| EMIDEC | V100 32GB | 1 | ~36 hours |

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=../checkpoints/lidc/logs
```

### Expected Loss Curves
- Diffusion loss: Should decrease steadily
- Perceptual loss: Stabilizes after 20 epochs
- SSIM loss: Improves gradually

## Checkpoints

Saved every 10 epochs:
```
checkpoints/lidc/
├── neuralsynth_epoch_10.pth
├── neuralsynth_epoch_20.pth
├── neuralsynth_epoch_30.pth
├── neuralsynth_epoch_40.pth
├── neuralsynth_epoch_50.pth
└── neuralsynth_best.pth  # Best validation
```

## Validation Metrics

During training, we monitor:
- **Reconstruction Quality**: MSE, PSNR
- **Perceptual Quality**: LPIPS, SSIM
- **Lesion Accuracy**: Dice coefficient of generated vs target lesions

## Tips for Training

1. **Start with smaller batch size** if OOM
2. **Use gradient checkpointing** for memory efficiency
3. **Enable mixed precision** for faster training
4. **Monitor validation loss** for overfitting
5. **Save checkpoints frequently** for resuming

## Next Step

After training completes, proceed to [STEP 2: Generate Synthetic Data](../STEP2_generate_synthetic_data/README.md)