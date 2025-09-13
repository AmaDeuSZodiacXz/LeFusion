# STEP 3: Train Segmentation Models with DiffTumor

## Overview

Train segmentation models using combinations of real and synthetic data with the DiffTumor framework from `utility_training_resources`.

## Integration with DiffTumor

This step integrates with the DiffTumor framework located at:
```
../../utility_training_resources/DiffTumor/STEP3.SegmentationModel/
```

## Data Preparation

### 1. Prepare Data Combinations

```bash
python prepare_data_combinations.py \
    --real_dir ../../data/LIDC/Pathological \
    --synthetic_dir ../synthetic_data/lidc \
    --output_dir ./data_combinations
```

### Data Combinations Explained

| Combination | Description | Expected DICE |
|-------------|-------------|---------------|
| **P** | Real pathological only (baseline) | ~78% |
| **P_P_prime** | Real + synthetic from pathological | ~81% |
| **P_N_prime** | Real + synthetic from normal (main) | ~85% |
| **P_P_prime_N_double_prime** | All combined | **~89%** |

## Training with DiffTumor

### 1. Train nnU-Net

```bash
python train_with_difftumor.py \
    --difftumor_path ../../utility_training_resources/DiffTumor/STEP3.SegmentationModel \
    --data_combination P_N_prime \
    --model_type nnunet \
    --dataset lidc \
    --epochs 200 \
    --batch_size 2 \
    --output_dir ../segmentation_models/lidc/neuralsynth_P_N_prime_nnunet
```

### 2. Train SwinUNETR

```bash
python train_with_difftumor.py \
    --difftumor_path ../../utility_training_resources/DiffTumor/STEP3.SegmentationModel \
    --data_combination P_N_prime \
    --model_type swinunetr \
    --dataset lidc \
    --epochs 200 \
    --batch_size 4 \
    --output_dir ../segmentation_models/lidc/neuralsynth_P_N_prime_swinunetr
```

## Configuration Files

### nnU-Net Configuration (`configs/nnunet_config.yaml`)
```yaml
model:
  architecture: "3d_fullres"
  in_channels: 1
  num_classes: 2
  
training:
  epochs: 200
  batch_size: 2
  learning_rate: 0.01
  optimizer: "SGD"
  momentum: 0.99
  weight_decay: 3e-5
  
  # nnU-Net specific
  deep_supervision: true
  dice_loss: true
  ce_loss: true
  
augmentation:
  do_elastic_deform: true
  do_rotation: true
  do_scaling: true
  do_mirror: true
```

### SwinUNETR Configuration (`configs/swinunetr_config.yaml`)
```yaml
model:
  img_size: [96, 96, 96]
  in_channels: 1
  out_channels: 2
  feature_size: 48
  depths: [2, 2, 2, 2]
  num_heads: [3, 6, 12, 24]
  
training:
  epochs: 200
  batch_size: 4
  learning_rate: 1e-4
  optimizer: "AdamW"
  weight_decay: 1e-5
  warmup_epochs: 50
  
loss:
  dice_ce_loss: true
  dice_weight: 0.5
  ce_weight: 0.5
```

## DiffTumor Integration Script

```python
# train_with_difftumor.py key components

def integrate_with_difftumor(difftumor_path, config):
    """
    Integrates NeuralSynth data with DiffTumor training
    """
    # Add DiffTumor to path
    sys.path.append(difftumor_path)
    
    # Import DiffTumor modules
    from DiffTumor import SegmentationTrainer
    
    # Initialize trainer with our data
    trainer = SegmentationTrainer(
        data_dir=config['data_dir'],
        model_type=config['model_type'],
        config=config
    )
    
    # Train model
    trainer.train()
    
    return trainer.get_best_model()
```

## Training Process

### 1. Data Loading
- Combines real pathological with synthetic data
- Creates train/validation splits (80/20)
- Applies data augmentation

### 2. Model Training
- Uses DiffTumor's optimized training loop
- Implements early stopping
- Saves best model based on validation DICE

### 3. Checkpointing
```
segmentation_models/lidc/neuralsynth_P_N_prime_nnunet/
├── model_best.pth          # Best validation DICE
├── model_latest.pth        # Latest epoch
├── training_log.txt        # Training history
└── validation_scores.json  # Validation metrics
```

## Expected Training Time

| Model | Dataset | GPU | Training Time |
|-------|---------|-----|---------------|
| nnU-Net | LIDC | A100 | ~12 hours |
| nnU-Net | EMIDEC | A100 | ~8 hours |
| SwinUNETR | LIDC | A100 | ~8 hours |
| SwinUNETR | EMIDEC | A100 | ~6 hours |

## Monitoring Progress

### TensorBoard
```bash
tensorboard --logdir=../segmentation_models/lidc/logs
```

### Expected Metrics
```
Epoch [50/200]
Train Loss: 0.234
Train DICE: 0.823
Val Loss: 0.198
Val DICE: 0.856 ↑ (Best)
```

## Training All Combinations

To train all data combinations automatically:

```bash
bash train_all_combinations.sh
```

This will train models for:
1. P (baseline)
2. P_P_prime
3. P_N_prime (main)
4. P_P_prime_N_double_prime

## Troubleshooting

### Common Issues

1. **DiffTumor not found**
   ```bash
   # Verify path exists
   ls ../../utility_training_resources/DiffTumor/STEP3.SegmentationModel
   ```

2. **CUDA Out of Memory**
   - Reduce batch_size
   - Enable gradient checkpointing
   - Use mixed precision training

3. **Slow Convergence**
   - Check data quality from STEP 2
   - Adjust learning rate
   - Increase augmentation

## Output Structure

```
segmentation_models/
├── lidc/
│   ├── baseline_P_only/
│   │   ├── nnunet/
│   │   └── swinunetr/
│   ├── neuralsynth_P_N_prime/
│   │   ├── nnunet/
│   │   └── swinunetr/
│   └── neuralsynth_all_combined/
│       ├── nnunet/
│       └── swinunetr/
└── emidec/
    └── [same structure]
```

## Next Step

After training completes, proceed to [STEP 4: Evaluation](../STEP4_evaluation/README.md)