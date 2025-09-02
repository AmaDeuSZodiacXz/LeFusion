# STEP 2: Generate Synthetic Data from Normal Cases

## Overview

Use the trained NeuralSynth diffusion model to generate synthetic pathological images from abundant normal medical scans. This leverages the key insight that >90% of medical imaging data consists of normal cases.

## Core Principle

```
Synthetic = Generated_Lesion * Mask + Normal_Background * (1 - Mask)
```

**100% Background Preservation**: We NEVER modify the anatomical background, only insert realistic lesions.

## Generation Scripts

### 1. Generate from Normal Cases (Main Approach)

```bash
python generate_from_normal.py \
    --model_path ../checkpoints/lidc/neuralsynth_best.pth \
    --normal_dir ../../data/LIDC/Normal \
    --output_dir ../synthetic_data/lidc/P_N_prime \
    --num_samples 1000 \
    --ddim_steps 50 \
    --batch_size 8 \
    --device cuda
```

### 2. Generate from Pathological Cases

```bash
python generate_from_pathological.py \
    --model_path ../checkpoints/lidc/neuralsynth_best.pth \
    --pathological_dir ../../data/LIDC/Pathological \
    --output_dir ../synthetic_data/lidc/P_P_prime \
    --num_samples 500 \
    --ddim_steps 50 \
    --batch_size 8
```

### 3. Generate Double Amount (2x Synthetic)

```bash
python batch_generation.py \
    --model_path ../checkpoints/lidc/neuralsynth_best.pth \
    --normal_dir ../../data/LIDC/Normal \
    --output_dir ../synthetic_data/lidc/P_N_double_prime \
    --num_samples 2000 \
    --ddim_steps 50 \
    --batch_size 8
```

## Generation Pipeline

### Step 1: Load Trained Model
```python
model = NeuralSynthDiffusion.load_from_checkpoint(
    checkpoint_path="../checkpoints/lidc/neuralsynth_best.pth"
)
model.eval()
```

### Step 2: Generate Lesion Masks
```python
# Using trained mask generator
mask_generator = MaskGenerator(
    min_size=5,
    max_size=30,
    num_lesions_range=(1, 3),
    shape_types=['nodular', 'irregular', 'spiculated']
)

mask = mask_generator.generate(image_shape)
```

### Step 3: Generate Lesion Texture
```python
# DDIM sampling (50 steps instead of 1000)
lesion_texture = model.sample(
    shape=image_shape,
    mask=mask,
    ddim_steps=50,
    eta=0.0  # Deterministic
)
```

### Step 4: Composite with Normal Background
```python
# Preserve 100% of background
synthetic = lesion_texture * mask + normal_image * (1 - mask)
```

## Lesion Control Parameters

### Histogram Control
```python
histogram_control = {
    'num_peaks': 2,  # Multi-peak for heterogeneous lesions
    'peak_positions': [0.3, 0.7],
    'peak_widths': [0.1, 0.15],
    'intensity_range': [0.2, 0.9]
}
```

### Spatial Control
```python
spatial_control = {
    'location_bias': 'peripheral',  # or 'central', 'random'
    'size_distribution': 'log-normal',
    'boundary_smoothness': 0.7
}
```

## Generation Speed Comparison

| Method | Steps | Time per Image | Quality |
|--------|-------|----------------|---------|
| LeFusion (DDPM) | 1000 | ~40 seconds | Baseline |
| NeuralSynth (DDIM) | 50 | **~2 seconds** | Better |
| NeuralSynth (DDIM) | 100 | ~4 seconds | Best |

## Data Organization

### Output Structure
```
synthetic_data/
├── lidc/
│   ├── P_N_prime/              # Synthetic from normal (main)
│   │   ├── images/
│   │   │   ├── synthetic_001.nii.gz
│   │   │   └── ...
│   │   ├── masks/
│   │   │   ├── mask_001.nii.gz
│   │   │   └── ...
│   │   └── metadata.json
│   ├── P_P_prime/              # Synthetic from pathological
│   │   └── [same structure]
│   └── P_N_double_prime/       # 2x synthetic from normal
│       └── [same structure]
└── emidec/
    └── [same structure]
```

### Metadata Format
```json
{
  "generation_params": {
    "model": "neuralsynth_best.pth",
    "ddim_steps": 50,
    "timestamp": "2024-01-15T10:30:00"
  },
  "samples": [
    {
      "id": "synthetic_001",
      "source": "normal_case_045.nii.gz",
      "lesion_count": 2,
      "lesion_sizes": [12, 8],
      "generation_time": 2.1
    }
  ]
}
```

## Quality Control

### Visual Inspection
```bash
python visualize_synthetic.py \
    --synthetic_dir ../synthetic_data/lidc/P_N_prime \
    --num_samples 20 \
    --output_dir ../results/visualizations
```

### Quantitative Metrics
```bash
python evaluate_generation_quality.py \
    --synthetic_dir ../synthetic_data/lidc/P_N_prime \
    --real_dir ../../data/LIDC/Pathological \
    --metrics fid,lpips,ssim
```

### Expected Quality Metrics
- **FID Score**: < 15 (lower is better)
- **LPIPS**: < 0.2 (lower is better)
- **SSIM**: > 0.85 (higher is better)
- **Lesion Dice (vs real)**: > 0.7

## Batch Processing

### For Large Datasets
```bash
python batch_generation.py \
    --model_path ../checkpoints/lidc/neuralsynth_best.pth \
    --normal_dir ../../data/LIDC/Normal \
    --output_dir ../synthetic_data/lidc/large_batch \
    --num_samples 10000 \
    --batch_size 32 \
    --num_workers 4 \
    --save_interval 100  # Save every 100 samples
```

### Multi-GPU Generation
```bash
python parallel_generation.py \
    --model_path ../checkpoints/lidc/neuralsynth_best.pth \
    --normal_dir ../../data/LIDC/Normal \
    --output_dir ../synthetic_data/lidc/multi_gpu \
    --num_samples 10000 \
    --gpus 0,1,2,3
```

## EMIDEC-Specific Generation

### Multi-Class Lesions (MI and PMO)
```bash
python generate_emidec.py \
    --model_path ../checkpoints/emidec/neuralsynth_best.pth \
    --normal_dir ../../data/EMIDEC/Normal \
    --output_dir ../synthetic_data/emidec/P_N_prime \
    --num_samples 500 \
    --lesion_type MI  # or PMO, or both
    --ddim_steps 50
```

### Class Distribution
```python
class_distribution = {
    'MI': 0.6,   # 60% Myocardial Infarction
    'PMO': 0.4   # 40% Persistent Microvascular Obstruction
}
```

## Advanced Options

### Conditional Generation
```python
# Generate with specific lesion characteristics
conditions = {
    'size': 'large',        # small, medium, large
    'intensity': 'high',    # low, medium, high
    'location': 'peripheral', # central, peripheral
    'texture': 'heterogeneous' # homogeneous, heterogeneous
}

synthetic = model.conditional_sample(
    normal_image=normal,
    conditions=conditions,
    ddim_steps=50
)
```

### Diversity Control
```python
# Control diversity vs quality trade-off
synthetic = model.sample(
    normal_image=normal,
    ddim_steps=50,
    eta=0.5,  # 0=deterministic, 1=maximum diversity
    temperature=0.8  # Lower = less diverse but higher quality
)
```

## Troubleshooting

### Common Issues

1. **Artifacts in Generated Images**
   - Reduce `ddim_steps` gradually (try 100)
   - Check mask quality
   - Verify model checkpoint is not corrupted

2. **Lesions Look Unrealistic**
   - Adjust histogram control parameters
   - Check training convergence of model
   - Increase diversity with higher `eta`

3. **Slow Generation**
   - Enable mixed precision: `--use_fp16`
   - Reduce batch size if memory limited
   - Use fewer DDIM steps (minimum 25)

4. **Memory Issues**
   ```python
   # Enable memory-efficient generation
   model.enable_xformers_memory_efficient_attention()
   ```

## Validation Before Training

Before using synthetic data for segmentation training:

1. **Visual QC**: Manually inspect 50-100 samples
2. **Histogram Analysis**: Compare with real pathological
3. **Radiologist Review**: Get clinical validation if possible
4. **Diversity Check**: Ensure variety in lesion characteristics

## Integration with STEP 3

The generated synthetic data will be used in combinations:
- `P`: Real pathological only (baseline)
- `P_N_prime`: Real + synthetic from normal (main)
- `P_P_prime`: Real + synthetic from pathological
- `P_P_prime_N_double_prime`: All combined

## Expected Generation Statistics

| Dataset | Normal Cases | Generated | Time | Storage |
|---------|--------------|-----------|------|---------|
| LIDC | 500 | 1000 | ~35 min | ~5 GB |
| LIDC | 500 | 2000 | ~70 min | ~10 GB |
| EMIDEC | 300 | 500 | ~20 min | ~2 GB |

## Next Step

After generation completes with quality validation, proceed to [STEP 3: Train Segmentation](../STEP3_train_segmentation/README.md)