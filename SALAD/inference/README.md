# SALAD Inference Pipeline

Clean inference pipeline for generating synthetic pathological images from normal images.

## Structure

```
inference/
├── inference.py          # Main inference script
├── configs/             
│   └── inference_config.yaml  # Configuration file
├── scripts/
│   └── run_inference.sh      # Bash runner script
└── README.md
```

## Usage

### Method 1: Direct Python
```bash
cd /content/LeFusion/SALAD

python inference/inference.py \
    --checkpoint checkpoints/lidc_steps/checkpoint_step_50000.pth \
    --normal_dir /content/LeFusion/data/LIDC/Normal/Image \
    --output_dir results/synthesis \
    --ddim_steps 50 \
    --device cuda
```

### Method 2: Using Shell Script
```bash
cd /content/LeFusion/SALAD/inference/scripts

./run_inference.sh \
    ../checkpoints/lidc_steps/checkpoint_step_50000.pth \
    /content/LeFusion/data/LIDC/Normal/Image \
    ../results/synthesis \
    50 \
    cuda
```

### Method 3: Using Config File
Edit `configs/inference_config.yaml` with your settings, then run:
```bash
python inference/inference.py --config inference/configs/inference_config.yaml
```

## Parameters

- `--checkpoint`: Path to trained model checkpoint
- `--normal_dir`: Directory containing normal images
- `--output_dir`: Output directory for synthetic images
- `--ddim_steps`: Number of DDIM sampling steps (50=fast, 1000=quality)
- `--device`: Device to use (cuda or cpu)

## Output

For each normal image, generates:
- `synthetic_XXXX_imagename.nii.gz` - Synthetic pathological image
- `synthetic_XXXX_imagename_mask.nii.gz` - Lesion mask

## Notes

- Processes ALL images in normal_dir (no duplicates)
- Automatically handles .nii.gz, .nii, .npy, .png formats
- Generates 1-3 random lesions per image
- Images are automatically resized to 256x256