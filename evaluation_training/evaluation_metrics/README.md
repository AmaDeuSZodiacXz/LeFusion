# LeFusion Evaluation Metrics

This directory contains the official evaluation metrics implementation from the LeFusion repository (https://github.com/M3DV/LeFusion).

## Metrics Included

### 1. Dice Coefficient (`get_Dice.py`)
- Measures the overlap between predicted and ground truth segmentations
- Returns a value between 0 and 1, where 1 indicates perfect overlap
- Handles edge cases where both masks are empty (returns 1.0)

### 2. Normalized Surface Dice - NSD (`get_NSD.py`)
- Measures the overlap of surfaces within a specified tolerance
- Default tolerance: 1.0mm (as used in the paper)
- Based on MONAI's `compute_surface_dice` function
- Considers voxel spacing from the NIfTI file's affine matrix

## Usage

### Basic Usage

```python
# When running from evaluation_training directory
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from evaluation_metrics import dice, nsd

# Calculate Dice coefficient
dice_score = dice('path/to/prediction.nii.gz', 'path/to/ground_truth.nii.gz')

# Calculate NSD with 1mm tolerance (paper default)
nsd_score = nsd('path/to/prediction.nii.gz', 
                'path/to/ground_truth.nii.gz',
                'path/to/reference.nii.gz',  # For spacing information
                tolerance=[1.0])
```

### Integration with Evaluation Pipeline

The metrics are integrated into `evaluation/evaluate_models.py`:

```python
from evaluation.evaluate_models import ModelEvaluator

evaluator = ModelEvaluator()

# Uses official metrics by default
dice = evaluator.calculate_dice(pred, gt, use_official=True)
nsd = evaluator.calculate_nsd(pred, gt, spacing_mm=(1,1,1), tolerance=1.0, use_official=True)

# Can also use file-based evaluation directly
result = evaluator.evaluate_single_case(pred_path, gt_path, use_file_metrics=True)
```

### Comparing Implementations

To verify consistency between different metric implementations:

```bash
# Run from evaluation_training directory
cd evaluation_training

# Compare with synthetic test cases
python evaluation/compare_metrics.py --tolerance 1.0

# Compare with specific files
python evaluation/compare_metrics.py \
    --test-files /path/to/pred.nii.gz /path/to/gt.nii.gz \
    --tolerance 1.0
```

## Requirements

- nibabel
- numpy
- torch
- monai

## Notes

1. **Data Format**: Both metrics expect NIfTI files (.nii.gz) as input
2. **Binary Masks**: The implementation automatically binarizes the input masks
3. **3D/4D Support**: Updated to handle both 3D and 4D data (takes first channel for 4D)
4. **Spacing**: NSD requires accurate voxel spacing information from the NIfTI header
5. **Tolerance**: Default tolerance is 1.0mm to match the paper methodology

## Differences from Original Implementation

This version has been updated to:
- Handle both 3D and 4D data properly
- Ensure tensor dimensions are correct for MONAI functions
- Integrate seamlessly with the evaluation pipeline
- Provide fallback options when official metrics fail

## Testing

Run the test suite to verify the metrics work correctly:

```bash
# Run from evaluation_training directory
cd evaluation_training

# Test the official metrics
python test_official_metrics.py
```

This will test:
- Perfect match cases
- No overlap cases  
- Partial overlap cases
- Edge cases (empty masks)
- Integration with the evaluation pipeline

## Citation

If you use these metrics, please cite the original LeFusion paper:
```
@article{lefusion2024,
  title={LeFusion: Synthesizing Pathological Medical Images using Controllable Diffusion Models},
  author={...},
  year={2024}
}
```