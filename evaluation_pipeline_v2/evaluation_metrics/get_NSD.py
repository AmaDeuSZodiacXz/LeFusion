from monai.metrics import compute_surface_dice
import nibabel as nib
import numpy as np
import torch


def nsd(predict_path, label_path, space_path, tolerance=[1.0]):
    """
    Calculate Normalized Surface Dice (NSD) between predicted and ground truth segmentation.
    
    Args:
        predict_path: Path to predicted segmentation NIfTI file
        label_path: Path to ground truth segmentation NIfTI file
        space_path: Path to reference NIfTI file for spacing information
        tolerance: List of tolerance values in mm (default: [1.0])
    
    Returns:
        torch.Tensor: NSD score(s) for the given tolerance value(s)
    """
    # Load NIfTI files
    predict = nib.load(predict_path)
    label = nib.load(label_path)
    space = nib.load(space_path)
    
    # Extract spacing from the affine matrix
    affine = space.affine
    spacing = affine[:3, :3].diagonal()
    spacing = np.abs(spacing)
    
    # Get the data arrays
    pred = predict.get_fdata()
    lal = label.get_fdata()
    
    # Convert both to tensors with proper dimensions
    pred = torch.tensor(pred).int()
    pred = pred.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    lal = torch.tensor(lal).int()
    lal = lal.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Compute surface dice with the given tolerance
    surface_dice = compute_surface_dice(
        y_pred=pred,
        y=lal,
        spacing=spacing,
        class_thresholds=tolerance
    )
    
    return surface_dice