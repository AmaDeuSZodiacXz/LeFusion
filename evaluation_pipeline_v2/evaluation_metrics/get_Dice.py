import nibabel as nib
import numpy as np
import torch


def compute_dice(preds, labels):
    """
    Compute Dice coefficient between predictions and labels.
    
    Args:
        preds: PyTorch tensor of predictions
        labels: PyTorch tensor of ground truth labels
    
    Returns:
        float: Dice coefficient score
    """
    preds = preds.numpy()
    labels = labels.numpy()
    
    # Add batch dimension if not present
    preds = preds[np.newaxis, :]
    labels = labels[np.newaxis, :]
    
    # Flatten the tensors
    predict = preds.reshape(preds.shape[0], -1)
    target = labels.reshape(labels.shape[0], -1)
    
    # Handle the case where both prediction and target are empty
    if np.sum(target) == 0 and np.sum(predict) == 0:
        return 1.0
    else:
        # Calculate intersection
        num = np.sum(np.multiply(predict, target), axis=1)
        # Calculate union
        den = np.sum(predict, axis=1) + np.sum(target, axis=1)
        # Calculate Dice
        dice = 2 * num / den
        
        return dice.mean()


def dice(predict_path, label_path):
    """
    Calculate Dice coefficient between predicted and ground truth segmentation.
    
    Args:
        predict_path: Path to predicted segmentation NIfTI file
        label_path: Path to ground truth segmentation NIfTI file
    
    Returns:
        float: Dice coefficient score
    """
    # Load NIfTI files
    predict = nib.load(predict_path)
    label = nib.load(label_path)
    
    # Get the data arrays
    pred = predict.get_fdata()
    lal = label.get_fdata()
    
    # Handle both 3D and 4D data
    if pred.ndim == 4:
        pred_sta = pred[0]  # Take first channel for 4D data
    else:
        pred_sta = pred  # Use as-is for 3D data
    
    if lal.ndim == 4:
        lal_sta = lal[0]  # Take first channel for 4D data
    else:
        lal_sta = lal  # Use as-is for 3D data
    
    # Convert to torch tensors
    lal_sta = torch.tensor(lal_sta).long()
    pred_sta = torch.tensor(pred_sta).long()
    
    # Compute Dice coefficient
    dice_score = compute_dice(pred_sta, lal_sta)
    
    return dice_score