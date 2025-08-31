"""
Evaluation metrics for medical image segmentation.
Based on the official LeFusion implementation from https://github.com/M3DV/LeFusion
"""

from .get_Dice import dice, compute_dice
from .get_NSD import nsd

__all__ = ['dice', 'compute_dice', 'nsd']