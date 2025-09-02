#!/usr/bin/env python3
"""
Train SALAD (Spatially-Aware Lesion Attention Diffusion) model on LIDC dataset using steps (like LeFusion).
No validation set - just train for fixed number of steps.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import nibabel as nib
from einops import rearrange

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.salad_core import (
    SALADDiffusion, 
    SALADConfig,
    AdaptiveNoiseScheduler
)
from models.advanced_losses import DiffusionLoss


class LIDCDataset(Dataset):
    """LIDC dataset loader for lung nodule images - ALL DATA FOR TRAINING."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get ALL pathological images (no train/val split)
        image_dir = self.data_dir / 'Image'
        mask_dir = self.data_dir / 'Mask'
        
        if image_dir.exists() and mask_dir.exists():
            self.image_files = sorted(list(image_dir.glob('*/*.nii.gz')))
            
            self.mask_files = []
            for img_path in self.image_files:
                patient_id = img_path.parent.name
                mask_pattern = mask_dir / patient_id / f"{patient_id}_Mask_*.nii.gz"
                mask_matches = list(mask_dir.glob(patient_id + f"/{patient_id}_Mask_*.nii.gz"))
                if mask_matches:
                    self.mask_files.append(mask_matches[0])
                else:
                    self.mask_files.append(None)
        
        # Filter out entries without masks
        valid_pairs = [(img, mask) for img, mask in zip(self.image_files, self.mask_files) if mask is not None]
        if valid_pairs:
            self.image_files, self.mask_files = zip(*valid_pairs)
            self.image_files = list(self.image_files)
            self.mask_files = list(self.mask_files)
        
        # Normal images for background
        normal_dir = self.data_dir.parent / 'Normal' / 'Image'
        self.normal_files = []
        if normal_dir.exists():
            self.normal_files = sorted(list(normal_dir.glob('*.nii.gz')))
        
        print(f"Using ALL {len(self.image_files)} pathological images for training")
        print(f"Found {len(self.normal_files)} normal images for background")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load pathological image and mask
        image_path = self.image_files[idx]
        image = nib.load(str(image_path)).get_fdata()
        
        # Load mask
        mask_path = self.mask_files[idx]
        mask = nib.load(str(mask_path)).get_fdata()
        
        # Robust normalization
        img_min, img_max = image.min(), image.max()
        if img_max - img_min < 1e-8:
            image = np.zeros_like(image)
        else:
            image = (image - img_min) / (img_max - img_min)
            image = 2 * image - 1
        
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Binary mask
        mask = (mask > 0).astype(np.float32)
        
        # Get random normal image for background
        background = None
        if len(self.normal_files) > 0:
            normal_idx = np.random.randint(0, len(self.normal_files))
            normal_path = self.normal_files[normal_idx]
            background = nib.load(str(normal_path)).get_fdata()
            bg_min, bg_max = background.min(), background.max()
            if bg_max - bg_min < 1e-8:
                background = np.zeros_like(background)
            else:
                background = (background - bg_min) / (bg_max - bg_min)
                background = 2 * background - 1
            
            if np.any(np.isnan(background)) or np.any(np.isinf(background)):
                background = np.nan_to_num(background, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        if background is not None:
            background = torch.from_numpy(background).float()
        else:
            background = image * (1 - mask)
        
        # Handle dimensions
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            background = background.unsqueeze(0)
        elif len(image.shape) == 3:
            # Take center slice for 2D training
            center = image.shape[-1] // 2
            image = image[:, :, center].unsqueeze(0)
            mask = mask[:, :, center].unsqueeze(0)
            background = background[:, :, center].unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'background': background,
            'path': str(image_path)
        }


def train_steps(model, dataloader, optimizer, criterion, device, num_steps, save_every=1000):
    """Train for fixed number of steps (like LeFusion)."""
    model.train()
    
    # Create infinite data iterator
    data_iter = iter(dataloader)
    
    # Progress bar for steps
    progress_bar = tqdm(range(num_steps), desc='Training')
    
    losses = []
    for step in progress_bar:
        # Get next batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Move to device
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        background = batch['background'].to(device)
        
        # Forward pass
        output = model(image, lesion_mask=mask, background=background)
        
        # Compute loss
        loss = criterion(output['predicted_noise'], output['target_noise'], output['timesteps'])
        
        # Skip if NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at step {step}, skipping...")
            optimizer.zero_grad()
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # Skip if gradients too large
        if grad_norm > 10.0:
            print(f"Warning: Large gradient norm {grad_norm:.2f}, skipping...")
            optimizer.zero_grad()
            continue
        
        optimizer.step()
        
        # Track loss
        losses.append(loss.item())
        avg_loss = np.mean(losses[-100:])  # Running average of last 100 steps
        
        # Update progress
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'step': step})
        
        # Save checkpoint
        if (step + 1) % save_every == 0:
            checkpoint_path = Path(f'../checkpoints/lidc_steps/checkpoint_step_{step+1}.pth')
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"\nSaved checkpoint at step {step+1}")
    
    return losses


def main():
    parser = argparse.ArgumentParser(description='Train SALAD by steps (like LeFusion)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../../data/LIDC/Pathological',
                       help='Path to LIDC pathological data')
    parser.add_argument('--output_dir', type=str, default='../checkpoints/lidc_steps',
                       help='Output directory for checkpoints')
    
    # Training arguments (following LeFusion)
    parser.add_argument('--train_num_steps', type=int, default=50001,
                       help='Number of training steps (like LeFusion)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--save_every', type=int, default=5000,
                       help='Save checkpoint every N steps')
    
    # Model arguments
    parser.add_argument('--use_adaptive_noise', action='store_true', default=True,
                       help='Use adaptive noise scheduling')
    parser.add_argument('--use_lesion_attention', action='store_true', default=True,
                       help='Use lesion-aware attention')
    parser.add_argument('--use_multi_scale', action='store_true', default=True,
                       help='Use multi-scale feature extraction')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SALAD Training (Step-based like LeFusion)")
    print("Spatially-Aware Lesion Attention Diffusion")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Training steps: {args.train_num_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Save every: {args.save_every} steps")
    print("=" * 60)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model configuration
    config = SALADConfig(
        image_size=256,
        in_channels=1,
        out_channels=1,
        model_channels=128,
        num_timesteps=1000,
        use_adaptive_noise=args.use_adaptive_noise,
        use_lesion_attention=args.use_lesion_attention,
        use_multi_scale=args.use_multi_scale
    )
    
    # Create model
    model = SALADDiffusion(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset (ALL data for training)
    dataset = LIDCDataset(args.data_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    criterion = DiffusionLoss(loss_type='l2', use_weighted=True)
    
    # Train for fixed steps
    print("\nStarting training...")
    losses = train_steps(
        model, dataloader, optimizer, criterion, 
        device, args.train_num_steps, args.save_every
    )
    
    # Save final model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f'salad_final_{args.train_num_steps}steps.pth'
    torch.save({
        'step': args.train_num_steps,
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    
    print("\nTraining completed!")
    print(f"Final model saved to: {final_path}")
    print(f"Final average loss: {np.mean(losses[-100:]):.4f}")


if __name__ == "__main__":
    main()