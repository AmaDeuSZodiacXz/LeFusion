#!/usr/bin/env python3
"""
Train SALAD (Spatially-Aware Lesion Attention Diffusion) model on EMIDEC dataset using steps (like LeFusion).
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


class EMIDECDataset(Dataset):
    """EMIDEC dataset loader - ALL DATA FOR TRAINING."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # EMIDEC structure:
        # Scar/Image/*.nii.gz
        # Scar/Mask/*.nii.gz
        # Normal/Image/*.nii.gz
        
        # Get ALL pathological (scar) images
        scar_image_dir = self.data_dir / 'Scar' / 'Image'
        scar_mask_dir = self.data_dir / 'Scar' / 'Mask'
        
        if scar_image_dir.exists() and scar_mask_dir.exists():
            # Get all image files
            self.image_files = sorted(list(scar_image_dir.glob('*.nii.gz')))
            
            # Get corresponding mask files
            self.mask_files = []
            for img_path in self.image_files:
                # EMIDEC uses same filename for image and mask
                img_name = img_path.stem.replace('.nii', '')  # Remove .nii.gz
                mask_path = scar_mask_dir / f"{img_name}.nii.gz"
                if mask_path.exists():
                    self.mask_files.append(mask_path)
                else:
                    # Try alternative naming
                    mask_path = scar_mask_dir / f"{img_name}_mask.nii.gz"
                    if mask_path.exists():
                        self.mask_files.append(mask_path)
                    else:
                        self.mask_files.append(None)
        else:
            self.image_files = []
            self.mask_files = []
        
        # Filter out entries without masks
        valid_pairs = [(img, mask) for img, mask in zip(self.image_files, self.mask_files) if mask is not None]
        if valid_pairs:
            self.image_files, self.mask_files = zip(*valid_pairs)
            self.image_files = list(self.image_files)
            self.mask_files = list(self.mask_files)
        
        # Normal images for background
        normal_dir = self.data_dir / 'Normal' / 'Image'
        self.normal_files = []
        if normal_dir.exists():
            self.normal_files = sorted(list(normal_dir.glob('*.nii.gz')))
        
        print(f"Using ALL {len(self.image_files)} scar images for training")
        print(f"Found {len(self.mask_files)} corresponding masks")
        print(f"Found {len(self.normal_files)} normal images for background")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No scar images found in {scar_image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load scar image and mask
        image_path = self.image_files[idx]
        image = nib.load(str(image_path)).get_fdata()
        
        # Load mask
        mask_path = self.mask_files[idx]
        mask = nib.load(str(mask_path)).get_fdata()
        
        # Ensure same shape
        if image.shape != mask.shape:
            print(f"Warning: Shape mismatch - Image: {image.shape}, Mask: {mask.shape}")
            # Try to match dimensions
            min_shape = tuple(min(i, m) for i, m in zip(image.shape, mask.shape))
            image = image[:min_shape[0], :min_shape[1], :min_shape[2]] if len(min_shape) == 3 else image[:min_shape[0], :min_shape[1]]
            mask = mask[:min_shape[0], :min_shape[1], :min_shape[2]] if len(min_shape) == 3 else mask[:min_shape[0], :min_shape[1]]
        
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
            
            # Match dimensions with image
            if background.shape != image.shape:
                # Crop or pad to match
                if len(image.shape) == 3 and len(background.shape) == 3:
                    # Take center slices if needed
                    min_shape = tuple(min(i, b) for i, b in zip(image.shape, background.shape))
                    background = background[:min_shape[0], :min_shape[1], :min_shape[2]]
                    # Resize if still different
                    if background.shape != image.shape:
                        background = np.zeros_like(image)  # Fallback
                elif len(image.shape) == 2 and len(background.shape) == 3:
                    # Take center slice from 3D background
                    center = background.shape[-1] // 2
                    background = background[:, :, center]
                    if background.shape != image.shape:
                        background = np.zeros_like(image)
            
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
        
        # Handle dimensions - EMIDEC typically has smaller slices
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            background = background.unsqueeze(0)
        elif len(image.shape) == 3:
            # For 3D, take multiple slices or center slice
            if image.shape[-1] > 10:  # If many slices
                # Take center slice for 2D training
                center = image.shape[-1] // 2
                image = image[:, :, center].unsqueeze(0)
                mask = mask[:, :, center].unsqueeze(0)
                background = background[:, :, center].unsqueeze(0)
            else:
                # If few slices, take middle one
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


def train_steps(model, dataloader, optimizer, criterion, device, num_steps, save_every=1000, output_dir='../checkpoints/emidec_steps'):
    """Train for fixed number of steps (like LeFusion)."""
    model.train()
    
    # Create infinite data iterator
    data_iter = iter(dataloader)
    
    # Progress bar for steps
    progress_bar = tqdm(range(num_steps), desc='Training')
    
    # TensorBoard writer
    log_dir = Path(output_dir) / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir)
    
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
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), step)
        writer.add_scalar('Loss/avg', avg_loss, step)
        writer.add_scalar('GradNorm', grad_norm, step)
        
        # Update progress
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'step': step})
        
        # Save checkpoint
        if (step + 1) % save_every == 0:
            checkpoint_path = Path(output_dir) / f'checkpoint_step_{step+1}.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"\nSaved checkpoint at step {step+1}")
    
    writer.close()
    return losses


def main():
    parser = argparse.ArgumentParser(description='Train SALAD on EMIDEC by steps')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../../data/EMIDEC',
                       help='Path to EMIDEC data directory')
    parser.add_argument('--output_dir', type=str, default='../checkpoints/emidec_steps',
                       help='Output directory for checkpoints')
    
    # Training arguments (following LeFusion)
    parser.add_argument('--train_num_steps', type=int, default=50001,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=2,  # Smaller for EMIDEC
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--save_every', type=int, default=5000,
                       help='Save checkpoint every N steps')
    
    # Model arguments
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size for training')
    parser.add_argument('--model_channels', type=int, default=128,
                       help='Base channel count for UNet')
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
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SALAD Training on EMIDEC (Step-based)")
    print("Spatially-Aware Lesion Attention Diffusion")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training steps: {args.train_num_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Image size: {args.image_size}")
    print(f"Save every: {args.save_every} steps")
    print("=" * 60)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model configuration
    config = SALADConfig(
        image_size=args.image_size,
        in_channels=1,
        out_channels=1,
        model_channels=args.model_channels,
        num_timesteps=1000,
        use_adaptive_noise=args.use_adaptive_noise,
        use_lesion_attention=args.use_lesion_attention,
        use_multi_scale=args.use_multi_scale,
        channel_mult=[1, 2, 4, 8],
        attention_resolutions=[16, 8],
        num_heads=8,
        num_res_blocks=3
    )
    
    # Create model
    model = SALADDiffusion(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset (ALL data for training)
    dataset = EMIDECDataset(args.data_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    criterion = DiffusionLoss(loss_type='l2', use_weighted=True)
    
    # Resume if checkpoint provided
    start_step = 0
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step'] + 1
            print(f"Resumed from step {start_step}")
    
    # Train for fixed steps
    print("\nStarting training...")
    remaining_steps = args.train_num_steps - start_step
    if remaining_steps > 0:
        losses = train_steps(
            model, dataloader, optimizer, criterion, 
            device, remaining_steps, args.save_every, args.output_dir
        )
    else:
        print("Already completed training steps!")
        losses = []
    
    # Save final model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f'neuralsynth_emidec_final_{args.train_num_steps}steps.pth'
    torch.save({
        'step': args.train_num_steps,
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_path)
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\nTraining completed!")
    print(f"Final model saved to: {final_path}")
    if losses:
        print(f"Final average loss: {np.mean(losses[-100:]):.4f}")


if __name__ == "__main__":
    main()