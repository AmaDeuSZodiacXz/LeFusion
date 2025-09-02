#!/usr/bin/env python3
"""
Train NeuralSynth diffusion model on EMIDEC dataset.
This script trains the synthetic generation model for cardiac lesion synthesis
with support for multi-class lesions (MI and PMO).
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

from models.neuralsynth_core import (
    NeuralSynthDiffusion, 
    NeuralSynthConfig,
    AdaptiveNoiseScheduler
)
from models.advanced_losses import NeuralSynthLoss, DiffusionLoss
from utils.path_utils import get_project_paths


class EMIDECDataset(Dataset):
    """EMIDEC dataset loader for cardiac lesion images."""
    
    def __init__(self, data_dir, split='train', num_classes=2, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_classes = num_classes  # MI and PMO
        self.transform = transform
        
        # Get all .nii.gz files
        self.image_files = sorted(list(self.data_dir.glob('**/*image*.nii.gz')))
        self.mask_files = sorted(list(self.data_dir.glob('**/*mask*.nii.gz')))
        
        # For EMIDEC, we may have separate MI and PMO masks
        self.mi_mask_files = sorted(list(self.data_dir.glob('**/*MI*.nii.gz')))
        self.pmo_mask_files = sorted(list(self.data_dir.glob('**/*PMO*.nii.gz')))
        
        # Also check for normal cases (for background)
        normal_dir = self.data_dir.parent / 'Normal'
        self.normal_files = []
        if normal_dir.exists():
            self.normal_files = sorted(list(normal_dir.glob('**/*image*.nii.gz')))
        
        print(f"Found {len(self.image_files)} pathological images")
        print(f"Found {len(self.mask_files)} total masks")
        print(f"Found {len(self.mi_mask_files)} MI masks")
        print(f"Found {len(self.pmo_mask_files)} PMO masks")
        print(f"Found {len(self.normal_files)} normal images for background")
        
        # Train/val split (80/20)
        num_train = int(0.8 * len(self.image_files))
        if split == 'train':
            self.image_files = self.image_files[:num_train]
            self.mask_files = self.mask_files[:num_train]
            if self.mi_mask_files:
                self.mi_mask_files = self.mi_mask_files[:num_train]
            if self.pmo_mask_files:
                self.pmo_mask_files = self.pmo_mask_files[:num_train]
        else:
            self.image_files = self.image_files[num_train:]
            self.mask_files = self.mask_files[num_train:]
            if self.mi_mask_files:
                self.mi_mask_files = self.mi_mask_files[num_train:]
            if self.pmo_mask_files:
                self.pmo_mask_files = self.pmo_mask_files[num_train:]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load pathological image
        image_path = self.image_files[idx]
        image = nib.load(str(image_path)).get_fdata()
        
        # Load masks (handle multi-class)
        if self.mi_mask_files and self.pmo_mask_files:
            # Separate MI and PMO masks
            mi_mask = nib.load(str(self.mi_mask_files[idx])).get_fdata()
            pmo_mask = nib.load(str(self.pmo_mask_files[idx])).get_fdata()
            
            # Create multi-class mask (0: background, 1: MI, 2: PMO)
            mask = np.zeros_like(mi_mask)
            mask[mi_mask > 0] = 1
            mask[pmo_mask > 0] = 2
        else:
            # Single mask file
            mask_path = self.mask_files[idx]
            mask = nib.load(str(mask_path)).get_fdata()
        
        # Normalize image to [-1, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = 2 * image - 1
        
        # Get random normal image for background if available
        background = None
        if len(self.normal_files) > 0:
            normal_idx = np.random.randint(0, len(self.normal_files))
            normal_path = self.normal_files[normal_idx]
            background = nib.load(str(normal_path)).get_fdata()
            background = (background - background.min()) / (background.max() - background.min() + 1e-8)
            background = 2 * background - 1
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        if background is not None:
            background = torch.from_numpy(background).float()
        else:
            # Use image without lesion as background
            background = image * (mask == 0).float()
        
        # Handle dimensions (EMIDEC typically has smaller z-dimension)
        if len(image.shape) == 3:
            # For 3D, we can either:
            # 1. Take center slices
            # 2. Use the whole volume if small enough
            if image.shape[-1] <= 16:  # Small z-dimension, use whole volume
                image = image.permute(2, 0, 1)  # [Z, H, W]
                mask = mask.permute(2, 0, 1)
                background = background.permute(2, 0, 1)
            else:
                # Take center slice
                center = image.shape[-1] // 2
                image = image[:, :, center].unsqueeze(0)
                mask = mask[:, :, center].unsqueeze(0)
                background = background[:, :, center].unsqueeze(0)
        else:
            # 2D image
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            background = background.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'background': background,
            'path': str(image_path)
        }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        background = batch['background'].to(device)
        
        # Forward pass
        output = model(image, lesion_mask=mask, background=background)
        
        # Compute loss
        loss = criterion(output['predicted_noise'], output['target_noise'], output['timesteps'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update progress
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            image = batch['image'].to(device)
            mask = batch['mask'].to(device)
            background = batch['background'].to(device)
            
            output = model(image, lesion_mask=mask, background=background)
            loss = criterion(output['predicted_noise'], output['target_noise'], output['timesteps'])
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train NeuralSynth on EMIDEC dataset')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to EMIDEC pathological data')
    parser.add_argument('--output_dir', type=str, default='../checkpoints/emidec',
                       help='Output directory for checkpoints')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of lesion classes (2 for MI and PMO)')
    
    # Model arguments
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--use_adaptive_noise', action='store_true',
                       help='Use adaptive noise scheduling')
    parser.add_argument('--use_lesion_attention', action='store_true',
                       help='Use lesion-aware attention')
    parser.add_argument('--use_multi_scale', action='store_true',
                       help='Use multi-scale feature extraction')
    parser.add_argument('--model_channels', type=int, default=128,
                       help='Base channel count for UNet')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--val_interval', type=int, default=5,
                       help='Validation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Model save interval (epochs)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print("NeuralSynth Training on EMIDEC Dataset")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Features: Adaptive={args.use_adaptive_noise}, Attention={args.use_lesion_attention}, MultiScale={args.use_multi_scale}")
    print("=" * 60)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model configuration
    config = NeuralSynthConfig(
        image_size=256,  # Will be adjusted based on data
        in_channels=1,
        out_channels=1,
        model_channels=args.model_channels,
        num_timesteps=args.num_timesteps,
        lesion_classes=args.num_classes,  # MI and PMO
        use_adaptive_noise=args.use_adaptive_noise,
        use_lesion_attention=args.use_lesion_attention,
        use_multi_scale=args.use_multi_scale,
        channel_mult=[1, 2, 4],  # Smaller for EMIDEC
        attention_resolutions=[18, 9],  # Adjusted for EMIDEC
        num_heads=8,
        num_res_blocks=3
    )
    
    # Create model
    model = NeuralSynthDiffusion(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    train_dataset = EMIDECDataset(args.data_dir, split='train', num_classes=args.num_classes)
    val_dataset = EMIDECDataset(args.data_dir, split='val', num_classes=args.num_classes)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    criterion = DiffusionLoss(loss_type='l2', use_weighted=True)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume and args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch}")
    
    # Create TensorBoard writer
    log_dir = output_dir / 'logs' / datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validate
        if epoch % args.val_interval == 0:
            val_loss = validate(model, val_loader, criterion, device)
            writer.add_scalar('Loss/val', val_loss, epoch)
            
            print(f"\nEpoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / 'neuralsynth_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config
                }, best_path)
                print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f'neuralsynth_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Update learning rate
        scheduler.step()
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
    
    # Save final model
    final_path = output_dir / f'neuralsynth_epoch_{args.epochs}.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, final_path)
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()