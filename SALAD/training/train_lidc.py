#!/usr/bin/env python3
"""
Train SALAD (Spatially-Aware Lesion Attention Diffusion) model on LIDC dataset.
This script trains the synthetic generation model with adaptive noise scheduling
and spatially-aware lesion attention for lung nodule synthesis.
"""

import os
import sys
import argparse
import json
import yaml
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
    """LIDC dataset loader for lung nodule images."""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # LIDC data structure:
        # Pathological/Image/LIDC-IDRI-XXXX/LIDC-IDRI-XXXX_Vol_000.nii.gz
        # Pathological/Mask/LIDC-IDRI-XXXX/LIDC-IDRI-XXXX_Mask_000.nii.gz
        
        image_dir = self.data_dir / 'Image'
        mask_dir = self.data_dir / 'Mask'
        
        if image_dir.exists() and mask_dir.exists():
            # Get all image files
            self.image_files = sorted(list(image_dir.glob('*/*.nii.gz')))
            
            # Get corresponding mask files
            self.mask_files = []
            for img_path in self.image_files:
                # Extract patient ID from image path
                patient_id = img_path.parent.name  # e.g., LIDC-IDRI-0001
                # Find corresponding mask
                mask_pattern = mask_dir / patient_id / f"{patient_id}_Mask_*.nii.gz"
                mask_matches = list(mask_dir.glob(patient_id + f"/{patient_id}_Mask_*.nii.gz"))
                if mask_matches:
                    self.mask_files.append(mask_matches[0])
                else:
                    # If no mask found, use a dummy placeholder
                    self.mask_files.append(None)
        else:
            # Fallback to old pattern
            self.image_files = sorted(list(self.data_dir.glob('**/*Vol*.nii.gz')))
            self.mask_files = sorted(list(self.data_dir.glob('**/*Mask*.nii.gz')))
        
        # Filter out entries without masks
        valid_pairs = [(img, mask) for img, mask in zip(self.image_files, self.mask_files) if mask is not None]
        if valid_pairs:
            self.image_files, self.mask_files = zip(*valid_pairs)
            self.image_files = list(self.image_files)
            self.mask_files = list(self.mask_files)
        
        # Also check for normal cases (for background)
        normal_dir = self.data_dir.parent / 'Normal' / 'Image'
        self.normal_files = []
        if normal_dir.exists():
            self.normal_files = sorted(list(normal_dir.glob('*.nii.gz')))
        
        print(f"Found {len(self.image_files)} pathological images")
        print(f"Found {len(self.mask_files)} masks")
        print(f"Found {len(self.normal_files)} normal images for background")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.data_dir}. Please check the data directory structure.")
        
        if len(self.image_files) != len(self.mask_files):
            print("Warning: Number of images and masks don't match!")
        
        # Train/val split (80/20)
        num_train = max(1, int(0.8 * len(self.image_files)))
        if split == 'train':
            self.image_files = self.image_files[:num_train]
            self.mask_files = self.mask_files[:num_train] if self.mask_files else []
        else:
            self.image_files = self.image_files[num_train:]
            self.mask_files = self.mask_files[num_train:] if self.mask_files else []
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load pathological image and mask
        image_path = self.image_files[idx]
        image = nib.load(str(image_path)).get_fdata()
        
        # Load mask if available
        if idx < len(self.mask_files):
            mask_path = self.mask_files[idx]
            mask = nib.load(str(mask_path)).get_fdata()
        else:
            # Create a dummy mask if no mask available
            mask = np.zeros_like(image)
        
        # Normalize image to [-1, 1] with safeguards
        img_min, img_max = image.min(), image.max()
        if img_max - img_min < 1e-8:
            # Handle constant images
            image = np.zeros_like(image)
        else:
            image = (image - img_min) / (img_max - img_min)
            image = 2 * image - 1
        
        # Check for NaN or Inf
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            print(f"Warning: NaN or Inf detected in image, replacing with zeros")
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Binary mask
        mask = (mask > 0).astype(np.float32)
        
        # Get random normal image for background if available
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
            
            # Check for NaN or Inf
            if np.any(np.isnan(background)) or np.any(np.isinf(background)):
                print(f"Warning: NaN or Inf detected in background, replacing with zeros")
                background = np.nan_to_num(background, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Extract histogram from lesion region (like LeFusion!)
        lesion_pixels = image[mask > 0]
        if len(lesion_pixels) > 0:
            hist, _ = np.histogram(lesion_pixels, bins=16, range=(-1, 1))
            hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        else:
            hist = np.ones(16, dtype=np.float32) / 16

        # Convert to tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        hist = torch.from_numpy(hist).float()

        if background is not None:
            background = torch.from_numpy(background).float()
        else:
            # Use image without lesion as background (rough approximation)
            background = image * (1 - mask)
        
        # Add channel dimension
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            background = background.unsqueeze(0)
        elif len(image.shape) == 3:
            # For 3D, take center slice for now
            center = image.shape[-1] // 2
            image = image[:, :, center].unsqueeze(0)
            mask = mask[:, :, center].unsqueeze(0)
            background = background[:, :, center].unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'background': background,
            'histogram': hist,  # Added histogram conditioning!
            'path': str(image_path)
        }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, gradient_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    skipped_updates = 0
    successful_updates = 0

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        image = batch['image'].to(device)
        mask = batch['mask'].to(device)
        background = batch['background'].to(device)

        # Forward pass
        output = model(image, lesion_mask=mask, background=background)

        # Compute loss with scaling to prevent gradient explosion
        loss = criterion(output['predicted_noise'], output['target_noise'], output['timesteps'])
        loss = loss * 0.1  # Scale down loss to prevent gradient explosion

        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss detected at batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            skipped_updates += 1
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients - more aggressive clipping for initial training
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Log gradient norm periodically
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: grad_norm={grad_norm:.2f}, loss={loss.item()*10:.4f}")

        # Only skip if gradient is truly exploding
        if grad_norm > 100.0:
            if batch_idx % 10 == 0:  # Only print every 10th warning to reduce spam
                print(f"Warning: Large gradient norm {grad_norm:.2f}, skipping update...")
            optimizer.zero_grad()
            skipped_updates += 1
            continue

        optimizer.step()
        successful_updates += 1
        
        # Update progress (multiply by 10 to show actual loss scale)
        total_loss += loss.item() * 10
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'grad': f'{grad_norm:.1f}',
            'skip': f'{skipped_updates}/{batch_idx+1}'
        })

    print(f"\nEpoch {epoch} complete: {successful_updates} successful updates, {skipped_updates} skipped")
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
    parser = argparse.ArgumentParser(description='Train SALAD on LIDC dataset')

    # Config file support
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (JSON or YAML)')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../data/LIDC/Pathological',
                       help='Path to LIDC pathological data (relative or absolute)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/lidc',
                       help='Output directory for checkpoints')
    
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
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--val_interval', type=int, default=5,
                       help='Validation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Model save interval (epochs)')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    parser.add_argument('--gradient_skip_threshold', type=float, default=100.0,
                       help='Skip update if gradient norm exceeds this')
    
    # DDIM arguments
    parser.add_argument('--ddim_steps', type=int, default=50,
                       help='Number of DDIM sampling steps')
    
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

    # Load config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent.parent / config_path

        if config_path.exists():
            print(f"Loading config from: {config_path}")

            # Load config based on file extension
            if config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

            # Override args with config values
            for section in ['data', 'training', 'output', 'hardware', 'model']:
                if section in config:
                    for key, value in config[section].items():
                        if hasattr(args, key):
                            setattr(args, key, value)

    # Handle relative paths - make them relative to SALAD directory
    salad_dir = Path(__file__).parent.parent

    # Fix data_dir path
    if args.data_dir:
        data_path = Path(args.data_dir)
        if not data_path.is_absolute():
            data_path = salad_dir / data_path
        args.data_dir = str(data_path.resolve())
    else:
        # Default path
        args.data_dir = str((salad_dir / '../data/LIDC/Pathological').resolve())

    # Fix output_dir path
    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = salad_dir / output_path
    output_dir = output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print("SALAD Training on LIDC Dataset")
    print("Spatially-Aware Lesion Attention Diffusion")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
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
    config = SALADConfig(
        image_size=256,  # Will be adjusted based on data
        in_channels=1,
        out_channels=1,
        model_channels=args.model_channels,
        num_timesteps=args.num_timesteps,
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

    # Initialize model weights properly to prevent gradient explosion
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    
    # Create datasets
    train_dataset = LIDCDataset(args.data_dir, split='train')
    val_dataset = LIDCDataset(args.data_dir, split='val')
    
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
    
    # Create optimizer with lower learning rate to prevent gradient explosion
    # Start with very low LR and increase if stable
    initial_lr = args.learning_rate * 0.1  # Start with 1/10th of requested LR
    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    print(f"Starting with learning rate: {initial_lr}")
    
    criterion = DiffusionLoss(loss_type='l2', use_weighted=True)
    
    # Learning rate scheduler with warmup
    # Use warmup to gradually increase LR and prevent initial explosion
    warmup_epochs = min(5, args.epochs // 10)

    def warmup_lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from 0.1 to 1.0
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    print(f"Using {warmup_epochs} warmup epochs")
    
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

    # Start with aggressive gradient clipping, then relax as training stabilizes
    for epoch in range(start_epoch, args.epochs):
        # Adaptive gradient clipping - start strict, relax over time
        if epoch < warmup_epochs:
            gradient_clip = 0.1  # Very strict during warmup
        elif epoch < 10:
            gradient_clip = 0.5  # Still strict early on
        elif epoch < 20:
            gradient_clip = 1.0  # Normal clipping
        else:
            gradient_clip = 5.0  # Relaxed clipping

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, gradient_clip)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Validate
        if epoch % args.val_interval == 0:
            val_loss = validate(model, val_loader, criterion, device)
            writer.add_scalar('Loss/val', val_loss, epoch)
            
            print(f"\nEpoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / 'salad_best.pth'
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
            checkpoint_path = output_dir / f'salad_epoch_{epoch}.pth'
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
    final_path = output_dir / f'salad_epoch_{args.epochs}.pth'
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