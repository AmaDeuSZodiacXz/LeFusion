"""
Segmentation Model Training for NeuralSynth
===========================================
Trains segmentation models on synthetic + real data.
Compatible with LeFusion's DiffTumor framework.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from monai.networks.nets import UNet, SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped
)
from monai.data import CacheDataset, decollate_batch
from monai.utils import set_determinism
from tqdm import tqdm

# Add parent directories
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))


class SegmentationDataset(Dataset):
    """Dataset for segmentation training with real + synthetic data."""
    
    def __init__(self, data_config):
        self.data_config = data_config
        self.data_files = self._prepare_data_files()
        
        # Define transforms
        self.transforms = self._get_transforms()
    
    def _prepare_data_files(self):
        """Prepare list of data files based on configuration."""
        files = []
        
        # Real pathological data (P)
        if 'real_path' in self.data_config:
            real_dir = Path(self.data_config['real_path'])
            for file in real_dir.glob('*.npz'):
                files.append({
                    'image': str(file),
                    'label': str(file),  # Mask is in same file
                    'type': 'real'
                })
        
        # Synthetic from pathological (P_P_prime)
        if 'synthetic_p_path' in self.data_config:
            synth_p_dir = Path(self.data_config['synthetic_p_path'])
            for file in synth_p_dir.glob('*.npz'):
                files.append({
                    'image': str(file),
                    'label': str(file),
                    'type': 'synthetic_p'
                })
        
        # Synthetic from normal (P_N_prime) - NeuralSynth's main output
        if 'synthetic_n_path' in self.data_config:
            synth_n_dir = Path(self.data_config['synthetic_n_path'])
            for file in synth_n_dir.glob('*.npz'):
                files.append({
                    'image': str(file),
                    'label': str(file),
                    'type': 'synthetic_n'
                })
        
        print(f"Loaded {len(files)} files for training")
        return files
    
    def _get_transforms(self):
        """Get data transforms for training."""
        if self.data_config.get('mode') == 'train':
            return Compose([
                LoadImaged(keys=['image', 'label']),
                EnsureChannelFirstd(keys=['image', 'label']),
                ScaleIntensityd(keys=['image']),
                RandCropByPosNegLabeld(
                    keys=['image', 'label'],
                    label_key='label',
                    spatial_size=self.data_config.get('patch_size', [64, 64, 32]),
                    pos=1,
                    neg=1,
                    num_samples=4
                ),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
                RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=3),
                EnsureTyped(keys=['image', 'label'])
            ])
        else:
            return Compose([
                LoadImaged(keys=['image', 'label']),
                EnsureChannelFirstd(keys=['image', 'label']),
                ScaleIntensityd(keys=['image']),
                EnsureTyped(keys=['image', 'label'])
            ])
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        return self.transforms(self.data_files[idx])


class SegmentationTrainer:
    """Trainer for segmentation models."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set determinism for reproducibility
        set_determinism(seed=config.get('seed', 42))
        
        # Initialize model
        self.model = self._build_model().to(self.device)
        
        # Loss and metrics
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 200)
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric = -1
        self.best_metric_epoch = -1
    
    def _build_model(self):
        """Build segmentation model based on configuration."""
        model_name = self.config.get('model_name', 'nnunet')
        
        if model_name.lower() == 'nnunet':
            return UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,  # Background + lesion
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm='batch'
            )
        elif model_name.lower() == 'swinunetr':
            return SwinUNETR(
                img_size=self.config.get('patch_size', [64, 64, 32]),
                in_channels=1,
                out_channels=2,
                feature_size=48,
                use_checkpoint=True
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train(self, train_loader, val_loader, num_epochs):
        """Train the segmentation model."""
        print(f"\n🎯 Starting segmentation training")
        print(f"   Model: {self.config.get('model_name', 'nnunet')}")
        print(f"   Dataset: {self.config.get('dataset', 'unknown')}")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        
        for epoch in range(num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            self.model.train()
            epoch_loss = 0
            step = 0
            
            train_bar = tqdm(train_loader, desc="Training")
            for batch_data in train_bar:
                step += 1
                
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                train_bar.set_postfix({'loss': loss.item()})
            
            # Learning rate scheduling
            self.scheduler.step()
            
            epoch_loss /= step
            print(f"   Training loss: {epoch_loss:.4f}")
            
            # Validation
            if (epoch + 1) % self.config.get('val_interval', 5) == 0:
                metric = self.validate(val_loader)
                
                # Save best model
                if metric > self.best_metric:
                    self.best_metric = metric
                    self.best_metric_epoch = epoch + 1
                    self.save_checkpoint(epoch + 1, metric, is_best=True)
                    print(f"   🏆 New best metric: {metric:.4f}")
                
                # Regular checkpoint
                if (epoch + 1) % self.config.get('save_interval', 20) == 0:
                    self.save_checkpoint(epoch + 1, metric, is_best=False)
        
        print(f"\n✅ Training completed!")
        print(f"   Best metric: {self.best_metric:.4f} at epoch {self.best_metric_epoch}")
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for val_data in val_bar:
                val_inputs = val_data["image"].to(self.device)
                val_labels = val_data["label"].to(self.device)
                
                # Forward pass
                val_outputs = self.model(val_inputs)
                
                # Compute metric
                val_outputs = [i for i in decollate_batch(val_outputs)]
                val_labels = [i for i in decollate_batch(val_labels)]
                
                self.dice_metric(y_pred=val_outputs, y=val_labels)
            
            # Aggregate metrics
            metric = self.dice_metric.aggregate().item()
            self.dice_metric.reset()
        
        print(f"   Validation Dice: {metric:.4f}")
        return metric
    
    def save_checkpoint(self, epoch, metric, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metric': metric,
            'config': self.config
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        print(f"   💾 Saved checkpoint: {path.name}")


def prepare_data_config(args):
    """Prepare data configuration based on combination type."""
    base_dir = Path(args.data_dir)
    synthetic_dir = base_dir / 'synthetic_data' / args.dataset / args.method
    real_dir = base_dir.parent / 'data' / args.dataset.upper()
    
    config = {
        'dataset': args.dataset,
        'combination': args.combination,
        'patch_size': [64, 64, 32] if args.dataset == 'lidc' else [72, 72, 10]
    }
    
    # Configure data paths based on combination
    if args.combination == 'P':
        # Real pathological only
        config['real_path'] = real_dir / 'pathological'
    
    elif args.combination == 'P_P_prime':
        # Real + synthetic from pathological
        config['real_path'] = real_dir / 'pathological'
        config['synthetic_p_path'] = synthetic_dir / 'P_P_prime'
    
    elif args.combination == 'P_N_prime':
        # Real + synthetic from normal (NeuralSynth's main output)
        config['real_path'] = real_dir / 'pathological'
        config['synthetic_n_path'] = synthetic_dir / 'P_N_prime'
    
    elif args.combination == 'P_P_prime_N_double_prime':
        # All combined
        config['real_path'] = real_dir / 'pathological'
        config['synthetic_p_path'] = synthetic_dir / 'P_P_prime'
        config['synthetic_n_path'] = synthetic_dir / 'P_N_prime'
        config['synthetic_n2_path'] = synthetic_dir / 'P_N_double_prime'
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Segmentation Model')
    parser.add_argument('--dataset', type=str, choices=['lidc', 'emidec'], required=True,
                        help='Dataset to train on')
    parser.add_argument('--data-dir', type=str, default='/Users/skb/Documents/LeFusion/NeuralSynth',
                        help='Base directory containing data')
    parser.add_argument('--method', type=str, default='neuralsynth',
                        choices=['neuralsynth', 'lefusion', 'lefusion_h', 'lefusion_h_diffmask'],
                        help='Synthesis method')
    parser.add_argument('--combination', type=str, default='P_N_prime',
                        choices=['P', 'P_P_prime', 'P_N_prime', 'P_P_prime_N_double_prime'],
                        help='Data combination for training')
    parser.add_argument('--model', type=str, default='nnunet',
                        choices=['nnunet', 'swinunetr'],
                        help='Segmentation model architecture')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--val-interval', type=int, default=5,
                        help='Validation interval')
    parser.add_argument('--save-interval', type=int, default=20,
                        help='Checkpoint save interval')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Prepare data configuration
    data_config = prepare_data_config(args)
    data_config['mode'] = 'train'
    
    # Training configuration
    train_config = {
        'dataset': args.dataset,
        'method': args.method,
        'combination': args.combination,
        'model_name': args.model,
        'patch_size': data_config['patch_size'],
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'device': args.device,
        'checkpoint_dir': f'{args.checkpoint_dir}/{args.dataset}/{args.method}/{args.combination}/{args.model}',
        'val_interval': args.val_interval,
        'save_interval': args.save_interval,
        'seed': args.seed
    }
    
    # Create datasets
    train_dataset = SegmentationDataset(data_config)
    
    # For validation, use same data with different transforms
    val_data_config = data_config.copy()
    val_data_config['mode'] = 'val'
    val_dataset = SegmentationDataset(val_data_config)
    
    # Create data loaders
    train_loader = DataLoader(
        CacheDataset(train_dataset.data_files, train_dataset.transforms, cache_rate=0.5),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        CacheDataset(val_dataset.data_files, val_dataset.transforms, cache_rate=1.0),
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = SegmentationTrainer(train_config)
    
    # Train model
    trainer.train(train_loader, val_loader, args.epochs)


if __name__ == '__main__':
    main()