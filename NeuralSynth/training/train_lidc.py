import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb
import json
from pathlib import Path
import nibabel as nib
from typing import Dict, Optional, Tuple
import random
from datetime import datetime

from models.neuralsynth_core import NeuralSynthConfig, NeuralSynthDiffusion
from models.advanced_losses import NeuralSynthLoss, DiffusionLoss
from evaluation.advanced_metrics import ComprehensiveEvaluator


class LIDCDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        self.image_dir = self.data_dir / 'images' / split
        self.mask_dir = self.data_dir / 'masks' / split
        
        self.image_files = sorted(list(self.image_dir.glob('*.nii.gz')))
        self.mask_files = sorted(list(self.mask_dir.glob('*.nii.gz')))
        
        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images and masks don't match: {len(self.image_files)} vs {len(self.mask_files)}"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        image = nib.load(str(image_path)).get_fdata()
        mask = nib.load(str(mask_path)).get_fdata()
        
        image = self.normalize_image(image)
        
        if len(image.shape) == 3:
            slice_idx = random.randint(0, image.shape[2] - 1)
            image = image[:, :, slice_idx]
            mask = mask[:, :, slice_idx]
        
        image = torch.FloatTensor(image).unsqueeze(0)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return {
            'image': image,
            'mask': mask,
            'filename': image_path.stem
        }
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image


class NeuralSynthTrainer:
    def __init__(self, config: NeuralSynthConfig, 
                 train_dir: str, 
                 val_dir: str,
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs'):
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = NeuralSynthDiffusion(config).to(self.device)
        
        self.diffusion_loss = DiffusionLoss(loss_type='l2', use_weighted=True)
        self.synthesis_loss = NeuralSynthLoss()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.scaler = GradScaler()
        
        self.train_dataset = LIDCDataset(train_dir, split='train')
        self.val_dataset = LIDCDataset(val_dir, split='val')
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.checkpoint_dir = Path(checkpoint_dir) / 'neuralsynth_lidc'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir) / 'neuralsynth_lidc'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = ComprehensiveEvaluator()
        
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb.init(
            project="NeuralSynth-LIDC",
            name=f"run_{timestamp}",
            config=vars(config)
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(images, masks)
                
                diff_loss = self.diffusion_loss(
                    outputs['predicted_noise'],
                    outputs['target_noise'],
                    outputs['timesteps']
                )
                
                loss = diff_loss
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
            
            if batch_idx % 100 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'batch': batch_idx
                })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_metrics = []
        
        progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                with autocast():
                    outputs = self.model(images, masks)
                    
                    diff_loss = self.diffusion_loss(
                        outputs['predicted_noise'],
                        outputs['target_noise'],
                        outputs['timesteps']
                    )
                    
                    loss = diff_loss
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches <= 5:
                    synthesized = self.model.sample(
                        shape=(1, 1, 256, 256),
                        lesion_mask=masks[:1],
                        device=self.device
                    )
                    
                    metrics = self.evaluator.evaluate_all(
                        synthesized[0, 0].cpu().numpy(),
                        images[0, 0].cpu().numpy(),
                        masks[0, 0].cpu().numpy() > 0.5,
                        masks[0, 0].cpu().numpy() > 0.5
                    )
                    all_metrics.append(metrics)
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if not np.isnan(m[key]) and not np.isinf(m[key])]
                if values:
                    avg_metrics[key] = np.mean(values)
                else:
                    avg_metrics[key] = 0.0
            
            wandb.log({
                **{f'val/{k}': v for k, v in avg_metrics.items()},
                'val/loss': avg_loss,
                'epoch': epoch
            })
            
            return {'val_loss': avg_loss, **avg_metrics}
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': vars(self.config)
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if metrics.get('val_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            self.best_metrics = metrics
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
    
    def train(self, num_epochs: int):
        print(f"Starting training on {self.device}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            self.scheduler.step()
            
            all_metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, all_metrics)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            if 'dice' in val_metrics:
                print(f"Val Dice: {val_metrics['dice']:.4f}")
            if 'ssim' in val_metrics:
                print(f"Val SSIM: {val_metrics['ssim']:.4f}")
            if 'psnr' in val_metrics:
                print(f"Val PSNR: {val_metrics['psnr']:.2f}")
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print("Best Metrics:")
        for key, value in self.best_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    config = NeuralSynthConfig(
        image_size=256,
        in_channels=1,
        out_channels=1,
        model_channels=128,
        num_res_blocks=3,
        attention_resolutions=[16, 8],
        dropout=0.1,
        channel_mult=[1, 2, 4, 8],
        num_heads=8,
        use_scale_shift_norm=True,
        num_timesteps=1000,
        beta_schedule="cosine",
        lesion_classes=5,
        use_adaptive_noise=True,
        use_multi_scale=True,
        use_lesion_attention=True
    )
    
    trainer = NeuralSynthTrainer(
        config=config,
        train_dir="/Users/skb/Documents/LeFusion/data/LIDC",
        val_dir="/Users/skb/Documents/LeFusion/data/LIDC",
        checkpoint_dir="./checkpoints",
        log_dir="./logs"
    )
    
    trainer.train(num_epochs=100)