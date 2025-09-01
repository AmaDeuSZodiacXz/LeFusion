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
from typing import Dict, Optional, Tuple, List
import random
from datetime import datetime
import cv2

from models.neuralsynth_core import NeuralSynthConfig, NeuralSynthDiffusion
from models.advanced_losses import NeuralSynthLoss, DiffusionLoss
from evaluation.advanced_metrics import ComprehensiveEvaluator


class EMIDECDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', 
                 image_size: int = 256, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[Dict]:
        samples = []
        
        split_file = self.data_dir / f'{self.split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                patient_ids = [line.strip() for line in f.readlines()]
        else:
            patient_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
            split_ratio = 0.8 if self.split == 'train' else 0.2
            split_idx = int(len(patient_dirs) * split_ratio)
            
            if self.split == 'train':
                patient_dirs = patient_dirs[:split_idx]
            else:
                patient_dirs = patient_dirs[split_idx:]
            
            patient_ids = [d.name for d in patient_dirs]
        
        for patient_id in patient_ids:
            patient_dir = self.data_dir / patient_id
            
            image_files = list(patient_dir.glob('*_image.nii.gz'))
            if not image_files:
                image_files = list(patient_dir.glob('*.nii.gz'))
                image_files = [f for f in image_files if 'mask' not in f.name.lower()]
            
            for image_file in image_files:
                mask_file = image_file.parent / image_file.name.replace('_image', '_mask')
                if not mask_file.exists():
                    mask_file = image_file.parent / image_file.name.replace('.nii.gz', '_mask.nii.gz')
                
                if mask_file.exists():
                    samples.append({
                        'image': image_file,
                        'mask': mask_file,
                        'patient_id': patient_id
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = nib.load(str(sample['image'])).get_fdata()
        mask = nib.load(str(sample['mask'])).get_fdata()
        
        image = self.normalize_image(image)
        
        if len(image.shape) == 3:
            num_slices = image.shape[2]
            valid_slices = []
            for i in range(num_slices):
                if np.sum(mask[:, :, i]) > 0:
                    valid_slices.append(i)
            
            if valid_slices:
                slice_idx = random.choice(valid_slices)
            else:
                slice_idx = random.randint(0, num_slices - 1)
            
            image = image[:, :, slice_idx]
            mask = mask[:, :, slice_idx]
        
        image = self.resize_image(image, self.image_size)
        mask = self.resize_mask(mask, self.image_size)
        
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)
        
        image = torch.FloatTensor(image).unsqueeze(0)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        
        mask = self.process_multi_class_mask(mask)
        
        return {
            'image': image,
            'mask': mask,
            'patient_id': sample['patient_id']
        }
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        
        percentile_99 = np.percentile(image, 99)
        percentile_1 = np.percentile(image, 1)
        
        image = np.clip(image, percentile_1, percentile_99)
        
        image = (image - percentile_1) / (percentile_99 - percentile_1 + 1e-8)
        
        return image
    
    def resize_image(self, image: np.ndarray, size: int) -> np.ndarray:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    
    def resize_mask(self, mask: np.ndarray, size: int) -> np.ndarray:
        return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    
    def process_multi_class_mask(self, mask: torch.Tensor) -> torch.Tensor:
        unique_values = torch.unique(mask)
        
        if len(unique_values) <= 2:
            return mask
        
        processed_mask = torch.zeros_like(mask)
        for i, val in enumerate(unique_values[1:], 1):
            processed_mask[mask == val] = min(i, 5)
        
        return processed_mask
    
    def apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
        
        if random.random() > 0.3:
            brightness = random.uniform(0.9, 1.1)
            image = np.clip(image * brightness, 0, 1)
        
        if random.random() > 0.3:
            contrast = random.uniform(0.9, 1.1)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast + mean, 0, 1)
        
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.01, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image, mask


class EMIDECTrainer:
    def __init__(self, config: NeuralSynthConfig, 
                 data_dir: str,
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs'):
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = NeuralSynthDiffusion(config).to(self.device)
        
        self.diffusion_loss = DiffusionLoss(loss_type='l2', use_weighted=True)
        self.synthesis_loss = NeuralSynthLoss(
            lambda_l1=1.0,
            lambda_perceptual=0.15,
            lambda_ssim=0.5,
            lambda_frequency=0.15,
            lambda_edge=0.25,
            lambda_lesion=0.4,
            lambda_adversarial=0.1
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=2e-4,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=2e-4,
            epochs=150,
            steps_per_epoch=100,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        self.scaler = GradScaler()
        
        self.train_dataset = EMIDECDataset(data_dir, split='train', augment=True)
        self.val_dataset = EMIDECDataset(data_dir, split='val', augment=False)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=6,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.checkpoint_dir = Path(checkpoint_dir) / 'neuralsynth_emidec'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir) / 'neuralsynth_emidec'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = ComprehensiveEvaluator()
        
        self.best_val_loss = float('inf')
        self.best_dice = 0.0
        self.best_metrics = {}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb.init(
            project="NeuralSynth-EMIDEC",
            name=f"run_{timestamp}",
            config=vars(config)
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_diff_loss = 0
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
                
                if epoch > 10 and batch_idx % 10 == 0:
                    with torch.no_grad():
                        synthesized = self.model.sample(
                            shape=(images.shape[0], 1, self.config.image_size, self.config.image_size),
                            lesion_mask=masks,
                            device=self.device
                        )
                    
                    synth_losses = self.synthesis_loss(
                        synthesized, images, masks
                    )
                    
                    loss = diff_loss + 0.1 * synth_losses['total']
                else:
                    loss = diff_loss
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += loss.item()
            total_diff_loss += diff_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'diff_loss': f'{diff_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            if batch_idx % 50 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/diff_loss': diff_loss.item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'batch': batch_idx
                })
        
        avg_loss = total_loss / num_batches
        avg_diff_loss = total_diff_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'train_diff_loss': avg_diff_loss
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_metrics = []
        
        progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
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
                
                if batch_idx < 10:
                    synthesized = self.model.sample(
                        shape=(1, 1, self.config.image_size, self.config.image_size),
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
                values = [m[key] for m in all_metrics 
                         if key in m and not np.isnan(m[key]) and not np.isinf(m[key])]
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
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': vars(self.config)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if metrics.get('val_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_path = self.checkpoint_dir / 'best_loss_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best loss model: {self.best_val_loss:.4f}")
        
        if metrics.get('dice', 0) > self.best_dice:
            self.best_dice = metrics['dice']
            self.best_metrics = metrics
            best_path = self.checkpoint_dir / 'best_dice_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best dice model: {self.best_dice:.4f}")
    
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
            
            all_metrics = {**train_metrics, **val_metrics}
            self.save_checkpoint(epoch, all_metrics)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            if 'dice' in val_metrics:
                print(f"Val Dice: {val_metrics['dice']:.4f}")
            if 'iou' in val_metrics:
                print(f"Val IoU: {val_metrics['iou']:.4f}")
            if 'ssim' in val_metrics:
                print(f"Val SSIM: {val_metrics['ssim']:.4f}")
            if 'psnr' in val_metrics:
                print(f"Val PSNR: {val_metrics['psnr']:.2f}")
            if 'hausdorff' in val_metrics:
                print(f"Val Hausdorff: {val_metrics['hausdorff']:.2f}")
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best Dice Score: {self.best_dice:.4f}")
        print("\nBest Metrics:")
        for key, value in self.best_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    config = NeuralSynthConfig(
        image_size=256,
        in_channels=1,
        out_channels=1,
        model_channels=128,
        num_res_blocks=4,
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
    
    trainer = EMIDECTrainer(
        config=config,
        data_dir="/Users/skb/Documents/LeFusion/data/EMIDEC",
        checkpoint_dir="./checkpoints",
        log_dir="./logs"
    )
    
    trainer.train(num_epochs=150)