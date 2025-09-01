"""
NeuralSynth Diffusion Model Training
====================================
Train the core diffusion model for synthetic pathological image generation.
Preserves LeFusion's background preservation approach.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))


class NeuralSynthDiffusion(nn.Module):
    """
    NeuralSynth Diffusion Model with Background Preservation.
    Core innovation: Preserves anatomical background 100% like LeFusion.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.in_channels = config.get('in_channels', 1)
        self.out_channels = config.get('out_channels', 1)
        self.hidden_dims = config.get('hidden_dims', [64, 128, 256, 512])
        
        # Diffusion parameters
        self.num_timesteps = config.get('num_timesteps', 1000)
        self.beta_start = config.get('beta_start', 0.0001)
        self.beta_end = config.get('beta_end', 0.02)
        
        # Adaptive noise scheduling (NeuralSynth innovation)
        self.adaptive_betas = nn.Parameter(
            torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        )
        
        # Build UNet backbone
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Lesion-aware attention (NeuralSynth innovation)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dims[-1],
            num_heads=8,
            batch_first=True
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256)
        )
    
    def _build_encoder(self):
        """Build encoder with multi-scale features."""
        layers = []
        in_ch = self.in_channels
        
        for out_ch in self.hidden_dims:
            layers.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU()
            ))
            in_ch = out_ch
        
        return nn.ModuleList(layers)
    
    def _build_decoder(self):
        """Build decoder with skip connections."""
        layers = []
        reversed_dims = list(reversed(self.hidden_dims))
        
        for i in range(len(reversed_dims) - 1):
            in_ch = reversed_dims[i]
            out_ch = reversed_dims[i + 1]
            
            layers.append(nn.Sequential(
                nn.ConvTranspose3d(in_ch * 2, out_ch, 2, stride=2),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU()
            ))
        
        # Final layer
        layers.append(nn.Conv3d(self.hidden_dims[0], self.out_channels, 1))
        
        return nn.ModuleList(layers)
    
    def forward(self, x, t, mask=None, background=None):
        """
        Forward pass with background preservation.
        
        Args:
            x: Input image [B, C, D, H, W]
            t: Timestep [B]
            mask: Lesion mask [B, 1, D, H, W]
            background: Original background to preserve [B, C, D, H, W]
        """
        # Time embedding
        t_emb = self.time_embed(t.float().unsqueeze(-1))
        
        # Encoder
        skip_features = []
        h = x
        for encoder_block in self.encoder:
            h = encoder_block(h)
            skip_features.append(h)
        
        # Apply attention at bottleneck
        b, c, d, h_dim, w = h.shape
        h_flat = h.view(b, c, -1).permute(0, 2, 1)
        h_attended, _ = self.attention(h_flat, h_flat, h_flat)
        h = h_attended.permute(0, 2, 1).view(b, c, d, h_dim, w)
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder[:-1]):
            skip = skip_features[-(i + 2)]
            h = torch.cat([h, skip], dim=1)
            h = decoder_block(h)
        
        # Final output
        output = self.decoder[-1](h)
        
        # CRITICAL: Preserve background (LeFusion's key insight)
        if mask is not None and background is not None:
            output = output * mask + background * (1 - mask)
        
        return output


class MedicalImageDataset(Dataset):
    """Dataset for medical images with pathological and normal cases."""
    
    def __init__(self, data_dir, dataset_type='lidc', split='train'):
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.split = split
        
        # Load file lists
        self.normal_files = list((self.data_dir / 'normal').glob('*.npz'))
        self.pathological_files = list((self.data_dir / 'pathological').glob('*.npz'))
        
        print(f"Loaded {len(self.normal_files)} normal, {len(self.pathological_files)} pathological cases")
    
    def __len__(self):
        return len(self.pathological_files) * 2  # Use each pathological case twice
    
    def __getitem__(self, idx):
        # Get pathological case
        path_idx = idx % len(self.pathological_files)
        path_data = np.load(self.pathological_files[path_idx])
        
        # Get random normal case for background
        normal_idx = np.random.randint(len(self.normal_files))
        normal_data = np.load(self.normal_files[normal_idx])
        
        # Extract data
        pathological_image = path_data['image'].astype(np.float32)
        lesion_mask = path_data['mask'].astype(np.float32)
        normal_image = normal_data['image'].astype(np.float32)
        
        # Normalize
        pathological_image = (pathological_image - pathological_image.mean()) / (pathological_image.std() + 1e-8)
        normal_image = (normal_image - normal_image.mean()) / (normal_image.std() + 1e-8)
        
        # Add channel dimension
        if len(pathological_image.shape) == 3:
            pathological_image = pathological_image[np.newaxis, ...]
            normal_image = normal_image[np.newaxis, ...]
            lesion_mask = lesion_mask[np.newaxis, ...]
        
        return {
            'pathological': torch.from_numpy(pathological_image),
            'normal': torch.from_numpy(normal_image),
            'mask': torch.from_numpy(lesion_mask)
        }


class DiffusionTrainer:
    """Trainer for NeuralSynth diffusion model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model
        self.model = NeuralSynthDiffusion(config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-6)
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Logging
        self.writer = SummaryWriter(log_dir=config.get('log_dir', 'logs'))
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, train_loader, val_loader, num_epochs):
        """Train the diffusion model."""
        print(f"\n🚀 Starting training on {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {self.config.get('batch_size', 4)}")
        print(f"   Learning rate: {self.config.get('learning_rate', 1e-4)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                pathological = batch['pathological'].to(self.device)
                normal = batch['normal'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Sample timesteps
                t = torch.randint(0, self.model.num_timesteps, (pathological.shape[0],), device=self.device)
                
                # Add noise (forward diffusion)
                noise = torch.randn_like(pathological)
                noisy_image = self.add_noise(pathological, noise, t)
                
                # Forward pass with background preservation
                pred_noise = self.model(noisy_image, t, mask, normal)
                
                # Compute loss
                loss = self.compute_loss(pred_noise, noise, mask)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Log to tensorboard
                if batch_idx % 10 == 0:
                    self.writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_idx)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Log epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            self.writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            elif (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
    
    def add_noise(self, x, noise, t):
        """Add noise to image (forward diffusion process)."""
        sqrt_alpha_cumprod = self.get_sqrt_alpha_cumprod(t)
        sqrt_one_minus_alpha_cumprod = self.get_sqrt_one_minus_alpha_cumprod(t)
        
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.view(-1, 1, 1, 1, 1)
        
        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
    
    def get_sqrt_alpha_cumprod(self, t):
        """Get sqrt of cumulative product of alphas."""
        alphas = 1 - self.model.adaptive_betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        return torch.sqrt(alpha_cumprod[t])
    
    def get_sqrt_one_minus_alpha_cumprod(self, t):
        """Get sqrt of (1 - cumulative product of alphas)."""
        alphas = 1 - self.model.adaptive_betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        return torch.sqrt(1 - alpha_cumprod[t])
    
    def compute_loss(self, pred, target, mask):
        """
        Compute loss with focus on lesion regions.
        
        Loss = MSE + λ1*L1 + λ2*Lesion_Loss
        """
        # Basic reconstruction loss
        mse_loss = self.mse_loss(pred, target)
        l1_loss = self.l1_loss(pred, target)
        
        # Lesion-focused loss (higher weight on lesion regions)
        lesion_loss = self.mse_loss(pred * mask, target * mask) * 2.0
        
        # Combined loss
        total_loss = mse_loss + 0.1 * l1_loss + lesion_loss
        
        return total_loss
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                pathological = batch['pathological'].to(self.device)
                normal = batch['normal'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                t = torch.randint(0, self.model.num_timesteps, (pathological.shape[0],), device=self.device)
                noise = torch.randn_like(pathological)
                noisy_image = self.add_noise(pathological, noise, t)
                
                pred_noise = self.model(noisy_image, t, mask, normal)
                loss = self.compute_loss(pred_noise, noise, mask)
                
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            print(f"💾 Saving best model (val_loss: {val_loss:.4f})")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            print(f"💾 Saving checkpoint at epoch {epoch+1}")
        
        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description='Train NeuralSynth Diffusion Model')
    parser.add_argument('--dataset', type=str, choices=['lidc', 'emidec'], required=True,
                        help='Dataset to train on')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'dataset': args.dataset,
        'in_channels': 1,
        'out_channels': 1,
        'hidden_dims': [64, 128, 256, 512],
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-6,
        'device': args.device,
        'checkpoint_dir': f'{args.checkpoint_dir}/{args.dataset}',
        'log_dir': f'{args.log_dir}/{args.dataset}'
    }
    
    # Create datasets
    train_dataset = MedicalImageDataset(args.data_dir, args.dataset, 'train')
    val_dataset = MedicalImageDataset(args.data_dir, args.dataset, 'val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize trainer
    trainer = DiffusionTrainer(config)
    
    # Train model
    trainer.train(train_loader, val_loader, args.epochs)
    
    print("\n✅ Training completed!")


if __name__ == '__main__':
    main()