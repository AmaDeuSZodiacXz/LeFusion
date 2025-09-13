#!/usr/bin/env python3
"""
SALAD Inference Pipeline - Streamlined Synthetic Data Generation
Clean implementation for generating synthetic pathological images from checkpoint
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import argparse
from tqdm import tqdm
import nibabel as nib
from dataclasses import dataclass

# Import SALAD core components
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.salad_core import NeuralSynthUNet, SALADConfig, AdaptiveNoiseScheduler


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    checkpoint_path: str
    output_dir: str = "results/synthesis"
    num_samples: int = 100
    batch_size: int = 8
    ddim_steps: int = 50
    device: str = "cuda"
    seed: int = 42
    guidance_scale: float = 1.0
    image_size: Tuple[int, int] = (256, 256)
    save_format: str = "nifti"  # "nifti" or "png"


class SALADInference:
    """Streamlined SALAD inference pipeline"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load model
        self.model = self._load_model()
        self.scheduler = self._setup_scheduler()
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def _load_model(self) -> nn.Module:
        """Load SALAD model from checkpoint"""
        print(f"Loading model from {self.config.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        # Extract config from checkpoint or use defaults
        if 'config' in checkpoint:
            model_config = SALADConfig(**checkpoint['config'])
        else:
            model_config = SALADConfig()
        
        # Create model
        model = NeuralSynthUNet(model_config)
        
        # Load weights - handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'model.' prefix if present
        if any(key.startswith('model.') for key in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        
        # Load the state dict with strict=False to handle mismatches
        model.load_state_dict(state_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        return model
    
    def _setup_scheduler(self) -> AdaptiveNoiseScheduler:
        """Setup adaptive noise scheduler"""
        scheduler = AdaptiveNoiseScheduler(num_timesteps=1000)
        scheduler.to(self.device)
        return scheduler
    
    @torch.no_grad()
    def ddim_sample(self, 
                    shape: Tuple[int, ...],
                    lesion_mask: Optional[torch.Tensor] = None,
                    background: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        DDIM sampling for fast inference
        
        Args:
            shape: Output shape (B, C, H, W)
            lesion_mask: Optional lesion mask for conditioning
            background: Optional background to preserve (LeFusion approach)
        
        Returns:
            Generated samples
        """
        batch_size = shape[0]
        
        # Initialize from noise
        x = torch.randn(shape, device=self.device)
        
        # Setup timesteps for DDIM
        timesteps = torch.linspace(999, 0, self.config.ddim_steps, dtype=torch.long, device=self.device)
        
        # Get alphas from scheduler
        betas = self.scheduler(torch.arange(1000, device=self.device))
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # DDIM sampling loop
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
            t_batch = t.repeat(batch_size)
            
            # Predict noise
            noise_pred = self.model(x, t_batch, lesion_mask)
            
            # DDIM step
            alpha_t = alphas_cumprod[t]
            alpha_prev = alphas_cumprod[timesteps[i-1]] if i > 0 else torch.tensor(1.0)
            
            # Predict x0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # Direction pointing to xt
            dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
            
            # Compute xt-1
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt
            
            # Blend with background if provided (LeFusion approach)
            if background is not None and lesion_mask is not None:
                # Gaussian smoothing for natural boundaries
                mask_smooth = self._smooth_mask(lesion_mask)
                x = x * mask_smooth + background * (1 - mask_smooth)
        
        return x
    
    def _smooth_mask(self, mask: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
        """Apply Gaussian smoothing to mask for natural boundaries"""
        # Simple box filter approximation for efficiency
        kernel_size = int(4 * sigma + 1)
        padding = kernel_size // 2
        
        # Create simple smoothing kernel
        mask_float = mask.float()
        smoothed = torch.nn.functional.avg_pool2d(
            mask_float, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        
        return smoothed
    
    def generate_synthetic_batch(self, 
                                 backgrounds: Optional[torch.Tensor] = None,
                                 masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate a batch of synthetic images
        
        Args:
            backgrounds: Optional background images to preserve
            masks: Optional lesion masks for conditioning
        
        Returns:
            Batch of synthetic images
        """
        shape = (self.config.batch_size, 1, *self.config.image_size)
        
        # Generate samples
        samples = self.ddim_sample(shape, masks, backgrounds)
        
        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        return samples
    
    def save_results(self, images: torch.Tensor, batch_idx: int):
        """Save generated images"""
        for i, img in enumerate(images):
            idx = batch_idx * self.config.batch_size + i
            
            if self.config.save_format == "nifti":
                # Save as NIfTI
                img_np = img.cpu().numpy().squeeze()
                nifti_img = nib.Nifti1Image(img_np, np.eye(4))
                save_path = Path(self.config.output_dir) / f"synthetic_{idx:04d}.nii.gz"
                nib.save(nifti_img, save_path)
            else:
                # Save as PNG
                from PIL import Image
                img_np = (img.cpu().numpy().squeeze() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                save_path = Path(self.config.output_dir) / f"synthetic_{idx:04d}.png"
                img_pil.save(save_path)
    
    def run(self):
        """Run the complete inference pipeline"""
        print(f"Starting SALAD synthetic data generation")
        print(f"Generating {self.config.num_samples} samples")
        print(f"Output directory: {self.config.output_dir}")
        
        num_batches = (self.config.num_samples + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_idx in range(num_batches):
            # Calculate actual batch size for last batch
            current_batch_size = min(
                self.config.batch_size,
                self.config.num_samples - batch_idx * self.config.batch_size
            )
            
            # Adjust shape if needed for last batch
            if current_batch_size < self.config.batch_size:
                shape = (current_batch_size, 1, *self.config.image_size)
                samples = self.ddim_sample(shape)
            else:
                samples = self.generate_synthetic_batch()
            
            # Save results
            self.save_results(samples, batch_idx)
            
            print(f"Batch {batch_idx + 1}/{num_batches} completed")
        
        print(f"Synthesis complete! Results saved to {self.config.output_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SALAD Synthetic Data Generation")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to SALAD model checkpoint")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="results/synthesis",
                       help="Output directory for synthetic images")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for generation")
    parser.add_argument("--ddim_steps", type=int, default=50,
                       help="Number of DDIM steps (50 for fast, 1000 for quality)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                       help="Guidance scale for conditional generation")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256],
                       help="Image size (height width)")
    parser.add_argument("--save_format", type=str, default="nifti",
                       choices=["nifti", "png"],
                       help="Output format for saved images")
    
    args = parser.parse_args()
    
    # Create config
    config = InferenceConfig(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        ddim_steps=args.ddim_steps,
        device=args.device,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        image_size=tuple(args.image_size),
        save_format=args.save_format
    )
    
    # Run inference
    pipeline = SALADInference(config)
    pipeline.run()


if __name__ == "__main__":
    main()