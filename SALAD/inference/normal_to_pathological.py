#!/usr/bin/env python3
"""
SALAD Normal-to-Pathological Synthesis Pipeline
Generates synthetic pathological images from normal images following LeFusion approach
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import argparse
from tqdm import tqdm
import nibabel as nib
from PIL import Image
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.salad_core import NeuralSynthUNet, SALADConfig, AdaptiveNoiseScheduler


@dataclass
class SynthesisConfig:
    """Configuration for normal-to-pathological synthesis"""
    checkpoint_path: str
    normal_images_dir: str
    output_dir: str = "results/synthesis"
    mask_dir: Optional[str] = None  # Optional: use existing masks
    num_samples: int = 100
    batch_size: int = 4
    ddim_steps: int = 50
    device: str = "cuda"
    seed: int = 42
    lesion_intensity: float = 1.0  # Control lesion prominence
    save_format: str = "nifti"  # "nifti" or "png"


class NormalToPathologicalPipeline:
    """
    Pipeline for generating synthetic pathological images from normal images.
    Following LeFusion's approach:
    1. Use normal images as background (never synthesize background)
    2. Generate lesions only in masked regions
    3. Combine using smooth blending
    """
    
    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Load model and scheduler
        self.model = self._load_model()
        self.scheduler = self._setup_scheduler()
        
        # Load normal images
        self.normal_images = self._load_normal_images()
        
        # Setup output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Pipeline initialized")
        print(f"  - Model loaded from: {config.checkpoint_path}")
        print(f"  - Found {len(self.normal_images)} normal images")
        print(f"  - Output directory: {config.output_dir}")
    
    def _load_model(self) -> nn.Module:
        """Load trained SALAD model"""
        print(f"Loading model from {self.config.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        # Extract config
        if 'config' in checkpoint:
            model_config = SALADConfig(**checkpoint['config'])
        else:
            # Default config
            model_config = SALADConfig()
        
        # Create model
        model = NeuralSynthUNet(model_config)
        
        # Load weights (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove prefixes if needed
        if any(key.startswith('model.') for key in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        return model
    
    def _setup_scheduler(self) -> AdaptiveNoiseScheduler:
        """Setup adaptive noise scheduler"""
        scheduler = AdaptiveNoiseScheduler(num_timesteps=1000)
        
        # Load scheduler state if available in checkpoint
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif 'noise_scheduler' in checkpoint:
            # Handle nested scheduler state
            scheduler_state = checkpoint['noise_scheduler']
            if isinstance(scheduler_state, dict):
                scheduler.load_state_dict(scheduler_state)
        
        scheduler.to(self.device)
        return scheduler
    
    def _load_normal_images(self) -> List[Path]:
        """Load list of normal images"""
        normal_dir = Path(self.config.normal_images_dir)
        
        # Support multiple formats
        extensions = ['*.nii.gz', '*.nii', '*.npy', '*.png', '*.jpg']
        normal_images = []
        
        for ext in extensions:
            normal_images.extend(normal_dir.glob(ext))
            normal_images.extend(normal_dir.glob(f"**/{ext}"))  # Recursive search
        
        if not normal_images:
            raise ValueError(f"No normal images found in {normal_dir}")
        
        return sorted(normal_images)
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess an image"""
        if image_path.suffix in ['.nii', '.gz']:
            # Load NIfTI
            img = nib.load(image_path).get_fdata()
            # Take middle slice if 3D
            if img.ndim == 3:
                img = img[:, :, img.shape[2]//2]
        elif image_path.suffix == '.npy':
            # Load numpy array
            img = np.load(image_path)
        else:
            # Load regular image
            img = np.array(Image.open(image_path).convert('L'))
        
        # Normalize to [-1, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img * 2 - 1
        
        # Convert to tensor and add batch and channel dimensions
        img_tensor = torch.from_numpy(img).float()
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def _generate_random_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate random lesion mask"""
        h, w = shape[-2:]
        mask = torch.zeros(1, 1, h, w)
        
        # Random number of lesions (1-3)
        num_lesions = np.random.randint(1, 4)
        
        for _ in range(num_lesions):
            # Random position
            cx = np.random.randint(w//4, 3*w//4)
            cy = np.random.randint(h//4, 3*h//4)
            
            # Random size (5-30 pixels radius)
            radius = np.random.randint(5, 30)
            
            # Create circular mask
            y, x = np.ogrid[:h, :w]
            circle_mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            mask[0, 0][circle_mask] = 1
        
        return mask.to(self.device)
    
    @torch.no_grad()
    def synthesize_pathological(self, 
                               normal_image: torch.Tensor,
                               lesion_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Synthesize pathological image from normal image.
        Following LeFusion approach:
        1. Forward diffuse the normal image to get noisy background
        2. Reverse diffuse only in masked regions to generate lesions
        3. Combine with smooth blending
        """
        
        # Generate random mask if not provided
        if lesion_mask is None:
            lesion_mask = self._generate_random_mask(normal_image.shape)
        
        # Ensure correct dimensions
        if normal_image.dim() == 3:
            normal_image = normal_image.unsqueeze(0)
        if lesion_mask.dim() == 3:
            lesion_mask = lesion_mask.unsqueeze(0)
        
        batch_size = normal_image.shape[0]
        
        # Initialize noise for lesion regions
        lesion_noise = torch.randn_like(normal_image)
        
        # DDIM sampling for lesion generation
        timesteps = torch.linspace(999, 0, self.config.ddim_steps, dtype=torch.long, device=self.device)
        
        # Get alphas from scheduler
        betas = self.scheduler(torch.arange(1000, device=self.device))
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Start with noisy lesion
        x_lesion = lesion_noise
        
        for i, t in enumerate(tqdm(timesteps, desc="Generating lesion", leave=False)):
            t_batch = t.repeat(batch_size)
            
            # Forward diffuse the normal image to current timestep (for background)
            alpha_t = alphas_cumprod[t]
            noise_background = torch.randn_like(normal_image)
            x_background = torch.sqrt(alpha_t) * normal_image + torch.sqrt(1 - alpha_t) * noise_background
            
            # Combine lesion and background before denoising
            x_combined = x_lesion * lesion_mask + x_background * (1 - lesion_mask)
            
            # Predict noise with lesion conditioning
            noise_pred = self.model(x_combined, t_batch, lesion_mask)
            
            # DDIM step for lesion region only
            if i < len(timesteps) - 1:
                alpha_next = alphas_cumprod[timesteps[i+1]]
            else:
                alpha_next = torch.tensor(1.0)
            
            # Compute x0 prediction
            x0_pred = (x_combined - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # DDIM update
            sigma = 0  # Deterministic
            x_lesion = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next - sigma**2) * noise_pred
            
            # Apply mask to keep only lesion region
            x_lesion = x_lesion * lesion_mask + x_background * (1 - lesion_mask)
        
        # Final combination with smooth blending
        mask_smooth = self._smooth_mask(lesion_mask)
        synthetic = x_lesion * mask_smooth * self.config.lesion_intensity + normal_image * (1 - mask_smooth * self.config.lesion_intensity)
        
        # Normalize to [0, 1]
        synthetic = (synthetic + 1) / 2
        synthetic = torch.clamp(synthetic, 0, 1)
        
        return synthetic, lesion_mask
    
    def _smooth_mask(self, mask: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
        """Apply Gaussian smoothing for natural boundaries"""
        # Simple smoothing using average pooling
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        
        smoothed = torch.nn.functional.avg_pool2d(
            mask.float(), 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        
        return smoothed
    
    def save_results(self, synthetic: torch.Tensor, mask: torch.Tensor, 
                    normal_path: Path, idx: int):
        """Save synthetic image and mask"""
        # Prepare filenames
        base_name = f"synthetic_{idx:04d}_{normal_path.stem}"
        
        # Convert to numpy
        synthetic_np = synthetic.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        
        if self.config.save_format == "nifti":
            # Save as NIfTI
            synthetic_nifti = nib.Nifti1Image(synthetic_np, np.eye(4))
            mask_nifti = nib.Nifti1Image(mask_np, np.eye(4))
            
            nib.save(synthetic_nifti, Path(self.config.output_dir) / f"{base_name}.nii.gz")
            nib.save(mask_nifti, Path(self.config.output_dir) / f"{base_name}_mask.nii.gz")
        else:
            # Save as PNG
            synthetic_img = Image.fromarray((synthetic_np * 255).astype(np.uint8))
            mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
            
            synthetic_img.save(Path(self.config.output_dir) / f"{base_name}.png")
            mask_img.save(Path(self.config.output_dir) / f"{base_name}_mask.png")
    
    def run(self):
        """Run the synthesis pipeline"""
        print(f"\n{'='*60}")
        print(f"Starting Normal-to-Pathological Synthesis")
        print(f"{'='*60}")
        print(f"Generating {self.config.num_samples} synthetic pathological images")
        print(f"Using {len(self.normal_images)} normal images as backgrounds")
        
        generated_count = 0
        
        # Cycle through normal images if needed
        for i in range(self.config.num_samples):
            # Select normal image (cycle if necessary)
            normal_idx = i % len(self.normal_images)
            normal_path = self.normal_images[normal_idx]
            
            print(f"\n[{i+1}/{self.config.num_samples}] Processing: {normal_path.name}")
            
            # Load normal image
            normal_image = self._load_image(normal_path)
            
            # Generate synthetic pathological image
            synthetic, mask = self.synthesize_pathological(normal_image)
            
            # Save results
            self.save_results(synthetic, mask, normal_path, i)
            generated_count += 1
            
            print(f"  ✓ Generated synthetic pathological image")
        
        print(f"\n{'='*60}")
        print(f"✅ Synthesis Complete!")
        print(f"  - Generated: {generated_count} synthetic images")
        print(f"  - Saved to: {self.config.output_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="SALAD Normal-to-Pathological Synthesis")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained SALAD model checkpoint")
    parser.add_argument("--normal_dir", type=str, required=True,
                       help="Directory containing normal images")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="results/synthesis",
                       help="Output directory for synthetic images")
    parser.add_argument("--mask_dir", type=str, default=None,
                       help="Optional: Directory with lesion masks")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for generation")
    parser.add_argument("--ddim_steps", type=int, default=50,
                       help="Number of DDIM steps")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--lesion_intensity", type=float, default=1.0,
                       help="Lesion intensity (0-1)")
    parser.add_argument("--save_format", type=str, default="nifti",
                       choices=["nifti", "png"],
                       help="Output format")
    
    args = parser.parse_args()
    
    # Create config
    config = SynthesisConfig(
        checkpoint_path=args.checkpoint,
        normal_images_dir=args.normal_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        ddim_steps=args.ddim_steps,
        device=args.device,
        seed=args.seed,
        lesion_intensity=args.lesion_intensity,
        save_format=args.save_format
    )
    
    # Run pipeline
    pipeline = NormalToPathologicalPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()