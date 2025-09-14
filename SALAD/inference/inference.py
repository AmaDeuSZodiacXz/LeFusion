#!/usr/bin/env python3
"""
SALAD Inference Pipeline - Single Clean File
Generates synthetic pathological images from normal images
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import argparse
from tqdm import tqdm
import nibabel as nib
from PIL import Image

# Add parent directory (SALAD) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.salad_core import SALADUNet, SALADConfig, AdaptiveNoiseScheduler


class SALADInference:
    """SALAD inference pipeline for normal-to-pathological synthesis"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)
        self.scheduler = AdaptiveNoiseScheduler(num_timesteps=1000).to(self.device)
        
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load trained SALAD model"""
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config
        if 'config' in checkpoint:
            config = SALADConfig(**checkpoint['config'])
        else:
            config = SALADConfig()  # Default config
        
        # Create model
        model = SALADUNet(config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Clean keys
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
        
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device).eval()
        
        print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
        return model
    
    def load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess image to tensor"""
        if image_path.suffix in ['.nii', '.gz']:
            img = nib.load(image_path).get_fdata()
            if img.ndim == 3:  # Take middle slice if 3D
                img = img[:, :, img.shape[2]//2]
        elif image_path.suffix == '.npy':
            img = np.load(image_path)
        else:
            img = np.array(Image.open(image_path).convert('L'))
        
        # Resize to 256x256
        if img.shape != (256, 256):
            from scipy.ndimage import zoom
            zoom_factors = (256/img.shape[0], 256/img.shape[1])
            img = zoom(img, zoom_factors, order=1)
        
        # Normalize to [-1, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 2 - 1
        
        return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(self.device)
    
    def load_mask(self, mask_path: Path) -> torch.Tensor:
        """Load and preprocess mask to tensor"""
        if mask_path.exists():
            mask = nib.load(mask_path).get_fdata()
            if mask.ndim == 3:  # Take middle slice if 3D
                mask = mask[:, :, mask.shape[2]//2]
            
            # Resize to 256x256
            if mask.shape != (256, 256):
                from scipy.ndimage import zoom
                zoom_factors = (256/mask.shape[0], 256/mask.shape[1])
                mask = zoom(mask, zoom_factors, order=0)  # Nearest neighbor for mask
            
            # Binarize mask
            mask = (mask > 0).astype(np.float32)
            return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            # If no mask file, return empty mask (will generate random)
            return None
    
    def generate_mask(self) -> torch.Tensor:
        """Generate random lesion mask (fallback if no mask provided)"""
        mask = torch.zeros(1, 1, 256, 256, device=self.device)
        
        # Random lesions (1-3)
        for _ in range(np.random.randint(1, 4)):
            cx = np.random.randint(64, 192)
            cy = np.random.randint(64, 192)
            radius = np.random.randint(5, 25)
            
            y, x = np.ogrid[:256, :256]
            circle = (x - cx)**2 + (y - cy)**2 <= radius**2
            mask[0, 0][circle] = 1
        
        return mask
    
    @torch.no_grad()
    def synthesize(self, normal_image: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                  ddim_steps: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize pathological from normal image using provided or generated mask"""
        # Use provided mask or generate random one
        if mask is None:
            mask = self.generate_mask()
        
        # DDIM sampling
        timesteps = torch.linspace(999, 0, ddim_steps, dtype=torch.long, device=self.device)
        betas = self.scheduler(torch.arange(1000, device=self.device))
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Initialize with noise in lesion regions
        x = torch.randn_like(normal_image) * mask + normal_image * (1 - mask)
        
        for i, t in enumerate(timesteps):
            # Background at current timestep
            alpha_t = alphas_cumprod[t]
            noise_bg = torch.randn_like(normal_image)
            x_bg = torch.sqrt(alpha_t) * normal_image + torch.sqrt(1 - alpha_t) * noise_bg
            
            # Combine lesion and background
            x_combined = x * mask + x_bg * (1 - mask)
            
            # Predict noise
            noise_pred = self.model(x_combined, t.unsqueeze(0), mask)
            
            # DDIM update
            alpha_next = alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else torch.tensor(1.0)
            x0_pred = (x_combined - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1, 1)
            x = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred
            x = x * mask + x_bg * (1 - mask)
        
        # Smooth blending
        kernel_size = 7
        mask_smooth = torch.nn.functional.avg_pool2d(mask, kernel_size, stride=1, padding=kernel_size//2)
        synthetic = x * mask_smooth + normal_image * (1 - mask_smooth)
        
        # Normalize to [0, 1]
        synthetic = torch.clamp((synthetic + 1) / 2, 0, 1)
        
        return synthetic, mask
    
    def process_directory(self, normal_dir: str, output_dir: str, ddim_steps: int = 50):
        """Process all normal images in directory"""
        normal_dir = Path(normal_dir)
        output_dir = Path(output_dir)
        
        # Create Image and Mask subdirectories
        image_dir = output_dir / "Image"
        mask_dir = output_dir / "Mask"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if normal_dir has Image/Mask subdirectory (LIDC structure)
        if (normal_dir / "Image").exists():
            normal_image_dir = normal_dir / "Image"
            normal_mask_dir = normal_dir / "Mask"
            print(f"Found Image subdirectory, using: {normal_image_dir}")
            print(f"Found Mask subdirectory, using: {normal_mask_dir}")
        else:
            normal_image_dir = normal_dir
            normal_mask_dir = None
        
        # Find all images
        image_files = []
        for ext in ['*.nii.gz', '*.nii', '*.npy', '*.png']:
            image_files.extend(normal_image_dir.glob(ext))
            # Don't recurse if we're already in Image directory
            if normal_image_dir == normal_dir:
                image_files.extend(normal_image_dir.glob(f"**/{ext}"))
        
        if not image_files:
            raise ValueError(f"No images found in {normal_dir}")
        
        image_files = sorted(set(image_files))  # Remove duplicates
        print(f"\nFound {len(image_files)} normal images")
        
        # Process each image
        for idx, img_path in enumerate(tqdm(image_files, desc="Synthesizing")):
            # Load normal image
            normal = self.load_image(img_path)
            
            # Try to load corresponding mask (CVol -> CMask)
            mask = None
            if normal_mask_dir and normal_mask_dir.exists():
                # Convert image name to mask name
                img_name = img_path.stem.replace('.nii', '')
                mask_name = img_name.replace('CVol', 'CMask') + '.nii.gz'
                mask_path = normal_mask_dir / mask_name
                
                if mask_path.exists():
                    mask = self.load_mask(mask_path)
                    print(f"  Using mask: {mask_name}")
                else:
                    print(f"  No mask found for {img_name}, will generate random mask")
            
            # Generate synthetic with actual or random mask
            synthetic, used_mask = self.synthesize(normal, mask, ddim_steps)
            
            # Save results
            self.save_results(synthetic, used_mask, img_path.stem, idx, image_dir, mask_dir)
        
        print(f"\n✓ Generated {len(image_files)} synthetic images")
        print(f"   Images saved to: {image_dir}")
        print(f"   Masks saved to: {mask_dir}")
    
    def save_results(self, synthetic: torch.Tensor, mask: torch.Tensor, 
                    name: str, idx: int, image_dir: Path, mask_dir: Path):
        """Save synthetic image and mask in separate directories"""
        synthetic_np = synthetic.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        
        # Use original name format for consistency with LIDC structure
        # Remove 'synthetic_' prefix and index for cleaner names
        file_name = f"{name}.nii.gz"
        
        # Save as NIfTI
        synthetic_nifti = nib.Nifti1Image(synthetic_np, np.eye(4))
        mask_nifti = nib.Nifti1Image(mask_np, np.eye(4))
        
        # Save image and mask in their respective directories
        nib.save(synthetic_nifti, image_dir / file_name)
        nib.save(mask_nifti, mask_dir / file_name)


def main():
    parser = argparse.ArgumentParser(description="SALAD Inference Pipeline")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--normal_dir", required=True, help="Normal images directory")
    parser.add_argument("--output_dir", default="results/synthesis", help="Output directory")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps (50=fast, 1000=quality)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize and run
    inference = SALADInference(args.checkpoint, args.device)
    inference.process_directory(args.normal_dir, args.output_dir, args.ddim_steps)


if __name__ == "__main__":
    main()