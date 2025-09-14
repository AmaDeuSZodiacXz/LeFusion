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
import yaml
import json

# Add parent directory (SALAD) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.salad_core import SALADUNet, SALADConfig, AdaptiveNoiseScheduler


class SALADInference:
    """SALAD inference pipeline for normal-to-pathological synthesis"""

    def __init__(self, checkpoint_path: str, device: str = "cuda", test_txt_path: str = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)
        self.scheduler = AdaptiveNoiseScheduler(num_timesteps=1000).to(self.device)

        # Load test cases (like LeFusion)
        self.test_cases = []
        if test_txt_path and Path(test_txt_path).exists():
            with open(test_txt_path, 'r') as f:
                self.test_cases = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.test_cases)} test cases")

        # Default histogram clusters for 3 lesion types (like LeFusion)
        self.histogram_clusters = self._get_default_clusters()
        
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
    
    def _get_default_clusters(self):
        """Load histogram clusters from file or use defaults"""
        # Try to load pre-computed clusters (like LeFusion)
        cluster_file = Path(__file__).parent / "hist_clusters" / "lidc_clusters.json"

        if cluster_file.exists():
            print(f"Loading histogram clusters from {cluster_file}")
            with open(cluster_file, 'r') as f:
                data = json.load(f)
                return np.array(data[0]['centers'], dtype=np.float32)
        else:
            print("Warning: No histogram clusters found, using defaults")
            print("Run: python utils/extract_histogram_clusters.py to generate clusters")
            # Fallback to default clusters
            return np.array([
                # Type 0: Dark lesion
                np.array([0.4, 0.3, 0.2, 0.1] + [0]*12),
                # Type 1: Bright lesion
                np.array([0.1, 0.2, 0.3, 0.4] + [0]*12),
                # Type 2: Mixed lesion
                np.array([0.25]*4 + [0]*12)
            ], dtype=np.float32)

    def generate_synthetic_mask(self, image_shape: tuple = (256, 256)) -> torch.Tensor:
        """Generate synthetic lesion mask (like LeFusion/DiffMask)"""
        mask = torch.zeros(1, 1, *image_shape, device=self.device)

        # Generate 1-3 lesions
        num_lesions = np.random.randint(1, 4)
        for _ in range(num_lesions):
            # Random position (avoid edges)
            cx = np.random.randint(image_shape[1]//4, 3*image_shape[1]//4)
            cy = np.random.randint(image_shape[0]//4, 3*image_shape[0]//4)

            # Random elliptical shape
            rx = np.random.randint(10, 30)
            ry = np.random.randint(10, 30)

            # Generate ellipse
            y, x = np.ogrid[:image_shape[0], :image_shape[1]]
            ellipse = ((x - cx)/rx)**2 + ((y - cy)/ry)**2 <= 1
            mask[0, 0][ellipse] = 1

        return mask

    def get_histogram_for_type(self, lesion_type: int) -> torch.Tensor:
        """Get histogram condition for specified lesion type"""
        if lesion_type >= len(self.histogram_clusters):
            lesion_type = lesion_type % len(self.histogram_clusters)
        hist = self.histogram_clusters[lesion_type]
        return torch.from_numpy(hist).float().to(self.device)

    def generate_simple_mask(self) -> torch.Tensor:
        """Generate simple circular mask as fallback"""
        mask = torch.zeros(1, 1, 256, 256, device=self.device)
        cx, cy = 128, 128
        radius = np.random.randint(20, 40)
        y, x = np.ogrid[:256, :256]
        circle = (x - cx)**2 + (y - cy)**2 <= radius**2
        mask[0, 0][circle] = 1
        return mask

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
    def synthesize(self, normal_image: torch.Tensor, lesion_type: int = None,
                  use_repaint: bool = False, ddim_steps: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize pathological from normal image (LeFusion-style)"""
        # Generate synthetic mask
        mask = self.generate_synthetic_mask()

        # Get histogram for lesion type (0, 1, or 2)
        if lesion_type is None:
            lesion_type = np.random.randint(0, 3)
        histogram = self.get_histogram_for_type(lesion_type)
        
        # DDIM sampling with adaptive noise (matching training)
        timesteps = torch.linspace(999, 0, ddim_steps, dtype=torch.long, device=self.device)

        # Use adaptive noise scheduler if available
        if hasattr(self.scheduler, 'learnable_beta'):
            # Get adaptive betas (matching training)
            all_timesteps = torch.arange(1000, device=self.device)
            betas = self.scheduler(all_timesteps)
        else:
            # Fallback to standard schedule
            betas = self.scheduler.base_beta if hasattr(self.scheduler, 'base_beta') else \
                    torch.linspace(0.0001, 0.02, 1000, device=self.device)

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
            
            # Predict noise with histogram conditioning (matching training)
            # Model expects timesteps not t
            timestep_tensor = t.unsqueeze(0) if t.dim() == 0 else t
            noise_pred = self.model(x_combined, timestep_tensor, mask, histogram.unsqueeze(0))
            
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
        """Process normal images (test cases if specified, otherwise all)"""
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
        
        # Find images based on test cases or all
        if self.test_cases:
            # Process only test cases (like LeFusion)
            print(f"Processing {len(self.test_cases)} test cases from test.txt...")
            image_files = []
            for case_name in self.test_cases:
                # Try to find matching normal image
                img_path = normal_image_dir / f"{case_name}.nii.gz"
                if img_path.exists():
                    image_files.append(img_path)
                else:
                    print(f"  Warning: Test case {case_name} not found in normal images")
        else:
            # Process all normal images
            image_files = []
            for ext in ['*.nii.gz', '*.nii', '*.npy']:
                image_files.extend(normal_image_dir.glob(ext))
        
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
            
            # Generate synthetic with synthetic mask and lesion type
            lesion_type = idx % 3  # Cycle through 3 lesion types (like LeFusion)
            synthetic, used_mask = self.synthesize(normal, lesion_type, use_repaint=False, ddim_steps=ddim_steps)
            
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
    parser = argparse.ArgumentParser(description="SALAD Inference Pipeline (LeFusion-style)")

    # Config file support
    parser.add_argument("--config", help="Path to config file (YAML or JSON)")

    # Individual arguments (matching LeFusion)
    parser.add_argument("--checkpoint", help="Model checkpoint path")
    parser.add_argument("--test_txt_path", help="Path to test.txt file")
    parser.add_argument("--normal_dir", help="Normal images directory")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--device", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            # Make relative to SALAD directory
            config_path = Path(__file__).parent.parent / config_path

        if config_path.exists():
            print(f"Loading config from: {config_path}")
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

            # Set defaults from config
            if not args.checkpoint and 'model' in config:
                args.checkpoint = config['model'].get('checkpoint')
            if not args.normal_dir and 'data' in config:
                args.normal_dir = config['data'].get('normal_dir')
            if not args.pathological_dir and 'data' in config:
                args.pathological_dir = config['data'].get('pathological_dir')
            if not args.output_dir and 'data' in config:
                args.output_dir = config['data'].get('output_dir')
            if not args.ddim_steps and 'sampling' in config:
                args.ddim_steps = config['sampling'].get('ddim_steps', 50)
            if not args.num_samples and 'sampling' in config:
                args.num_samples = config['sampling'].get('num_samples', 100)
            if not args.device and 'model' in config:
                args.device = config['model'].get('device', 'cuda')
        else:
            print(f"Warning: Config file not found: {config_path}")

    # Set defaults if not provided (LeFusion-style)
    args.checkpoint = args.checkpoint or "checkpoints/lidc_fixed/checkpoint_latest.pth"
    args.test_txt_path = getattr(args, 'test_txt_path', None) or "../data/LIDC/Pathological/test.txt"
    args.normal_dir = args.normal_dir or "../data/LIDC/Normal/Image"
    args.output_dir = args.output_dir or "results/synthesis"
    args.ddim_steps = args.ddim_steps or 50
    args.device = args.device or "cuda"

    # Handle relative paths
    salad_dir = Path(__file__).parent.parent

    # Make paths absolute if relative
    if not Path(args.checkpoint).is_absolute():
        args.checkpoint = str((salad_dir / args.checkpoint).resolve())
    if args.test_txt_path and not Path(args.test_txt_path).is_absolute():
        args.test_txt_path = str((salad_dir / args.test_txt_path).resolve())
    if not Path(args.normal_dir).is_absolute():
        args.normal_dir = str((salad_dir / args.normal_dir).resolve())
    if not Path(args.output_dir).is_absolute():
        args.output_dir = str((salad_dir / args.output_dir).resolve())

    # Print configuration (LeFusion-style)
    print("=" * 60)
    print("SALAD Inference Configuration (LeFusion-style)")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test file: {args.test_txt_path}")
    print(f"Normal dir: {args.normal_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"DDIM steps: {args.ddim_steps}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Initialize with test cases (like LeFusion)
    inference = SALADInference(args.checkpoint, args.device, args.test_txt_path)
    inference.process_directory(args.normal_dir, args.output_dir, args.ddim_steps)


if __name__ == "__main__":
    main()