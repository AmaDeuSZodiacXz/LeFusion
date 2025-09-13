#!/usr/bin/env python3
"""
LIDC-specific inference script for SALAD
Handles checkpoint format from training
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import nibabel as nib
from PIL import Image

# Import SALAD components
from models.salad_core import NeuralSynthUNet, SALADConfig

class LIDCInference:
    def __init__(self, checkpoint_path, output_dir="results/lidc_synthesis", device="cuda"):
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        self.model = self.load_model()
        
    def load_model(self):
        """Load model with flexible checkpoint handling"""
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Print checkpoint structure for debugging
        print(f"Checkpoint keys: {checkpoint.keys()}")
        
        # Extract config if available
        if 'config' in checkpoint:
            config = SALADConfig(**checkpoint['config'])
        else:
            # Use default config for LIDC
            config = SALADConfig(
                image_size=256,
                in_channels=1,
                out_channels=1,
                model_channels=128,
                num_res_blocks=3,
                attention_resolutions=[16, 8],
                channel_mult=[1, 2, 4, 8],
                num_heads=8,
                dropout=0.1,
                use_adaptive_noise=True,
                use_multi_scale=True,
                use_lesion_attention=True,
                num_timesteps=1000,
                lesion_classes=5
            )
        
        # Create model
        model = NeuralSynthUNet(config)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Handle nested model state dict
        if 'model' in state_dict and isinstance(state_dict['model'], dict):
            state_dict = state_dict['model']
        
        # Remove prefixes if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'model.' prefix if present
            if k.startswith('model.'):
                new_key = k[6:]  # Remove 'model.'
            # Remove 'module.' prefix (from DataParallel)
            elif k.startswith('module.'):
                new_key = k[7:]  # Remove 'module.'
            else:
                new_key = k
            new_state_dict[new_key] = v
        
        # Load with strict=False to handle architecture mismatches
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
            print(f"First 5 missing: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            print(f"First 5 unexpected: {unexpected_keys[:5]}")
        
        model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded successfully with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        return model
    
    @torch.no_grad()
    def generate_samples(self, num_samples=100, batch_size=8, ddim_steps=50):
        """Generate synthetic samples using DDIM"""
        print(f"\nGenerating {num_samples} synthetic images...")
        print(f"Batch size: {batch_size}, DDIM steps: {ddim_steps}")
        
        all_samples = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            # Adjust batch size for last batch
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            # Generate noise
            shape = (current_batch_size, 1, 256, 256)
            x = torch.randn(shape, device=self.device)
            
            # DDIM sampling
            timesteps = torch.linspace(999, 0, ddim_steps, dtype=torch.long, device=self.device)
            
            for t in timesteps:
                t_batch = t.repeat(current_batch_size)
                
                # Predict noise (without lesion mask for unconditional generation)
                noise_pred = self.model(x, t_batch, lesion_mask=None)
                
                # DDIM update
                alpha = 1 - (t.float() / 1000) * 0.02
                x = (x - 0.02 * noise_pred) / (1 + 0.01 * alpha)
            
            # Normalize to [0, 1]
            x = (x + 1) / 2
            x = torch.clamp(x, 0, 1)
            
            all_samples.append(x.cpu())
        
        # Concatenate all samples
        all_samples = torch.cat(all_samples, dim=0)
        return all_samples
    
    def save_samples(self, samples, format="nifti"):
        """Save generated samples"""
        print(f"\nSaving {len(samples)} samples to {self.output_dir}")
        
        for i, sample in enumerate(tqdm(samples, desc="Saving")):
            if format == "nifti":
                # Save as NIfTI
                img_np = sample.squeeze().numpy()
                nifti_img = nib.Nifti1Image(img_np, np.eye(4))
                save_path = self.output_dir / f"synthetic_{i:04d}.nii.gz"
                nib.save(nifti_img, save_path)
            else:
                # Save as PNG
                img_np = (sample.squeeze().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                save_path = self.output_dir / f"synthetic_{i:04d}.png"
                img_pil.save(save_path)
        
        print(f"✓ All samples saved to {self.output_dir}")
    
    def run(self, num_samples=100, batch_size=8, ddim_steps=50, save_format="nifti"):
        """Run complete inference pipeline"""
        samples = self.generate_samples(num_samples, batch_size, ddim_steps)
        self.save_samples(samples, save_format)
        return samples

def main():
    parser = argparse.ArgumentParser(description="LIDC Inference for SALAD")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--output_dir", type=str, default="results/lidc_synthesis",
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for generation")
    parser.add_argument("--ddim_steps", type=int, default=50,
                       help="Number of DDIM steps")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--save_format", type=str, default="nifti",
                       choices=["nifti", "png"],
                       help="Output format")
    
    args = parser.parse_args()
    
    # Run inference
    inference = LIDCInference(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device
    )
    
    inference.run(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        ddim_steps=args.ddim_steps,
        save_format=args.save_format
    )
    
    print("\n" + "="*50)
    print("Inference completed successfully!")
    print(f"Generated {args.num_samples} synthetic images")
    print(f"Saved to: {args.output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()