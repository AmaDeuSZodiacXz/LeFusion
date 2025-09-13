#!/usr/bin/env python3
"""
Quick SALAD Inference - Minimal script for fast synthetic generation
"""

import torch
import numpy as np
from pathlib import Path
from models.salad_core import NeuralSynthUNet, SALADConfig

def generate_synthetic(checkpoint_path, num_samples=10, device="cuda"):
    """Quick generation function"""
    
    # Load model
    print(f"Loading SALAD model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with default config
    config = SALADConfig()
    model = NeuralSynthUNet(config).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Quick DDIM sampling (50 steps)
    print(f"Generating {num_samples} synthetic images...")
    
    with torch.no_grad():
        # Initialize from noise
        x = torch.randn(num_samples, 1, 256, 256).to(device)
        
        # Simple 50-step DDIM
        timesteps = torch.linspace(999, 0, 50, dtype=torch.long)
        
        for t in timesteps:
            t_batch = t.repeat(num_samples).to(device)
            
            # Predict noise
            noise_pred = model(x, t_batch)
            
            # DDIM update (simplified)
            alpha = 1 - (t / 1000) * 0.02  # Approximate schedule
            x = (x - noise_pred * 0.02) / (1 + 0.01)
        
        # Normalize to [0, 1]
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
    
    print(f"✓ Generated {num_samples} images")
    return x.cpu().numpy()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quick_inference.py <checkpoint_path> [num_samples]")
        sys.exit(1)
    
    checkpoint = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Generate
    images = generate_synthetic(checkpoint, num_samples)
    
    # Save results
    output_dir = Path("quick_results")
    output_dir.mkdir(exist_ok=True)
    
    for i, img in enumerate(images):
        np.save(output_dir / f"synthetic_{i:03d}.npy", img)
    
    print(f"Results saved to {output_dir}/")