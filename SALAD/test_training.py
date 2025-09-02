#!/usr/bin/env python3
"""Test script to verify NaN issues are fixed."""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.neuralsynth_core import NeuralSynthDiffusion, NeuralSynthConfig
from models.advanced_losses import DiffusionLoss

def test_model():
    """Test model with various inputs to check for NaN."""
    
    config = NeuralSynthConfig(
        image_size=64,  # Smaller for quick testing
        in_channels=1,
        out_channels=1,
        model_channels=64,  # Smaller model
        use_adaptive_noise=True,
        use_lesion_attention=True,
        use_multi_scale=True
    )
    
    model = NeuralSynthDiffusion(config)
    criterion = DiffusionLoss(loss_type='l2', use_weighted=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Lower learning rate
    
    print("Testing model stability...")
    
    # Test with various inputs
    for i in range(10):
        # Create test batch
        x = torch.randn(2, 1, 64, 64)
        mask = torch.ones(2, 1, 64, 64) * (i % 2)  # Alternate between full and no mask
        bg = torch.randn(2, 1, 64, 64)
        
        # Forward pass
        output = model(x, lesion_mask=mask, background=bg)
        
        # Compute loss
        loss = criterion(output['predicted_noise'], output['target_noise'], output['timesteps'])
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"❌ NaN detected at iteration {i}")
            print(f"  Predicted noise: min={output['predicted_noise'].min():.3f}, max={output['predicted_noise'].max():.3f}")
            print(f"  Target noise: min={output['target_noise'].min():.3f}, max={output['target_noise'].max():.3f}")
            return False
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
                if torch.isnan(param_norm):
                    print(f"❌ NaN gradient detected at iteration {i}")
                    return False
        
        total_grad_norm = total_grad_norm ** 0.5
        
        # Clip and step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        print(f"✓ Iteration {i}: loss={loss.item():.4f}, grad_norm={total_grad_norm:.4f}")
    
    print("\n✅ All tests passed! Model is stable.")
    return True

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)