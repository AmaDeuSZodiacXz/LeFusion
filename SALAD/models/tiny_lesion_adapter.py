"""
Tiny Lesion Adapter for SALAD (Spatially-Aware Lesion Attention Diffusion)
Specialized module for detecting and synthesizing lesions as small as 1mm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class TinyLesionDetector(nn.Module):
    """
    Specialized detector for very small lesions (1-3mm)
    Uses high-resolution processing and special attention mechanisms
    """
    
    def __init__(self, in_channels: int = 1, min_lesion_size_mm: float = 1.0):
        super().__init__()
        self.min_lesion_size = min_lesion_size_mm
        
        # High-resolution feature extraction (no downsampling initially)
        self.high_res_conv = nn.Sequential(
            # Preserve full resolution
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, dilation=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            
            # Dilated convolutions to capture tiny features without downsampling
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
        )
        
        # Attention specifically for tiny regions
        self.tiny_attention = TinyRegionAttention(128)
        
        # Detection head
        self.detector = nn.Conv2d(128, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor, pixel_spacing: float = 1.0) -> torch.Tensor:
        """
        Args:
            x: Input image [B, C, H, W]
            pixel_spacing: mm per pixel (for calibration)
        
        Returns:
            Tiny lesion attention map [B, 1, H, W]
        """
        # Calculate minimum pixel size for lesion
        min_pixels = self.min_lesion_size / pixel_spacing
        
        # High-resolution processing
        features = self.high_res_conv(x)
        
        # Apply specialized attention
        features = self.tiny_attention(features, min_pixels)
        
        # Generate attention map
        attention_map = torch.sigmoid(self.detector(features))
        
        return attention_map


class TinyRegionAttention(nn.Module):
    """
    Attention mechanism optimized for regions as small as 1-3 pixels
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Local attention with very small receptive field
        self.local_attention = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        
        # Point-wise attention for single pixels
        self.point_attention = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Combine local and point attention
        self.combine = nn.Conv2d(channels * 2, channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, min_pixels: float = 1.0) -> torch.Tensor:
        # Local attention (3x3 neighborhood)
        local_attn = self.local_attention(x)
        
        # Point attention (1x1)
        point_attn = self.point_attention(x)
        
        # Adaptively combine based on expected lesion size
        if min_pixels <= 1.5:
            # Emphasize point attention for very tiny lesions
            combined = torch.cat([point_attn * 1.5, local_attn * 0.5], dim=1)
        else:
            # Balance both for slightly larger lesions
            combined = torch.cat([point_attn, local_attn], dim=1)
        
        return self.combine(combined)


class HighResolutionSALAD(nn.Module):
    """
    Enhanced SALAD for handling lesions down to 1mm
    Key changes:
    1. Higher base resolution (512x512 or 1024x1024)
    2. Specialized tiny lesion detection
    3. Adaptive multi-scale with finer scales
    """
    
    def __init__(self, 
                 base_resolution: int = 512,  # Higher than 256
                 min_lesion_mm: float = 1.0,
                 pixel_spacing_mm: float = 0.5):  # 0.5mm per pixel
        super().__init__()
        
        self.base_resolution = base_resolution
        self.pixel_spacing = pixel_spacing_mm
        
        # Tiny lesion detector
        self.tiny_detector = TinyLesionDetector(min_lesion_size_mm=min_lesion_mm)
        
        # Adaptive scales based on minimum lesion size
        self.scales = self._compute_adaptive_scales(min_lesion_mm, pixel_spacing_mm)
        
        # Multi-scale extractors with finer granularity
        self.multi_scale_extractors = nn.ModuleList([
            self._create_scale_extractor(scale) for scale in self.scales
        ])
        
    def _compute_adaptive_scales(self, min_lesion_mm: float, pixel_spacing: float) -> List[float]:
        """
        Compute optimal scales for capturing lesions of different sizes
        """
        min_pixels = min_lesion_mm / pixel_spacing
        
        if min_pixels <= 2:
            # Very fine scales for tiny lesions
            return [1.0, 0.9, 0.75, 0.5, 0.25]
        elif min_pixels <= 5:
            # Standard + fine scales
            return [1.0, 0.75, 0.5, 0.25]
        else:
            # Standard scales
            return [1.0, 0.5, 0.25]
    
    def _create_scale_extractor(self, scale: float) -> nn.Module:
        """
        Create extractor for specific scale
        """
        if scale >= 0.75:
            # High-resolution extractor (minimal downsampling)
            return nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU()
            )
        else:
            # Standard extractor for larger scales
            return nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU()
            )
    
    def forward(self, x: torch.Tensor, lesion_mask: Optional[torch.Tensor] = None):
        """
        Process with focus on tiny lesions
        """
        # Detect tiny lesions
        tiny_attention = self.tiny_detector(x, self.pixel_spacing)
        
        # Multi-scale processing with tiny lesion awareness
        multi_scale_features = []
        for scale, extractor in zip(self.scales, self.multi_scale_extractors):
            if scale == 1.0:
                # Full resolution with tiny lesion attention
                feat = extractor(x * (1 + tiny_attention))
            else:
                # Other scales
                scaled_x = F.interpolate(x, scale_factor=scale, mode='bilinear')
                feat = extractor(scaled_x)
                feat = F.interpolate(feat, size=x.shape[-2:], mode='bilinear')
            
            multi_scale_features.append(feat)
        
        # Combine all scales
        combined = torch.cat(multi_scale_features, dim=1)
        
        return combined, tiny_attention


class SubPixelSynthesis(nn.Module):
    """
    Sub-pixel synthesis for lesions smaller than 1 pixel
    Uses super-resolution techniques
    """
    
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.upscale = upscale_factor
        
        # Sub-pixel convolution for super-resolution
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv3 = nn.Conv2d(32 // (upscale_factor ** 2), 1, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate higher resolution output for tiny lesions
        """
        # Feature extraction
        feat = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(feat))
        
        # Sub-pixel upsampling
        high_res = self.pixel_shuffle(feat)
        
        # Final synthesis
        output = self.conv3(high_res)
        
        return output


def create_tiny_lesion_model(min_lesion_mm: float = 1.0,
                             image_size_mm: float = 256.0,
                             target_resolution: int = 512):
    """
    Create a model configuration optimized for tiny lesions
    
    Args:
        min_lesion_mm: Minimum lesion size in mm
        image_size_mm: Physical size of image in mm
        target_resolution: Target image resolution in pixels
    
    Returns:
        Configured model for tiny lesion synthesis
    """
    
    # Calculate required pixel spacing
    pixel_spacing = image_size_mm / target_resolution
    
    print(f"Configuration for {min_lesion_mm}mm lesions:")
    print(f"  Target resolution: {target_resolution}x{target_resolution}")
    print(f"  Pixel spacing: {pixel_spacing:.3f}mm/pixel")
    print(f"  Min lesion size: {min_lesion_mm/pixel_spacing:.1f} pixels")
    
    if min_lesion_mm / pixel_spacing < 3:
        print("  ⚠️ Warning: Lesions will be <3 pixels. Using enhanced tiny lesion mode.")
        return HighResolutionSALAD(
            base_resolution=target_resolution,
            min_lesion_mm=min_lesion_mm,
            pixel_spacing_mm=pixel_spacing
        )
    else:
        print("  ✓ Standard multi-scale processing sufficient")
        return None  # Use standard SALAD


# Example usage for 1mm lesion detection
if __name__ == "__main__":
    # For 1mm lesions in 256mm field of view
    model = create_tiny_lesion_model(
        min_lesion_mm=1.0,
        image_size_mm=256.0,
        target_resolution=512  # Need higher resolution!
    )
    
    if model:
        # Test with sample input
        x = torch.randn(1, 1, 512, 512)
        features, tiny_attention = model(x)
        print(f"\nOutput shapes:")
        print(f"  Features: {features.shape}")
        print(f"  Tiny attention: {tiny_attention.shape}")