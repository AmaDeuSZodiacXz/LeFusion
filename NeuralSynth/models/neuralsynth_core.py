import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
from einops import rearrange, repeat
import math


@dataclass
class NeuralSynthConfig:
    image_size: int = 256
    in_channels: int = 1
    out_channels: int = 1
    model_channels: int = 128
    num_res_blocks: int = 3
    attention_resolutions: List[int] = None
    dropout: float = 0.1
    channel_mult: List[int] = None
    num_heads: int = 8
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    use_new_attention_order: bool = True
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    lesion_classes: int = 5
    use_adaptive_noise: bool = True
    use_multi_scale: bool = True
    use_lesion_attention: bool = True
    
    def __post_init__(self):
        if self.attention_resolutions is None:
            self.attention_resolutions = [16, 8]
        if self.channel_mult is None:
            self.channel_mult = [1, 2, 4, 8]


class AdaptiveNoiseScheduler(nn.Module):
    def __init__(self, num_timesteps: int = 1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.learnable_beta = nn.Parameter(torch.zeros(num_timesteps))
        self.base_beta = self._cosine_beta_schedule(num_timesteps)
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        adaptive_factor = torch.sigmoid(self.learnable_beta)
        betas = self.base_beta * (1 + 0.1 * adaptive_factor)
        return betas[t]


class LesionAwareAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, num_classes: int = 5):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.lesion_embed = nn.Embedding(num_classes + 1, dim)
        self.lesion_proj = nn.Linear(dim, dim)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor, lesion_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        if lesion_mask is not None:
            lesion_emb = self.lesion_embed(lesion_mask.long())
            lesion_emb = self.lesion_proj(lesion_emb)
            lesion_emb = rearrange(lesion_emb, 'b n d -> b 1 n d')
            v = v + lesion_emb
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scales: List[int] = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        
        # Make channels_per_scale divisible by common group sizes
        # Round to nearest multiple of 8 for better compatibility
        channels_per_scale = (out_channels // len(scales) // 8) * 8
        if channels_per_scale == 0:
            channels_per_scale = 8
        
        # Calculate appropriate number of groups
        # Find the largest divisor of channels_per_scale that's <= 8
        possible_groups = [g for g in [8, 4, 2, 1] if channels_per_scale % g == 0]
        num_groups = possible_groups[0] if possible_groups else 1
        
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, channels_per_scale, 3, padding=1),
                nn.GroupNorm(num_groups, channels_per_scale),
                nn.SiLU(),
                nn.Conv2d(channels_per_scale, channels_per_scale, 3, padding=1),
                nn.GroupNorm(num_groups, channels_per_scale),
                nn.SiLU()
            ) for _ in scales
        ])
        
        # Fusion layer to combine and adjust to exact output channels
        self.fusion = nn.Conv2d(channels_per_scale * len(scales), out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for scale, extractor in zip(self.scales, self.extractors):
            if scale != 1:
                scaled_x = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=False)
                feat = extractor(scaled_x)
                feat = F.interpolate(feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
            else:
                feat = extractor(x)
            features.append(feat)
        
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 dropout: float = 0.1, use_scale_shift_norm: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2 if use_scale_shift_norm else out_channels)
        )
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
            
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        time_emb = self.time_emb_proj(time_emb)[:, :, None, None]
        
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(time_emb, 2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
        else:
            h = self.norm2(h + time_emb)
        
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_connection(x)


class NeuralSynthUNet(nn.Module):
    def __init__(self, config: NeuralSynthConfig):
        super().__init__()
        self.config = config
        
        time_emb_dim = config.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(config.model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.lesion_embed = nn.Embedding(config.lesion_classes + 1, time_emb_dim)
        
        if config.use_multi_scale:
            self.input_conv = MultiScaleFeatureExtractor(
                config.in_channels, config.model_channels
            )
        else:
            self.input_conv = nn.Conv2d(config.in_channels, config.model_channels, 3, padding=1)
        
        channels = [config.model_channels * m for m in config.channel_mult]
        
        self.down_blocks = nn.ModuleList()
        ch = config.model_channels
        for level, mult in enumerate(config.channel_mult):
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(ch, channels[level], time_emb_dim, 
                                config.dropout, config.use_scale_shift_norm)
                )
                ch = channels[level]
            
            if level != len(config.channel_mult) - 1:
                self.down_blocks.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                )
        
        self.middle_block = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, config.dropout, config.use_scale_shift_norm),
            LesionAwareAttention(ch, config.num_heads, config.lesion_classes) if config.use_lesion_attention else nn.Identity(),
            ResidualBlock(ch, ch, time_emb_dim, config.dropout, config.use_scale_shift_norm)
        ])
        
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(config.channel_mult))):
            for i in range(config.num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(
                        ch + channels[level] if i == 0 else channels[level],
                        channels[level], time_emb_dim, 
                        config.dropout, config.use_scale_shift_norm
                    )
                )
                if i == config.num_res_blocks and level != 0:
                    self.up_blocks.append(
                        nn.ConvTranspose2d(channels[level], channels[level-1], 4, stride=2, padding=1)
                    )
                    ch = channels[level-1]
                else:
                    ch = channels[level]
        
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, config.model_channels),
            nn.SiLU(),
            nn.Conv2d(config.model_channels, config.out_channels, 3, padding=1)
        )
        
    def timestep_embedding(self, timesteps: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, 
                lesion_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        time_emb = self.timestep_embedding(timesteps, self.config.model_channels)
        time_emb = self.time_embed(time_emb)
        
        if lesion_mask is not None:
            lesion_emb = self.lesion_embed(lesion_mask.long().mean(dim=[2, 3]))
            time_emb = time_emb + lesion_emb
        
        h = self.input_conv(x)
        
        skips = []
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
                skips.append(h)
            else:
                h = block(h)
        
        for block in self.middle_block:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
            elif isinstance(block, LesionAwareAttention):
                b, c, height, width = h.shape
                h_flat = rearrange(h, 'b c h w -> b (h w) c')
                if lesion_mask is not None:
                    mask_flat = rearrange(lesion_mask, 'b c h w -> b (h w) c').squeeze(-1)
                else:
                    mask_flat = None
                h_flat = block(h_flat, mask_flat)
                h = rearrange(h_flat, 'b (h w) c -> b c h w', h=height, w=width)
            else:
                h = block(h)
        
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                if skips:
                    h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, time_emb)
            else:
                h = block(h)
        
        return self.output_conv(h)


class NeuralSynthDiffusion(nn.Module):
    """NeuralSynth Diffusion with Background Preservation.
    
    Key innovation beyond LeFusion:
    - Preserves background using forward diffusion (like LeFusion)
    - Adds adaptive boundary blending for smoother transitions
    - Implements multi-resolution background preservation
    """
    
    def __init__(self, config: NeuralSynthConfig):
        super().__init__()
        self.config = config
        self.model = NeuralSynthUNet(config)
        self.preserve_background = True  # Core feature from LeFusion
        
        if config.use_adaptive_noise:
            self.noise_scheduler = AdaptiveNoiseScheduler(config.num_timesteps)
        else:
            self.register_buffer('betas', self._get_beta_schedule(config.num_timesteps))
            self.register_buffer('alphas', 1 - self.betas)
            self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # Boundary smoothing for better lesion-background transition
        self.boundary_smoother = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            # Gaussian kernel for smoothing
            kernel = torch.tensor([[1, 4, 6, 4, 1],
                                 [4, 16, 24, 16, 4],
                                 [6, 24, 36, 24, 6],
                                 [4, 16, 24, 16, 4],
                                 [1, 4, 6, 4, 1]], dtype=torch.float32)
            kernel = kernel / kernel.sum()
            self.boundary_smoother.weight = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0))
            
    def _get_beta_schedule(self, timesteps: int) -> torch.Tensor:
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def forward_diffusion(self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x)
        
        if self.config.use_adaptive_noise:
            betas_t = self.noise_scheduler(t)
            alphas_t = 1 - betas_t
            alphas_cumprod_t = torch.cumprod(alphas_t, dim=0)
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t)
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod_t)
        else:
            sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def forward(self, x: torch.Tensor, lesion_mask: Optional[torch.Tensor] = None,
                background: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with background preservation.
        
        Following LeFusion's approach:
        - x: Full image with lesion
        - background: Clean background without lesion (for preservation)
        - lesion_mask: Binary mask indicating lesion region
        """
        batch_size = x.shape[0]
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=x.device).long()
        
        # Extract lesion region for focused training
        if self.preserve_background and lesion_mask is not None and background is not None:
            # Get lesion-only region
            lesion_only = x * lesion_mask
            
            # Forward diffuse both lesion and background
            lesion_noisy, lesion_noise = self.forward_diffusion(lesion_only, t)
            background_noisy, _ = self.forward_diffusion(background, t)
            
            # Combine: noisy lesion in foreground, noisy background elsewhere
            # This is the key LeFusion approach
            x_combined = lesion_noisy * lesion_mask + background_noisy * (1 - lesion_mask)
            
            # Smooth boundaries for better transition
            mask_smooth = self.boundary_smoother(lesion_mask)
            x_combined = x_combined * mask_smooth + background_noisy * (1 - mask_smooth)
            
            # Model predicts only lesion noise (focused learning)
            predicted_noise = self.model(x_combined, t, lesion_mask)
            
            # Loss computed only on lesion region (like LeFusion)
            target_noise = lesion_noise * lesion_mask
        else:
            # Standard diffusion without background preservation
            x_noisy, noise = self.forward_diffusion(x, t)
            predicted_noise = self.model(x_noisy, t, lesion_mask)
            target_noise = noise
        
        return {
            'predicted_noise': predicted_noise,
            'target_noise': target_noise,
            'timesteps': t,
            'lesion_mask': lesion_mask
        }
    
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], lesion_mask: Optional[torch.Tensor] = None,
               background: Optional[torch.Tensor] = None, device: str = 'cuda') -> torch.Tensor:
        """Sampling with background preservation (LeFusion-style).
        
        Key innovation: Preserves background by combining:
        - Reverse diffusion for lesion (foreground)
        - Forward diffusion for background
        """
        
        if self.preserve_background and background is not None and lesion_mask is not None:
            # Start with noise for lesion, actual background elsewhere
            x = torch.randn(shape, device=device) * lesion_mask
            
            for t in reversed(range(self.config.num_timesteps)):
                t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
                
                # Forward diffuse background to current timestep
                if t > 0:
                    noise_bg = torch.randn_like(background)
                    if self.config.use_adaptive_noise:
                        betas_t = self.noise_scheduler(t_batch)
                        alphas_cumprod_t = torch.cumprod(1 - betas_t, dim=0)[0]
                    else:
                        alphas_cumprod_t = self.alphas_cumprod[t]
                    
                    sqrt_alpha = torch.sqrt(alphas_cumprod_t)
                    sqrt_one_minus_alpha = torch.sqrt(1 - alphas_cumprod_t)
                    background_noised = sqrt_alpha * background + sqrt_one_minus_alpha * noise_bg
                else:
                    background_noised = background
                
                # Combine noisy lesion with noisy background
                mask_smooth = self.boundary_smoother(lesion_mask)
                x_combined = x * mask_smooth + background_noised * (1 - mask_smooth)
                
                # Predict noise for lesion region
                predicted_noise = self.model(x_combined, t_batch, lesion_mask)
                
                # Update only lesion region
                if self.config.use_adaptive_noise:
                    betas_t = self.noise_scheduler(t_batch)
                    alphas_t = 1 - betas_t
                    alphas_cumprod_t = torch.cumprod(alphas_t, dim=0)
                    alpha_t = alphas_t[0]
                    alpha_cumprod_t = alphas_cumprod_t[0]
                    beta_t = betas_t[0]
                else:
                    alpha_t = self.alphas[t]
                    alpha_cumprod_t = self.alphas_cumprod[t]
                    beta_t = self.betas[t]
                
                # Denoise lesion region
                mean = (x - beta_t * predicted_noise / torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_t)
                
                if t > 0:
                    noise = torch.randn_like(x)
                    var = beta_t
                    x = mean * lesion_mask + torch.sqrt(var) * noise * lesion_mask
                else:
                    x = mean * lesion_mask
            
            # Final combination: generated lesion + original background
            result = x + background * (1 - lesion_mask)
            return result
        else:
            # Standard sampling without background preservation
            x = torch.randn(shape, device=device)
            
            for t in reversed(range(self.config.num_timesteps)):
                t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
                predicted_noise = self.model(x, t_batch, lesion_mask)
                
                if self.config.use_adaptive_noise:
                    betas_t = self.noise_scheduler(t_batch)
                    alphas_t = 1 - betas_t
                    alphas_cumprod_t = torch.cumprod(alphas_t, dim=0)
                    alpha_t = alphas_t[0]
                    alpha_cumprod_t = alphas_cumprod_t[0]
                    beta_t = betas_t[0]
                else:
                    alpha_t = self.alphas[t]
                    alpha_cumprod_t = self.alphas_cumprod[t]
                    beta_t = self.betas[t]
                
                mean = (x - beta_t * predicted_noise / torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_t)
                
                if t > 0:
                    noise = torch.randn_like(x)
                    var = beta_t
                    x = mean + torch.sqrt(var) * noise
                else:
                    x = mean
            
            return x