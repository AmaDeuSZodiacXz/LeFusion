import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time
from einops import rearrange
import math


class DDIMSampler:
    def __init__(self, num_inference_steps: int = 50, eta: float = 0.0):
        self.num_inference_steps = num_inference_steps
        self.eta = eta
    
    def get_sampling_timesteps(self, num_train_timesteps: int) -> torch.Tensor:
        step_ratio = num_train_timesteps // self.num_inference_steps
        timesteps = torch.arange(0, num_train_timesteps, step_ratio).flip(0)
        return timesteps
    
    def sample_step(self, model_output: torch.Tensor, timestep: int, 
                   sample: torch.Tensor, alphas_cumprod: torch.Tensor,
                   prev_timestep: Optional[int] = None) -> torch.Tensor:
        
        alpha_prod_t = alphas_cumprod[timestep]
        
        if prev_timestep is None:
            prev_timestep = max(0, timestep - self.num_inference_steps // len(alphas_cumprod))
        
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = self.eta * variance ** 0.5
        
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output
        
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        if self.eta > 0 and timestep > 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + std_dev_t * noise
        
        return prev_sample


class OptimizedNeuralSynth(nn.Module):
    def __init__(self, base_model, config):
        super().__init__()
        self.model = base_model
        self.config = config
        
        self.ddim_sampler = DDIMSampler(num_inference_steps=50, eta=0.0)
        
        self.use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        
        if self.use_fp16:
            self.model = self.model.half()
        
        self.compiled_model = None
        if hasattr(torch, 'compile'):
            try:
                self.compiled_model = torch.compile(self.model, mode='reduce-overhead')
            except:
                self.compiled_model = None
    
    @torch.no_grad()
    def fast_sample(self, shape: Tuple[int, ...], 
                   lesion_mask: Optional[torch.Tensor] = None,
                   num_steps: int = 50,
                   guidance_scale: float = 1.0,
                   device: str = 'cuda') -> torch.Tensor:
        
        model = self.compiled_model if self.compiled_model is not None else self.model
        model.eval()
        
        x = torch.randn(shape, device=device)
        if self.use_fp16:
            x = x.half()
            if lesion_mask is not None:
                lesion_mask = lesion_mask.half()
        
        timesteps = self.ddim_sampler.get_sampling_timesteps(self.config.num_timesteps)
        
        alphas = 1 - self._get_beta_schedule(self.config.num_timesteps)
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            if guidance_scale > 1.0 and lesion_mask is not None:
                noise_pred_uncond = model.model(x, t_batch, None)
                noise_pred_cond = model.model(x, t_batch, lesion_mask)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = model.model(x, t_batch, lesion_mask)
            
            prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else -1
            
            x = self.ddim_sampler.sample_step(
                noise_pred, t, x, alphas_cumprod, prev_t
            )
        
        if self.use_fp16:
            x = x.float()
        
        return x
    
    def _get_beta_schedule(self, timesteps: int) -> torch.Tensor:
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    @torch.no_grad()
    def progressive_sample(self, shape: Tuple[int, ...],
                          lesion_mask: Optional[torch.Tensor] = None,
                          start_steps: int = 10,
                          refine_steps: int = 20,
                          device: str = 'cuda') -> torch.Tensor:
        
        coarse = self.fast_sample(shape, lesion_mask, num_steps=start_steps, device=device)
        
        noise_level = 0.3
        noisy = coarse + noise_level * torch.randn_like(coarse)
        
        refined = self.fast_sample(shape, lesion_mask, num_steps=refine_steps, device=device)
        
        return 0.7 * coarse + 0.3 * refined
    
    @torch.no_grad()
    def batch_sample(self, batch_size: int, image_size: int,
                    lesion_masks: Optional[torch.Tensor] = None,
                    device: str = 'cuda') -> torch.Tensor:
        
        shape = (batch_size, 1, image_size, image_size)
        
        return self.fast_sample(shape, lesion_masks, num_steps=50, device=device)


class CachedNeuralSynth(nn.Module):
    def __init__(self, base_model, config, cache_size: int = 100):
        super().__init__()
        self.optimized_model = OptimizedNeuralSynth(base_model, config)
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_cache_key(self, lesion_mask: torch.Tensor) -> str:
        mask_hash = hash(lesion_mask.cpu().numpy().tobytes())
        return str(mask_hash)
    
    @torch.no_grad()
    def sample_with_cache(self, shape: Tuple[int, ...],
                         lesion_mask: Optional[torch.Tensor] = None,
                         device: str = 'cuda',
                         use_cache: bool = True) -> torch.Tensor:
        
        if use_cache and lesion_mask is not None:
            cache_key = self._get_cache_key(lesion_mask)
            
            if cache_key in self.cache:
                self.cache_hits += 1
                cached_result = self.cache[cache_key]
                
                noise = 0.05 * torch.randn_like(cached_result)
                return cached_result + noise
        
        self.cache_misses += 1
        
        result = self.optimized_model.fast_sample(shape, lesion_mask, device=device)
        
        if use_cache and lesion_mask is not None:
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result.clone()
        
        return result
    
    def get_cache_stats(self) -> Dict[str, float]:
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


class ParallelInference:
    def __init__(self, model, num_workers: int = 4):
        self.model = model
        self.num_workers = num_workers
    
    def parallel_generate(self, masks: List[torch.Tensor], 
                         batch_size: int = 4) -> List[torch.Tensor]:
        
        results = []
        
        for i in range(0, len(masks), batch_size):
            batch_masks = masks[i:i+batch_size]
            
            if len(batch_masks) < batch_size:
                padding = batch_size - len(batch_masks)
                batch_masks.extend([batch_masks[-1]] * padding)
            
            batch_tensor = torch.stack(batch_masks)
            
            outputs = self.model.batch_sample(
                batch_size=batch_size,
                image_size=batch_masks[0].shape[-1],
                lesion_masks=batch_tensor
            )
            
            results.extend(outputs[:len(masks[i:i+batch_size])])
        
        return results


class InferenceBenchmark:
    def __init__(self, model):
        self.model = model
        self.results = []
    
    def benchmark_inference(self, num_samples: int = 100, 
                          image_size: int = 256,
                          device: str = 'cuda') -> Dict[str, float]:
        
        torch.cuda.empty_cache()
        
        warmup_samples = 5
        for _ in range(warmup_samples):
            _ = self.model.fast_sample(
                shape=(1, 1, image_size, image_size),
                device=device
            )
        
        torch.cuda.synchronize()
        
        inference_times = []
        memory_usage = []
        
        for _ in range(num_samples):
            torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            _ = self.model.fast_sample(
                shape=(1, 1, image_size, image_size),
                device=device
            )
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            memory_usage.append(peak_memory)
        
        results = {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'median_inference_time': np.median(inference_times),
            'p95_inference_time': np.percentile(inference_times, 95),
            'throughput_fps': 1.0 / np.mean(inference_times),
            'mean_memory_gb': np.mean(memory_usage),
            'peak_memory_gb': np.max(memory_usage)
        }
        
        return results
    
    def compare_sampling_methods(self, image_size: int = 256, 
                                device: str = 'cuda') -> Dict[str, Dict]:
        
        comparison = {}
        
        methods = [
            ('DDIM-50', lambda m: m.fast_sample(
                shape=(1, 1, image_size, image_size), 
                num_steps=50, device=device
            )),
            ('DDIM-25', lambda m: m.fast_sample(
                shape=(1, 1, image_size, image_size), 
                num_steps=25, device=device
            )),
            ('Progressive', lambda m: m.progressive_sample(
                shape=(1, 1, image_size, image_size),
                start_steps=10, refine_steps=20, device=device
            ))
        ]
        
        for method_name, method_func in methods:
            times = []
            
            for _ in range(20):
                start = time.time()
                _ = method_func(self.model)
                torch.cuda.synchronize()
                times.append(time.time() - start)
            
            comparison[method_name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'throughput_fps': 1.0 / np.mean(times)
            }
        
        return comparison


def create_optimized_model(checkpoint_path: str, config_path: Optional[str] = None):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.neuralsynth_core import NeuralSynthConfig, NeuralSynthDiffusion
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if config_path:
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = NeuralSynthConfig(**config_dict)
    else:
        config_dict = checkpoint.get('config', {})
        config = NeuralSynthConfig(**config_dict)
    
    base_model = NeuralSynthDiffusion(config)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    optimized = OptimizedNeuralSynth(base_model, config)
    
    cached_model = CachedNeuralSynth(base_model, config)
    
    return optimized, cached_model