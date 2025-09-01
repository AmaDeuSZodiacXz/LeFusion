import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from torchvision import models
import lpips


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers: list = None):
        super().__init__()
        if feature_layers is None:
            feature_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        
        vgg = models.vgg16(pretrained=True).features.eval()
        self.slices = nn.ModuleList()
        
        layer_map = {
            'relu1_2': 4,
            'relu2_2': 9,
            'relu3_3': 16,
            'relu4_3': 23,
            'relu5_3': 30
        }
        
        prev_layer = 0
        for layer_name in feature_layers:
            if layer_name in layer_map:
                self.slices.append(vgg[prev_layer:layer_map[layer_name]+1])
                prev_layer = layer_map[layer_name]+1
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        loss = 0
        x_pred = pred
        x_target = target
        
        for slice_layer in self.slices:
            x_pred = slice_layer(x_pred)
            x_target = slice_layer(x_target)
            loss += F.l1_loss(x_pred, x_target)
        
        return loss / len(self.slices)


class StructuralSimilarityLoss(nn.Module):
    def __init__(self, window_size: int = 11, channel: int = 1):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                                 for x in range(window_size)])
            return gauss/gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channel = pred.size(1)
        
        if channel == self.channel and self.window.data.type() == pred.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.type_as(pred)
            self.window = window
            self.channel = channel
        
        mu1 = F.conv2d(pred, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred*pred, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target*target, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred*target, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class FrequencyDomainLoss(nn.Module):
    def __init__(self, weight_low: float = 1.0, weight_high: float = 0.5):
        super().__init__()
        self.weight_low = weight_low
        self.weight_high = weight_high
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        h, w = pred.shape[-2:]
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        low_freq_mask = torch.zeros_like(pred_mag)
        high_freq_mask = torch.ones_like(pred_mag)
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        y = y.to(pred.device)
        x = x.to(pred.device)
        
        dist = torch.sqrt((x - center_w)**2 + (y - center_h)**2)
        low_freq_mask[dist < radius] = 1
        high_freq_mask[dist < radius] = 0
        
        low_freq_loss = F.l1_loss(pred_mag * low_freq_mask, target_mag * low_freq_mask)
        high_freq_loss = F.l1_loss(pred_mag * high_freq_mask, target_mag * high_freq_mask)
        
        return self.weight_low * low_freq_loss + self.weight_high * high_freq_loss


class EdgePreservingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)
        
        return F.l1_loss(pred_edges, target_edges)


class LesionConsistencyLoss(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                lesion_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if lesion_mask is None:
            return torch.tensor(0.0, device=pred.device)
        
        loss = 0
        for class_idx in range(1, self.num_classes + 1):
            mask = (lesion_mask == class_idx).float()
            if mask.sum() > 0:
                pred_lesion = pred * mask
                target_lesion = target * mask
                
                pred_mean = pred_lesion.sum() / mask.sum()
                target_mean = target_lesion.sum() / mask.sum()
                
                pred_std = torch.sqrt(((pred_lesion - pred_mean * mask) ** 2).sum() / mask.sum())
                target_std = torch.sqrt(((target_lesion - target_mean * mask) ** 2).sum() / mask.sum())
                
                loss += F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)
        
        return loss / self.num_classes


class AdversarialLoss(nn.Module):
    def __init__(self, gan_mode: str = 'lsgan'):
        super().__init__()
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgan':
            self.loss = lambda x, y: -torch.mean(x) if y else torch.mean(x)
        else:
            raise ValueError(f"Unknown GAN mode: {gan_mode}")
    
    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        if self.gan_mode in ['lsgan', 'vanilla']:
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
            return self.loss(pred, target)
        else:
            return self.loss(pred, is_real)


class NeuralSynthLoss(nn.Module):
    def __init__(self, 
                 lambda_l1: float = 1.0,
                 lambda_perceptual: float = 0.1,
                 lambda_ssim: float = 0.5,
                 lambda_frequency: float = 0.1,
                 lambda_edge: float = 0.2,
                 lambda_lesion: float = 0.3,
                 lambda_adversarial: float = 0.1):
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_frequency = lambda_frequency
        self.lambda_edge = lambda_edge
        self.lambda_lesion = lambda_lesion
        self.lambda_adversarial = lambda_adversarial
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.ssim_loss = StructuralSimilarityLoss()
        self.frequency_loss = FrequencyDomainLoss()
        self.edge_loss = EdgePreservingLoss()
        self.lesion_loss = LesionConsistencyLoss()
        self.adversarial_loss = AdversarialLoss()
        
        try:
            self.lpips_loss = lpips.LPIPS(net='vgg')
        except:
            self.lpips_loss = None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                lesion_mask: Optional[torch.Tensor] = None,
                discriminator_pred: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        losses['l1'] = self.l1_loss(pred, target)
        
        losses['perceptual'] = self.perceptual_loss(pred, target)
        
        losses['ssim'] = self.ssim_loss(pred, target)
        
        losses['frequency'] = self.frequency_loss(pred, target)
        
        losses['edge'] = self.edge_loss(pred, target)
        
        if lesion_mask is not None:
            losses['lesion'] = self.lesion_loss(pred, target, lesion_mask)
        else:
            losses['lesion'] = torch.tensor(0.0, device=pred.device)
        
        if discriminator_pred is not None:
            losses['adversarial'] = self.adversarial_loss(discriminator_pred, is_real=False)
        else:
            losses['adversarial'] = torch.tensor(0.0, device=pred.device)
        
        if self.lpips_loss is not None:
            with torch.no_grad():
                losses['lpips'] = self.lpips_loss(pred, target).mean()
        else:
            losses['lpips'] = torch.tensor(0.0, device=pred.device)
        
        total_loss = (
            self.lambda_l1 * losses['l1'] +
            self.lambda_perceptual * losses['perceptual'] +
            self.lambda_ssim * losses['ssim'] +
            self.lambda_frequency * losses['frequency'] +
            self.lambda_edge * losses['edge'] +
            self.lambda_lesion * losses['lesion'] +
            self.lambda_adversarial * losses['adversarial']
        )
        
        losses['total'] = total_loss
        
        return losses


class DiffusionLoss(nn.Module):
    def __init__(self, loss_type: str = 'l2', use_weighted: bool = True):
        super().__init__()
        self.loss_type = loss_type
        self.use_weighted = use_weighted
        
        if loss_type == 'l1':
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.base_loss = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        loss = self.base_loss(predicted, target)
        
        if self.use_weighted and timesteps is not None:
            weights = 1.0 / (1.0 + timesteps.float() / 1000.0)
            weights = weights.view(-1, 1, 1, 1)
            loss = loss * weights
        
        return loss.mean()