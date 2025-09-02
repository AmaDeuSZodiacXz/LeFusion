import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import ndimage, stats
from skimage import measure, morphology
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings
warnings.filterwarnings('ignore')


class AdvancedMetrics:
    
    @staticmethod
    def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = np.sum(pred_flat * target_flat)
        return (2.0 * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)
    
    @staticmethod
    def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = np.sum(pred_flat * target_flat)
        union = np.sum(pred_flat) + np.sum(target_flat) - intersection
        return (intersection + smooth) / (union + smooth)
    
    @staticmethod
    def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
        from scipy.spatial.distance import directed_hausdorff
        
        pred_points = np.column_stack(np.where(pred > 0))
        target_points = np.column_stack(np.where(target > 0))
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        forward_hd = directed_hausdorff(pred_points, target_points)[0]
        backward_hd = directed_hausdorff(target_points, pred_points)[0]
        
        return max(forward_hd, backward_hd)
    
    @staticmethod
    def average_surface_distance(pred: np.ndarray, target: np.ndarray) -> float:
        from scipy.ndimage import distance_transform_edt
        
        pred_surface = morphology.erosion(pred) ^ pred
        target_surface = morphology.erosion(target) ^ target
        
        if not np.any(pred_surface) or not np.any(target_surface):
            return float('inf')
        
        dist_pred = distance_transform_edt(~target_surface)
        dist_target = distance_transform_edt(~pred_surface)
        
        distances_pred = dist_pred[pred_surface > 0]
        distances_target = dist_target[target_surface > 0]
        
        avg_dist = (np.mean(distances_pred) + np.mean(distances_target)) / 2.0
        return avg_dist
    
    @staticmethod
    def volumetric_similarity(pred: np.ndarray, target: np.ndarray) -> float:
        vol_pred = np.sum(pred)
        vol_target = np.sum(target)
        
        if vol_pred + vol_target == 0:
            return 1.0
        
        vol_diff = abs(vol_pred - vol_target)
        vol_sum = vol_pred + vol_target
        
        return 1.0 - (2.0 * vol_diff / vol_sum)
    
    @staticmethod
    def false_positive_rate(pred: np.ndarray, target: np.ndarray) -> float:
        fp = np.sum((pred == 1) & (target == 0))
        tn = np.sum((pred == 0) & (target == 0))
        
        if fp + tn == 0:
            return 0.0
        
        return fp / (fp + tn)
    
    @staticmethod
    def false_negative_rate(pred: np.ndarray, target: np.ndarray) -> float:
        fn = np.sum((pred == 0) & (target == 1))
        tp = np.sum((pred == 1) & (target == 1))
        
        if fn + tp == 0:
            return 0.0
        
        return fn / (fn + tp)
    
    @staticmethod
    def sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
        tp = np.sum((pred == 1) & (target == 1))
        fn = np.sum((pred == 0) & (target == 1))
        
        if tp + fn == 0:
            return 0.0
        
        return tp / (tp + fn)
    
    @staticmethod
    def specificity(pred: np.ndarray, target: np.ndarray) -> float:
        tn = np.sum((pred == 0) & (target == 0))
        fp = np.sum((pred == 1) & (target == 0))
        
        if tn + fp == 0:
            return 0.0
        
        return tn / (tn + fp)
    
    @staticmethod
    def precision(pred: np.ndarray, target: np.ndarray) -> float:
        tp = np.sum((pred == 1) & (target == 1))
        fp = np.sum((pred == 1) & (target == 0))
        
        if tp + fp == 0:
            return 0.0
        
        return tp / (tp + fp)
    
    @staticmethod
    def f1_score(pred: np.ndarray, target: np.ndarray) -> float:
        prec = AdvancedMetrics.precision(pred, target)
        sens = AdvancedMetrics.sensitivity(pred, target)
        
        if prec + sens == 0:
            return 0.0
        
        return 2 * (prec * sens) / (prec + sens)
    
    @staticmethod
    def matthews_correlation_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
        tp = np.sum((pred == 1) & (target == 1))
        tn = np.sum((pred == 0) & (target == 0))
        fp = np.sum((pred == 1) & (target == 0))
        fn = np.sum((pred == 0) & (target == 1))
        
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        return (tp * tn - fp * fn) / denominator
    
    @staticmethod
    def balanced_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
        sens = AdvancedMetrics.sensitivity(pred, target)
        spec = AdvancedMetrics.specificity(pred, target)
        return (sens + spec) / 2.0


class ImageQualityMetrics:
    
    @staticmethod
    def structural_similarity_index(pred: np.ndarray, target: np.ndarray, 
                                   data_range: Optional[float] = None) -> float:
        if data_range is None:
            data_range = target.max() - target.min()
        
        return ssim(pred, target, data_range=data_range, multichannel=False)
    
    @staticmethod
    def peak_signal_noise_ratio(pred: np.ndarray, target: np.ndarray, 
                               data_range: Optional[float] = None) -> float:
        if data_range is None:
            data_range = target.max() - target.min()
        
        return psnr(target, pred, data_range=data_range)
    
    @staticmethod
    def mean_absolute_error(pred: np.ndarray, target: np.ndarray) -> float:
        return np.mean(np.abs(pred - target))
    
    @staticmethod
    def root_mean_squared_error(pred: np.ndarray, target: np.ndarray) -> float:
        return np.sqrt(np.mean((pred - target) ** 2))
    
    @staticmethod
    def normalized_cross_correlation(pred: np.ndarray, target: np.ndarray) -> float:
        pred_norm = (pred - np.mean(pred)) / (np.std(pred) + 1e-7)
        target_norm = (target - np.mean(target)) / (np.std(target) + 1e-7)
        return np.mean(pred_norm * target_norm)
    
    @staticmethod
    def gradient_magnitude_similarity(pred: np.ndarray, target: np.ndarray) -> float:
        sobelx = ndimage.sobel(pred, axis=0)
        sobely = ndimage.sobel(pred, axis=1)
        pred_grad = np.sqrt(sobelx**2 + sobely**2)
        
        sobelx = ndimage.sobel(target, axis=0)
        sobely = ndimage.sobel(target, axis=1)
        target_grad = np.sqrt(sobelx**2 + sobely**2)
        
        c = 1e-4
        gms = (2 * pred_grad * target_grad + c) / (pred_grad**2 + target_grad**2 + c)
        return np.mean(gms)
    
    @staticmethod
    def mutual_information(pred: np.ndarray, target: np.ndarray, bins: int = 256) -> float:
        hist_2d, _, _ = np.histogram2d(pred.ravel(), target.ravel(), bins=bins)
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / (px_py[nzs] + 1e-7)))
        return mi
    
    @staticmethod
    def visual_information_fidelity(pred: np.ndarray, target: np.ndarray) -> float:
        sigma_nsq = 2
        sigma_sq = np.var(target)
        
        if sigma_sq < 1e-7:
            return 1.0
        
        error = pred - target
        mse = np.mean(error ** 2)
        
        vif = np.log10(1 + sigma_sq / (mse + sigma_nsq))
        return vif


class TexturalMetrics:
    
    @staticmethod
    def gray_level_cooccurrence_matrix(image: np.ndarray, distances: List[int] = [1], 
                                      angles: List[float] = [0]) -> np.ndarray:
        from skimage.feature import graycomatrix
        
        image_uint = (image * 255).astype(np.uint8)
        glcm = graycomatrix(image_uint, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        return glcm
    
    @staticmethod
    def contrast(glcm: np.ndarray) -> float:
        n = glcm.shape[0]
        contrast = 0
        for i in range(n):
            for j in range(n):
                contrast += (i - j) ** 2 * glcm[i, j, 0, 0]
        return contrast
    
    @staticmethod
    def dissimilarity(glcm: np.ndarray) -> float:
        n = glcm.shape[0]
        diss = 0
        for i in range(n):
            for j in range(n):
                diss += abs(i - j) * glcm[i, j, 0, 0]
        return diss
    
    @staticmethod
    def homogeneity(glcm: np.ndarray) -> float:
        n = glcm.shape[0]
        homo = 0
        for i in range(n):
            for j in range(n):
                homo += glcm[i, j, 0, 0] / (1 + abs(i - j))
        return homo
    
    @staticmethod
    def energy(glcm: np.ndarray) -> float:
        return np.sum(glcm[:, :, 0, 0] ** 2)
    
    @staticmethod
    def correlation(glcm: np.ndarray) -> float:
        n = glcm.shape[0]
        mean_i = 0
        mean_j = 0
        
        for i in range(n):
            for j in range(n):
                mean_i += i * glcm[i, j, 0, 0]
                mean_j += j * glcm[i, j, 0, 0]
        
        std_i = 0
        std_j = 0
        
        for i in range(n):
            for j in range(n):
                std_i += (i - mean_i) ** 2 * glcm[i, j, 0, 0]
                std_j += (j - mean_j) ** 2 * glcm[i, j, 0, 0]
        
        std_i = np.sqrt(std_i)
        std_j = np.sqrt(std_j)
        
        if std_i < 1e-7 or std_j < 1e-7:
            return 0.0
        
        corr = 0
        for i in range(n):
            for j in range(n):
                corr += (i - mean_i) * (j - mean_j) * glcm[i, j, 0, 0] / (std_i * std_j)
        
        return corr


class ClinicalRelevanceMetrics:
    
    @staticmethod
    def lesion_detection_rate(pred_masks: List[np.ndarray], 
                             target_masks: List[np.ndarray], 
                             threshold: float = 0.5) -> float:
        detected = 0
        total = len(target_masks)
        
        for pred, target in zip(pred_masks, target_masks):
            if np.any(target > 0):
                overlap = AdvancedMetrics.dice_coefficient(pred > threshold, target > 0)
                if overlap > 0.1:
                    detected += 1
        
        return detected / total if total > 0 else 0.0
    
    @staticmethod
    def lesion_localization_error(pred_centroid: Tuple[float, float], 
                                 target_centroid: Tuple[float, float]) -> float:
        return np.sqrt((pred_centroid[0] - target_centroid[0])**2 + 
                      (pred_centroid[1] - target_centroid[1])**2)
    
    @staticmethod
    def size_estimation_error(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
        pred_size = np.sum(pred_mask > 0)
        target_size = np.sum(target_mask > 0)
        
        if target_size == 0:
            return float('inf')
        
        return abs(pred_size - target_size) / target_size
    
    @staticmethod
    def shape_similarity(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
        from skimage.measure import find_contours
        from scipy.spatial import procrustes
        
        pred_contours = find_contours(pred_mask, 0.5)
        target_contours = find_contours(target_mask, 0.5)
        
        if len(pred_contours) == 0 or len(target_contours) == 0:
            return 0.0
        
        pred_points = pred_contours[0]
        target_points = target_contours[0]
        
        min_points = min(len(pred_points), len(target_points))
        pred_points = pred_points[:min_points]
        target_points = target_points[:min_points]
        
        _, _, disparity = procrustes(target_points, pred_points)
        
        return 1.0 / (1.0 + disparity)
    
    @staticmethod
    def boundary_f1_score(pred_mask: np.ndarray, target_mask: np.ndarray, 
                         theta: int = 3) -> float:
        from skimage.segmentation import find_boundaries
        
        pred_boundary = find_boundaries(pred_mask, mode='inner')
        target_boundary = find_boundaries(target_mask, mode='inner')
        
        dist_pred = ndimage.distance_transform_edt(~target_boundary)
        dist_target = ndimage.distance_transform_edt(~pred_boundary)
        
        precision = np.sum(dist_target[pred_boundary] <= theta) / np.sum(pred_boundary)
        recall = np.sum(dist_pred[target_boundary] <= theta) / np.sum(target_boundary)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)


class ComprehensiveEvaluator:
    def __init__(self):
        self.segmentation_metrics = AdvancedMetrics()
        self.quality_metrics = ImageQualityMetrics()
        self.texture_metrics = TexturalMetrics()
        self.clinical_metrics = ClinicalRelevanceMetrics()
    
    def evaluate_all(self, pred: np.ndarray, target: np.ndarray, 
                    pred_mask: Optional[np.ndarray] = None,
                    target_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        
        results = {}
        
        if pred_mask is not None and target_mask is not None:
            results['dice'] = self.segmentation_metrics.dice_coefficient(pred_mask, target_mask)
            results['iou'] = self.segmentation_metrics.iou_score(pred_mask, target_mask)
            results['hausdorff'] = self.segmentation_metrics.hausdorff_distance(pred_mask, target_mask)
            results['asd'] = self.segmentation_metrics.average_surface_distance(pred_mask, target_mask)
            results['volumetric_similarity'] = self.segmentation_metrics.volumetric_similarity(pred_mask, target_mask)
            results['sensitivity'] = self.segmentation_metrics.sensitivity(pred_mask, target_mask)
            results['specificity'] = self.segmentation_metrics.specificity(pred_mask, target_mask)
            results['precision'] = self.segmentation_metrics.precision(pred_mask, target_mask)
            results['f1_score'] = self.segmentation_metrics.f1_score(pred_mask, target_mask)
            results['mcc'] = self.segmentation_metrics.matthews_correlation_coefficient(pred_mask, target_mask)
            results['balanced_accuracy'] = self.segmentation_metrics.balanced_accuracy(pred_mask, target_mask)
            results['boundary_f1'] = self.clinical_metrics.boundary_f1_score(pred_mask, target_mask)
        
        results['ssim'] = self.quality_metrics.structural_similarity_index(pred, target)
        results['psnr'] = self.quality_metrics.peak_signal_noise_ratio(pred, target)
        results['mae'] = self.quality_metrics.mean_absolute_error(pred, target)
        results['rmse'] = self.quality_metrics.root_mean_squared_error(pred, target)
        results['ncc'] = self.quality_metrics.normalized_cross_correlation(pred, target)
        results['gms'] = self.quality_metrics.gradient_magnitude_similarity(pred, target)
        results['mi'] = self.quality_metrics.mutual_information(pred, target)
        results['vif'] = self.quality_metrics.visual_information_fidelity(pred, target)
        
        glcm_pred = self.texture_metrics.gray_level_cooccurrence_matrix(pred)
        glcm_target = self.texture_metrics.gray_level_cooccurrence_matrix(target)
        
        results['contrast_diff'] = abs(self.texture_metrics.contrast(glcm_pred) - 
                                      self.texture_metrics.contrast(glcm_target))
        results['homogeneity_diff'] = abs(self.texture_metrics.homogeneity(glcm_pred) - 
                                         self.texture_metrics.homogeneity(glcm_target))
        results['energy_diff'] = abs(self.texture_metrics.energy(glcm_pred) - 
                                    self.texture_metrics.energy(glcm_target))
        results['correlation_diff'] = abs(self.texture_metrics.correlation(glcm_pred) - 
                                         self.texture_metrics.correlation(glcm_target))
        
        return results