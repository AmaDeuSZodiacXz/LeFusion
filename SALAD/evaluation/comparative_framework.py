import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
import json
from tqdm import tqdm
import nibabel as nib

from models.neuralsynth_core import NeuralSynthConfig, NeuralSynthDiffusion
from evaluation.advanced_metrics import ComprehensiveEvaluator


class ModelComparator:
    def __init__(self, results_dir: str = './comparison_results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ComprehensiveEvaluator()
        self.models = {}
        self.results = []
        
    def load_neuralsynth_model(self, checkpoint_path: str, config: Optional[NeuralSynthConfig] = None):
        if config is None:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            config_dict = checkpoint['config']
            config = NeuralSynthConfig(**config_dict)
        
        model = NeuralSynthDiffusion(config)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        self.models['NeuralSynth'] = {
            'model': model,
            'config': config,
            'type': 'neuralsynth'
        }
        
        return model
    
    def load_lefusion_model(self, checkpoint_path: str):
        try:
            sys.path.append('/Users/skb/Documents/LeFusion/LeFusion')
            from lefusion_model import LeFusionModel
            
            model = LeFusionModel.load_from_checkpoint(checkpoint_path)
            model.eval()
            
            self.models['LeFusion'] = {
                'model': model,
                'type': 'lefusion'
            }
        except Exception as e:
            print(f"Could not load LeFusion model: {e}")
            return None
    
    def load_scar_model(self, checkpoint_path: str):
        try:
            sys.path.append('/Users/skb/Documents/LeFusion/CLAIM-Scar-Synthesis')
            from scar_model import SCARModel
            
            model = SCARModel.load_from_checkpoint(checkpoint_path)
            model.eval()
            
            self.models['SCAR'] = {
                'model': model,
                'type': 'scar'
            }
        except Exception as e:
            print(f"Could not load SCAR model: {e}")
            return None
    
    def generate_synthetic_image(self, model_name: str, mask: torch.Tensor, 
                                condition: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, float]:
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        mask = mask.to(device)
        
        start_time = time.time()
        
        with torch.no_grad():
            if model_info['type'] == 'neuralsynth':
                output = model.sample(
                    shape=(1, 1, mask.shape[-2], mask.shape[-1]),
                    lesion_mask=mask,
                    device=device
                )
            elif model_info['type'] == 'lefusion':
                output = model.generate(mask, condition)
            elif model_info['type'] == 'scar':
                output = model.synthesize(mask)
            else:
                raise ValueError(f"Unknown model type: {model_info['type']}")
        
        inference_time = time.time() - start_time
        
        return output, inference_time
    
    def evaluate_model_on_dataset(self, model_name: str, test_dataset, 
                                 num_samples: int = 100) -> Dict[str, float]:
        
        all_metrics = []
        inference_times = []
        
        progress_bar = tqdm(range(min(num_samples, len(test_dataset))), 
                          desc=f"Evaluating {model_name}")
        
        for idx in progress_bar:
            sample = test_dataset[idx]
            real_image = sample['image']
            mask = sample['mask']
            
            synthetic_image, inference_time = self.generate_synthetic_image(
                model_name, mask.unsqueeze(0)
            )
            
            synthetic_image = synthetic_image.squeeze().cpu().numpy()
            real_image = real_image.squeeze().numpy()
            mask_np = mask.squeeze().numpy()
            
            metrics = self.evaluator.evaluate_all(
                synthetic_image,
                real_image,
                mask_np > 0.5,
                mask_np > 0.5
            )
            
            all_metrics.append(metrics)
            inference_times.append(inference_time)
            
            avg_dice = np.mean([m.get('dice', 0) for m in all_metrics if 'dice' in m])
            avg_ssim = np.mean([m.get('ssim', 0) for m in all_metrics if 'ssim' in m])
            
            progress_bar.set_postfix({
                'dice': f'{avg_dice:.4f}',
                'ssim': f'{avg_ssim:.4f}',
                'time': f'{np.mean(inference_times):.3f}s'
            })
        
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics 
                     if key in m and not np.isnan(m[key]) and not np.isinf(m[key])]
            if values:
                aggregated_metrics[f'{key}_mean'] = np.mean(values)
                aggregated_metrics[f'{key}_std'] = np.std(values)
                aggregated_metrics[f'{key}_median'] = np.median(values)
        
        aggregated_metrics['inference_time_mean'] = np.mean(inference_times)
        aggregated_metrics['inference_time_std'] = np.std(inference_times)
        aggregated_metrics['throughput_fps'] = 1.0 / np.mean(inference_times)
        
        return aggregated_metrics
    
    def compare_all_models(self, test_dataset, num_samples: int = 100):
        comparison_results = {}
        
        for model_name in self.models.keys():
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name}")
            print(f"{'='*50}")
            
            metrics = self.evaluate_model_on_dataset(
                model_name, test_dataset, num_samples
            )
            
            comparison_results[model_name] = metrics
            
            print(f"\n{model_name} Results:")
            print(f"  Dice: {metrics.get('dice_mean', 0):.4f} ± {metrics.get('dice_std', 0):.4f}")
            print(f"  IoU: {metrics.get('iou_mean', 0):.4f} ± {metrics.get('iou_std', 0):.4f}")
            print(f"  SSIM: {metrics.get('ssim_mean', 0):.4f} ± {metrics.get('ssim_std', 0):.4f}")
            print(f"  PSNR: {metrics.get('psnr_mean', 0):.2f} ± {metrics.get('psnr_std', 0):.2f}")
            print(f"  Inference Time: {metrics.get('inference_time_mean', 0):.3f}s")
            print(f"  Throughput: {metrics.get('throughput_fps', 0):.2f} FPS")
        
        self.save_comparison_results(comparison_results)
        self.create_comparison_plots(comparison_results)
        
        return comparison_results
    
    def save_comparison_results(self, results: Dict):
        with open(self.results_dir / 'comparison_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        df = pd.DataFrame(results).T
        df.to_csv(self.results_dir / 'comparison_results.csv')
        
        df_summary = df[['dice_mean', 'iou_mean', 'ssim_mean', 'psnr_mean', 
                        'inference_time_mean', 'throughput_fps']]
        df_summary.to_csv(self.results_dir / 'comparison_summary.csv')
        
        print(f"\nResults saved to {self.results_dir}")
    
    def create_comparison_plots(self, results: Dict):
        metrics_to_plot = ['dice_mean', 'iou_mean', 'ssim_mean', 'psnr_mean', 
                          'hausdorff_mean', 'sensitivity_mean', 'specificity_mean', 
                          'f1_score_mean']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            models = list(results.keys())
            values = [results[model].get(metric, 0) for model in models]
            errors = [results[model].get(metric.replace('_mean', '_std'), 0) for model in models]
            
            colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(models)]
            
            bars = ax.bar(models, values, yerr=errors, capsize=5, color=colors, alpha=0.8)
            
            ax.set_title(metric.replace('_mean', '').upper(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        models = list(results.keys())
        inference_times = [results[model].get('inference_time_mean', 0) for model in models]
        throughput = [results[model].get('throughput_fps', 0) for model in models]
        
        ax1.bar(models, inference_times, color='#2E86AB', alpha=0.8)
        ax1.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(inference_times):
            ax1.text(i, v + 0.01, f'{v:.3f}s', ha='center', fontsize=10)
        
        ax2.bar(models, throughput, color='#A23B72', alpha=0.8)
        ax2.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('FPS', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(throughput):
            ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        quality_metrics = ['ssim_mean', 'psnr_mean', 'ncc_mean', 'vif_mean']
        clinical_metrics = ['dice_mean', 'iou_mean', 'sensitivity_mean', 'precision_mean']
        
        df_quality = pd.DataFrame({
            model: [results[model].get(m, 0) for m in quality_metrics]
            for model in models
        }, index=quality_metrics)
        
        df_clinical = pd.DataFrame({
            model: [results[model].get(m, 0) for m in clinical_metrics]
            for model in models
        }, index=clinical_metrics)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        df_quality.T.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Image Quality Metrics', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
        
        df_clinical.T.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Clinical Relevance Metrics', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'detailed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comparison_report(self, results: Dict) -> str:
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE MODEL COMPARISON REPORT")
        report.append("="*80)
        report.append("")
        
        models = list(results.keys())
        
        report.append("1. OVERALL PERFORMANCE RANKING")
        report.append("-"*40)
        
        ranking_metrics = ['dice_mean', 'iou_mean', 'ssim_mean', 'psnr_mean', 'f1_score_mean']
        overall_scores = {}
        
        for model in models:
            scores = []
            for metric in ranking_metrics:
                if metric in results[model]:
                    scores.append(results[model][metric])
            overall_scores[model] = np.mean(scores) if scores else 0
        
        ranked_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (model, score) in enumerate(ranked_models, 1):
            report.append(f"  {rank}. {model}: {score:.4f}")
        
        report.append("")
        report.append("2. DETAILED METRICS COMPARISON")
        report.append("-"*40)
        
        for model in models:
            report.append(f"\n{model}:")
            report.append(f"  Segmentation Metrics:")
            report.append(f"    - Dice: {results[model].get('dice_mean', 0):.4f} ± {results[model].get('dice_std', 0):.4f}")
            report.append(f"    - IoU: {results[model].get('iou_mean', 0):.4f} ± {results[model].get('iou_std', 0):.4f}")
            report.append(f"    - F1 Score: {results[model].get('f1_score_mean', 0):.4f}")
            
            report.append(f"  Image Quality Metrics:")
            report.append(f"    - SSIM: {results[model].get('ssim_mean', 0):.4f}")
            report.append(f"    - PSNR: {results[model].get('psnr_mean', 0):.2f} dB")
            report.append(f"    - MAE: {results[model].get('mae_mean', 0):.4f}")
            
            report.append(f"  Clinical Metrics:")
            report.append(f"    - Sensitivity: {results[model].get('sensitivity_mean', 0):.4f}")
            report.append(f"    - Specificity: {results[model].get('specificity_mean', 0):.4f}")
            report.append(f"    - Precision: {results[model].get('precision_mean', 0):.4f}")
            
            report.append(f"  Performance Metrics:")
            report.append(f"    - Inference Time: {results[model].get('inference_time_mean', 0):.3f}s")
            report.append(f"    - Throughput: {results[model].get('throughput_fps', 0):.2f} FPS")
        
        report.append("")
        report.append("3. KEY FINDINGS")
        report.append("-"*40)
        
        best_dice = max(models, key=lambda m: results[m].get('dice_mean', 0))
        best_ssim = max(models, key=lambda m: results[m].get('ssim_mean', 0))
        fastest = min(models, key=lambda m: results[m].get('inference_time_mean', float('inf')))
        
        report.append(f"  - Best Dice Score: {best_dice} ({results[best_dice].get('dice_mean', 0):.4f})")
        report.append(f"  - Best SSIM Score: {best_ssim} ({results[best_ssim].get('ssim_mean', 0):.4f})")
        report.append(f"  - Fastest Inference: {fastest} ({results[fastest].get('inference_time_mean', 0):.3f}s)")
        
        report.append("")
        report.append("4. RECOMMENDATIONS")
        report.append("-"*40)
        
        if 'NeuralSynth' in results:
            ns_dice = results['NeuralSynth'].get('dice_mean', 0)
            ns_time = results['NeuralSynth'].get('inference_time_mean', 0)
            
            improvements = []
            for model in models:
                if model != 'NeuralSynth':
                    dice_improvement = ((ns_dice - results[model].get('dice_mean', 0)) / 
                                      results[model].get('dice_mean', 1)) * 100
                    time_improvement = ((results[model].get('inference_time_mean', 1) - ns_time) / 
                                       results[model].get('inference_time_mean', 1)) * 100
                    
                    if dice_improvement > 0:
                        improvements.append(f"  - NeuralSynth shows {dice_improvement:.1f}% Dice improvement over {model}")
                    if time_improvement > 0:
                        improvements.append(f"  - NeuralSynth is {time_improvement:.1f}% faster than {model}")
            
            if improvements:
                report.extend(improvements)
            else:
                report.append("  - NeuralSynth shows competitive performance across all metrics")
        
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        with open(self.results_dir / 'comparison_report.txt', 'w') as f:
            f.write(report_text)
        
        return report_text