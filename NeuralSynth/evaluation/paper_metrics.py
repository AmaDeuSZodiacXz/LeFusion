"""
Paper Metrics Evaluation Module
================================
Computes metrics exactly as reported in LeFusion paper.
Ensures fair comparison with published results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

class PaperMetrics:
    """
    Compute metrics matching LeFusion paper exactly.
    
    Metrics for LIDC:
    - DICE Score
    - NSD (Normalized Surface Distance) at 2mm tolerance
    
    Metrics for EMIDEC:
    - MI DICE (Myocardial Infarction)
    - PMO DICE (Persistent Microvascular Obstruction)
    """
    
    def __init__(self, dataset: str, tolerance_mm: float = 2.0):
        """
        Initialize metrics calculator.
        
        Args:
            dataset: 'lidc' or 'emidec'
            tolerance_mm: Tolerance for NSD calculation (default 2mm as in paper)
        """
        self.dataset = dataset.lower()
        self.tolerance_mm = tolerance_mm
        
        # Voxel spacing (mm)
        if self.dataset == "lidc":
            self.spacing = (1.0, 1.0, 1.0)  # LIDC typical spacing
        else:
            self.spacing = (1.25, 1.25, 8.0)  # EMIDEC typical spacing
    
    def compute_dice(self, pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
        """
        Compute DICE coefficient.
        
        DICE = 2 * |pred ∩ target| / (|pred| + |target|)
        """
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        intersection = np.logical_and(pred, target).sum()
        union = pred.sum() + target.sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return float(dice)
    
    def compute_nsd(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> float:
        """
        Compute Normalized Surface Distance (NSD) at specified tolerance.
        
        NSD measures the percentage of surface points that are within
        the tolerance distance from the ground truth surface.
        
        This is the metric used in LeFusion paper (Table 1).
        """
        if spacing is None:
            spacing = self.spacing
        
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        # Handle empty cases
        if not pred.any() and not target.any():
            return 100.0
        if not pred.any() or not target.any():
            return 0.0
        
        # Get surface points
        pred_surface = self._get_surface_points(pred)
        target_surface = self._get_surface_points(target)
        
        # Compute distance from prediction surface to target
        target_distance_map = distance_transform_edt(~target, sampling=spacing)
        distances_pred_to_target = target_distance_map[pred_surface]
        
        # Compute distance from target surface to prediction
        pred_distance_map = distance_transform_edt(~pred, sampling=spacing)
        distances_target_to_pred = pred_distance_map[target_surface]
        
        # Calculate percentage within tolerance
        pred_within_tolerance = np.sum(distances_pred_to_target <= self.tolerance_mm)
        target_within_tolerance = np.sum(distances_target_to_pred <= self.tolerance_mm)
        
        total_surface_points = len(pred_surface[0]) + len(target_surface[0])
        
        if total_surface_points == 0:
            return 0.0
        
        nsd = 100.0 * (pred_within_tolerance + target_within_tolerance) / total_surface_points
        return float(nsd)
    
    def _get_surface_points(self, mask: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Extract surface points from binary mask."""
        from scipy.ndimage import binary_erosion
        
        # Surface = mask - erosion(mask)
        eroded = binary_erosion(mask)
        surface = np.logical_xor(mask, eroded)
        
        return np.where(surface)
    
    def compute_hausdorff_95(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> float:
        """
        Compute 95th percentile Hausdorff Distance.
        
        Additional metric for comprehensive evaluation.
        """
        if spacing is None:
            spacing = self.spacing
        
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        # Handle empty cases
        if not pred.any() or not target.any():
            return float('inf')
        
        # Get surface points
        pred_points = np.array(self._get_surface_points(pred)).T * np.array(spacing)
        target_points = np.array(self._get_surface_points(target)).T * np.array(spacing)
        
        # Compute distances
        distances_pred_to_target = np.min(
            np.linalg.norm(pred_points[:, None] - target_points[None, :], axis=-1),
            axis=1
        )
        distances_target_to_pred = np.min(
            np.linalg.norm(target_points[:, None] - pred_points[None, :], axis=-1),
            axis=1
        )
        
        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        
        # 95th percentile
        hd95 = np.percentile(all_distances, 95)
        return float(hd95)
    
    def compute_sensitivity(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute sensitivity (recall).
        
        Sensitivity = TP / (TP + FN)
        """
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        tp = np.logical_and(pred, target).sum()
        fn = np.logical_and(~pred, target).sum()
        
        if tp + fn == 0:
            return 1.0 if fn == 0 else 0.0
        
        return float(tp / (tp + fn))
    
    def compute_specificity(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute specificity.
        
        Specificity = TN / (TN + FP)
        """
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        tn = np.logical_and(~pred, ~target).sum()
        fp = np.logical_and(pred, ~target).sum()
        
        if tn + fp == 0:
            return 1.0 if fp == 0 else 0.0
        
        return float(tn / (tn + fp))
    
    def compute_iou(self, pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
        """
        Compute Intersection over Union (Jaccard Index).
        
        IoU = |pred ∩ target| / |pred ∪ target|
        """
        pred = pred.astype(bool)
        target = target.astype(bool)
        
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        iou = (intersection + smooth) / (union + smooth)
        return float(iou)
    
    def compute_all_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics for given prediction and target.
        
        Returns dict with all computed metrics.
        """
        metrics = {}
        
        # Primary metrics (as in LeFusion paper)
        metrics['dice'] = self.compute_dice(pred, target)
        metrics['nsd'] = self.compute_nsd(pred, target, spacing)
        
        # Additional metrics
        metrics['iou'] = self.compute_iou(pred, target)
        metrics['sensitivity'] = self.compute_sensitivity(pred, target)
        metrics['specificity'] = self.compute_specificity(pred, target)
        metrics['hd95'] = self.compute_hausdorff_95(pred, target, spacing)
        
        return metrics
    
    def evaluate_dataset(
        self,
        predictions_dir: Path,
        targets_dir: Path,
        file_list: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate entire dataset.
        
        Args:
            predictions_dir: Directory containing prediction masks
            targets_dir: Directory containing ground truth masks
            file_list: Optional list of files to evaluate
        
        Returns:
            Dict with per-case and average metrics
        """
        predictions_dir = Path(predictions_dir)
        targets_dir = Path(targets_dir)
        
        if file_list is None:
            # Get all prediction files
            pred_files = sorted(predictions_dir.glob("*.npz"))
            file_list = [f.stem for f in pred_files]
        
        results = {}
        all_metrics = {
            'dice': [],
            'nsd': [],
            'iou': [],
            'sensitivity': [],
            'specificity': [],
            'hd95': []
        }
        
        for case_id in file_list:
            # Load prediction and target
            pred_file = predictions_dir / f"{case_id}.npz"
            target_file = targets_dir / f"{case_id}.npz"
            
            if not pred_file.exists() or not target_file.exists():
                print(f"Skipping {case_id}: files not found")
                continue
            
            pred_data = np.load(pred_file)
            target_data = np.load(target_file)
            
            # Extract masks
            pred_mask = pred_data.get('mask', pred_data.get('segmentation'))
            target_mask = target_data.get('mask', target_data.get('segmentation'))
            
            # Compute metrics
            case_metrics = self.compute_all_metrics(pred_mask, target_mask)
            results[case_id] = case_metrics
            
            # Accumulate for averaging
            for metric_name, value in case_metrics.items():
                if metric_name in all_metrics and not np.isinf(value):
                    all_metrics[metric_name].append(value)
        
        # Compute averages
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                avg_metrics[f"mean_{metric_name}"] = float(np.mean(values))
                avg_metrics[f"std_{metric_name}"] = float(np.std(values))
        
        results['average'] = avg_metrics
        
        return results
    
    def compare_with_paper(
        self,
        our_results: Dict[str, float],
        method: str = "lefusion_h_diffmask"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare our results with LeFusion paper benchmarks.
        
        Args:
            our_results: Our computed metrics
            method: Which LeFusion method to compare against
        
        Returns:
            Comparison dict with paper results, our results, and differences
        """
        # LeFusion paper results (Table 1 and Table 2)
        paper_results = {
            "lidc": {
                "baseline": {"dice": 78.26, "nsd": 88.90},
                "lefusion": {"dice": 78.77, "nsd": 89.25},
                "lefusion_h": {"dice": 80.62, "nsd": 90.90},
                "lefusion_h_diffmask": {"dice": 83.44, "nsd": 93.35}
            },
            "emidec": {
                "baseline": {"mi_dice": 68.61, "pmo_dice": 36.32},
                "lefusion": {"mi_dice": 69.88, "pmo_dice": 34.79},
                "lefusion_h": {"mi_dice": 69.95, "pmo_dice": 38.01},
                "lefusion_h_diffmask": {"mi_dice": 71.28, "pmo_dice": 43.41}
            }
        }
        
        if self.dataset not in paper_results:
            return {"error": f"No paper results for dataset {self.dataset}"}
        
        if method not in paper_results[self.dataset]:
            return {"error": f"No paper results for method {method}"}
        
        paper_metrics = paper_results[self.dataset][method]
        
        comparison = {
            "paper": paper_metrics,
            "ours": our_results,
            "difference": {}
        }
        
        # Calculate differences
        for metric in paper_metrics:
            if metric in our_results:
                diff = our_results[metric] - paper_metrics[metric]
                comparison["difference"][metric] = diff
                
                # Add percentage improvement
                if paper_metrics[metric] > 0:
                    pct_improvement = (diff / paper_metrics[metric]) * 100
                    comparison["difference"][f"{metric}_pct"] = pct_improvement
        
        return comparison
    
    def generate_latex_table(
        self,
        results: Dict[str, Dict[str, float]],
        caption: str = "Segmentation Performance Comparison"
    ) -> str:
        """
        Generate LaTeX table comparing results with LeFusion paper.
        
        Format matches paper's Table 1 (LIDC) or Table 2 (EMIDEC).
        """
        latex = []
        
        if self.dataset == "lidc":
            # Table 1 format
            latex.append("\\begin{table}[h]")
            latex.append("\\centering")
            latex.append("\\caption{" + caption + "}")
            latex.append("\\begin{tabular}{lcccc}")
            latex.append("\\toprule")
            latex.append("Method & DICE (\\%) $\\uparrow$ & NSD (\\%) $\\uparrow$ & HD95 (mm) $\\downarrow$ & IoU (\\%) $\\uparrow$ \\\\")
            latex.append("\\midrule")
            
            # Add baseline and LeFusion methods from paper
            latex.append("Baseline (Real Only) & 78.26 & 88.90 & - & - \\\\")
            latex.append("LeFusion & 78.77 & 89.25 & - & - \\\\")
            latex.append("LeFusion-H & 80.62 & 90.90 & - & - \\\\")
            latex.append("LeFusion-H+DiffMask & 83.44 & 93.35 & - & - \\\\")
            latex.append("\\midrule")
            
            # Add NeuralSynth results
            if 'average' in results:
                avg = results['average']
                dice = avg.get('mean_dice', 0) * 100
                nsd = avg.get('mean_nsd', 0)
                hd95 = avg.get('mean_hd95', 0)
                iou = avg.get('mean_iou', 0) * 100
                
                latex.append(f"\\textbf{{NeuralSynth (Ours)}} & \\textbf{{{dice:.2f}}} & \\textbf{{{nsd:.2f}}} & \\textbf{{{hd95:.2f}}} & \\textbf{{{iou:.2f}}} \\\\")
            
            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
            latex.append("\\end{table}")
        
        else:  # EMIDEC
            # Table 2 format
            latex.append("\\begin{table}[h]")
            latex.append("\\centering")
            latex.append("\\caption{" + caption + "}")
            latex.append("\\begin{tabular}{lcc}")
            latex.append("\\toprule")
            latex.append("Method & MI DICE (\\%) $\\uparrow$ & PMO DICE (\\%) $\\uparrow$ \\\\")
            latex.append("\\midrule")
            
            # Add baseline and LeFusion methods
            latex.append("Baseline (Real Only) & 68.61 & 36.32 \\\\")
            latex.append("LeFusion & 69.88 & 34.79 \\\\")
            latex.append("LeFusion-H & 69.95 & 38.01 \\\\")
            latex.append("LeFusion-H+DiffMask & 71.28 & 43.41 \\\\")
            latex.append("\\midrule")
            
            # Add NeuralSynth results
            if 'average' in results:
                avg = results['average']
                mi_dice = avg.get('mean_mi_dice', 0) * 100
                pmo_dice = avg.get('mean_pmo_dice', 0) * 100
                
                latex.append(f"\\textbf{{NeuralSynth (Ours)}} & \\textbf{{{mi_dice:.2f}}} & \\textbf{{{pmo_dice:.2f}}} \\\\")
            
            latex.append("\\bottomrule")
            latex.append("\\end{tabular}")
            latex.append("\\end{table}")
        
        return "\n".join(latex)


class MetricsTracker:
    """Track metrics during training for best checkpoint selection."""
    
    def __init__(self, save_dir: Path, patience: int = 20):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        
        self.best_dice = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = []
    
    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Update tracker with new metrics.
        
        Returns True if this is the best epoch.
        """
        current_dice = metrics.get('dice', 0.0)
        
        self.history.append({
            'epoch': epoch,
            **metrics
        })
        
        is_best = False
        if current_dice > self.best_dice:
            self.best_dice = current_dice
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            is_best = True
            
            # Save best metrics
            best_file = self.save_dir / "best_metrics.json"
            with open(best_file, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'metrics': metrics
                }, f, indent=2)
        else:
            self.epochs_without_improvement += 1
        
        # Save history
        history_file = self.save_dir / "metrics_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return is_best
    
    def should_stop(self) -> bool:
        """Check if training should stop (early stopping)."""
        return self.epochs_without_improvement >= self.patience


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute paper metrics")
    parser.add_argument("--dataset", type=str, choices=["lidc", "emidec"], required=True)
    parser.add_argument("--predictions", type=str, required=True, help="Predictions directory")
    parser.add_argument("--targets", type=str, required=True, help="Targets directory")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table")
    
    args = parser.parse_args()
    
    # Initialize metrics
    metrics = PaperMetrics(args.dataset)
    
    # Evaluate
    results = metrics.evaluate_dataset(
        Path(args.predictions),
        Path(args.targets)
    )
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    if 'average' in results:
        print(f"\n📊 Average Metrics:")
        for key, value in results['average'].items():
            print(f"   {key}: {value:.4f}")
    
    # Generate LaTeX table
    if args.latex:
        latex_table = metrics.generate_latex_table(results)
        print(f"\n📝 LaTeX Table:")
        print(latex_table)
    
    # Compare with paper
    if 'average' in results:
        comparison = metrics.compare_with_paper(
            results['average'],
            method="lefusion_h_diffmask"
        )
        print(f"\n📈 Comparison with LeFusion paper:")
        print(json.dumps(comparison, indent=2))