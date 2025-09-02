"""
NeuralSynth Pipeline: Normal-to-Pathological Synthesis
Following LeFusion's approach with enhancements
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import nibabel as nib
from PIL import Image
import json
from dataclasses import dataclass
import logging
from tqdm import tqdm

import sys
sys.path.append('..')
from models.neuralsynth_core import NeuralSynthDiffusion, NeuralSynthConfig
from models.advanced_losses import NeuralSynthLoss


@dataclass
class SynthesisConfig:
    """Configuration for normal-to-pathological synthesis."""
    # Data paths
    normal_data_path: str = "/Users/skb/Documents/LeFusion/data/LIDC/normal"
    pathological_data_path: str = "/Users/skb/Documents/LeFusion/data/LIDC/pathological"
    output_path: str = "/Users/skb/Documents/LeFusion/data/LIDC/synthetic"
    
    # Model settings
    checkpoint_path: str = "./checkpoints/neuralsynth_lidc/best_model.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Synthesis settings
    num_synthetic_per_normal: int = 3  # Generate 3 synthetic lesions per normal case
    lesion_size_range: Tuple[int, int] = (10, 30)  # Lesion size in voxels
    
    # Histogram control (for multi-peak lesions like LIDC)
    use_histogram_control: bool = True
    histogram_clusters: int = 3  # Ground-glass, part-solid, solid
    histogram_weights: List[float] = [0.75, 0.20, 0.05]  # Sampling weights
    
    # Multi-class settings (for EMIDEC)
    multi_class: bool = False
    num_classes: int = 2  # MI and PMO for EMIDEC
    
    # Background preservation
    preserve_background: bool = True
    boundary_smoothing_sigma: float = 2.0
    
    # DDIM sampling
    ddim_steps: int = 50
    ddim_eta: float = 0.0


class LesionMaskGenerator:
    """Generate realistic lesion masks following LeFusion's approach."""
    
    def __init__(self, config: SynthesisConfig):
        self.config = config
        
    def generate_ellipsoid_mask(self, shape: Tuple[int, ...], 
                                center: Optional[Tuple[int, ...]] = None,
                                size: Optional[int] = None) -> np.ndarray:
        """Generate ellipsoid mask for lesion."""
        if center is None:
            # Random center within safe bounds
            center = tuple(np.random.randint(s//4, 3*s//4) for s in shape)
        
        if size is None:
            size = np.random.randint(*self.config.lesion_size_range)
        
        # Create ellipsoid with random aspect ratios
        mask = np.zeros(shape, dtype=np.float32)
        indices = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        # Random ellipsoid radii
        radii = size * (0.5 + 0.5 * np.random.rand(3))
        
        # Calculate distance from center
        distances = sum(((idx - c) / r) ** 2 for idx, c, r in zip(indices, center, radii))
        mask[distances <= 1] = 1.0
        
        # Apply morphological operations for realism
        from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
        if np.random.rand() > 0.5:
            mask = binary_erosion(mask, iterations=1)
        if np.random.rand() > 0.5:
            mask = binary_dilation(mask, iterations=1)
        
        # Smooth boundaries
        mask = gaussian_filter(mask, sigma=1.0)
        mask = (mask > 0.5).astype(np.float32)
        
        return mask
    
    def generate_spiculated_mask(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate spiculated mask for lung nodules."""
        # Start with ellipsoid
        mask = self.generate_ellipsoid_mask(shape)
        
        # Add spiculations
        from scipy.ndimage import rotate, gaussian_filter
        for _ in range(np.random.randint(3, 8)):
            spike = np.zeros(shape)
            # Create thin spike
            center = np.where(mask > 0)
            if len(center[0]) > 0:
                idx = np.random.randint(len(center[0]))
                start = (center[0][idx], center[1][idx], center[2][idx])
                
                # Random direction
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                
                # Draw spike
                length = np.random.randint(5, 15)
                for i in range(length):
                    pos = tuple(int(start[j] + i * direction[j]) for j in range(3))
                    if all(0 <= p < s for p, s in zip(pos, shape)):
                        spike[pos] = 1.0
                
                mask = np.maximum(mask, spike)
        
        # Smooth
        mask = gaussian_filter(mask, sigma=0.5)
        mask = (mask > 0.3).astype(np.float32)
        
        return mask


class HistogramController:
    """Control lesion texture via histogram following LeFusion."""
    
    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.histogram_templates = self._load_histogram_templates()
    
    def _load_histogram_templates(self) -> Dict[int, np.ndarray]:
        """Load histogram templates for different lesion types."""
        templates = {}
        
        # Cluster 1: Ground-glass (lighter, more spread)
        templates[0] = self._create_gaussian_histogram(mean=50, std=20)
        
        # Cluster 2: Part-solid (medium intensity)
        templates[1] = self._create_gaussian_histogram(mean=100, std=15)
        
        # Cluster 3: Solid (darker, concentrated)
        templates[2] = self._create_gaussian_histogram(mean=150, std=10)
        
        return templates
    
    def _create_gaussian_histogram(self, mean: float, std: float, bins: int = 256) -> np.ndarray:
        """Create Gaussian histogram template."""
        x = np.arange(bins)
        hist = np.exp(-(x - mean) ** 2 / (2 * std ** 2))
        hist = hist / hist.sum()
        return hist
    
    def sample_histogram(self) -> np.ndarray:
        """Sample a histogram based on configured weights."""
        cluster = np.random.choice(
            self.config.histogram_clusters,
            p=self.config.histogram_weights
        )
        
        # Add random variation
        base_hist = self.histogram_templates[cluster].copy()
        noise = np.random.randn(len(base_hist)) * 0.01
        hist = base_hist + noise
        hist = np.maximum(hist, 0)
        hist = hist / hist.sum()
        
        return hist


class NormalToPathologicalPipeline:
    """Main pipeline for synthesizing pathological images from normal cases."""
    
    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize components
        self.mask_generator = LesionMaskGenerator(config)
        self.histogram_controller = HistogramController(config) if config.use_histogram_control else None
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Setup output directory
        Path(config.output_path).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self) -> NeuralSynthDiffusion:
        """Load pre-trained NeuralSynth model."""
        model_config = NeuralSynthConfig(
            use_adaptive_noise=True,
            use_multi_scale=True,
            use_lesion_attention=True,
            num_timesteps=1000,
            ddim_steps=self.config.ddim_steps
        )
        
        model = NeuralSynthDiffusion(model_config).to(self.device)
        
        if Path(self.config.checkpoint_path).exists():
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded checkpoint from {self.config.checkpoint_path}")
        else:
            self.logger.warning("No checkpoint found, using random initialization")
        
        return model
    
    def load_normal_case(self, path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """Load a normal medical image."""
        path = Path(path)
        
        if path.suffix in ['.nii', '.nii.gz']:
            # NIfTI format
            img = nib.load(str(path))
            data = img.get_fdata()
            metadata = {'affine': img.affine, 'header': img.header}
        elif path.suffix in ['.npz']:
            # NumPy format
            loaded = np.load(str(path))
            data = loaded['data']
            metadata = {k: v for k, v in loaded.items() if k != 'data'}
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        return data, metadata
    
    def synthesize_lesion(self, normal_image: np.ndarray, 
                         mask: np.ndarray,
                         histogram: Optional[np.ndarray] = None) -> np.ndarray:
        """Synthesize lesion in normal image using NeuralSynth."""
        
        # Prepare inputs
        normal_tensor = torch.from_numpy(normal_image).unsqueeze(0).unsqueeze(0).float().to(self.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        if histogram is not None:
            histogram_tensor = torch.from_numpy(histogram).unsqueeze(0).float().to(self.device)
        else:
            histogram_tensor = None
        
        with torch.no_grad():
            # Key innovation: Preserve background completely (LeFusion approach)
            synthetic = self.model.sample(
                shape=normal_tensor.shape,
                lesion_mask=mask_tensor,
                background=normal_tensor,  # Preserve original background
                histogram=histogram_tensor if self.config.use_histogram_control else None,
                device=self.device
            )
        
        # Convert back to numpy
        synthetic_np = synthetic.squeeze().cpu().numpy()
        
        # Ensure background is perfectly preserved
        if self.config.preserve_background:
            synthetic_np = synthetic_np * mask + normal_image * (1 - mask)
        
        return synthetic_np
    
    def process_single_case(self, normal_path: Union[str, Path]) -> List[Dict]:
        """Process a single normal case to generate synthetic pathological cases."""
        normal_path = Path(normal_path)
        self.logger.info(f"Processing {normal_path.name}")
        
        # Load normal case
        normal_image, metadata = self.load_normal_case(normal_path)
        
        results = []
        for i in range(self.config.num_synthetic_per_normal):
            # Generate lesion mask
            if np.random.rand() > 0.7:  # 30% spiculated for lung nodules
                mask = self.mask_generator.generate_spiculated_mask(normal_image.shape)
            else:
                mask = self.mask_generator.generate_ellipsoid_mask(normal_image.shape)
            
            # Sample histogram if using control
            histogram = self.histogram_controller.sample_histogram() if self.config.use_histogram_control else None
            
            # Synthesize lesion
            synthetic_image = self.synthesize_lesion(normal_image, mask, histogram)
            
            # Save results
            case_id = f"{normal_path.stem}_synthetic_{i}"
            output_path = Path(self.config.output_path) / f"{case_id}.npz"
            
            np.savez_compressed(
                output_path,
                image=synthetic_image,
                mask=mask,
                normal_source=str(normal_path),
                histogram=histogram if histogram is not None else [],
                **metadata
            )
            
            results.append({
                'case_id': case_id,
                'output_path': str(output_path),
                'mask_volume': mask.sum(),
                'histogram_cluster': np.argmax(histogram) if histogram is not None else -1
            })
            
            self.logger.info(f"  Generated {case_id}")
        
        return results
    
    def run_batch_synthesis(self, normal_dir: Optional[str] = None) -> Dict:
        """Run synthesis on all normal cases."""
        if normal_dir is None:
            normal_dir = self.config.normal_data_path
        
        normal_dir = Path(normal_dir)
        normal_cases = list(normal_dir.glob("*.nii*")) + list(normal_dir.glob("*.npz"))
        
        self.logger.info(f"Found {len(normal_cases)} normal cases")
        
        all_results = []
        for normal_path in tqdm(normal_cases, desc="Synthesizing"):
            try:
                results = self.process_single_case(normal_path)
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Failed to process {normal_path}: {e}")
                continue
        
        # Save summary
        summary = {
            'total_normal_cases': len(normal_cases),
            'total_synthetic_generated': len(all_results),
            'config': self.config.__dict__,
            'results': all_results
        }
        
        summary_path = Path(self.config.output_path) / "synthesis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Synthesis complete! Generated {len(all_results)} synthetic cases")
        self.logger.info(f"Summary saved to {summary_path}")
        
        return summary


def main():
    """Main entry point for synthesis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuralSynth Normal-to-Pathological Synthesis")
    parser.add_argument("--dataset", type=str, default="lidc", choices=["lidc", "emidec"],
                       help="Dataset to use")
    parser.add_argument("--normal_dir", type=str, default=None,
                       help="Directory containing normal cases")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for synthetic data")
    parser.add_argument("--num_synthetic", type=int, default=3,
                       help="Number of synthetic cases per normal")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint path")
    parser.add_argument("--no_histogram", action="store_true",
                       help="Disable histogram control")
    
    args = parser.parse_args()
    
    # Configure based on dataset
    if args.dataset == "lidc":
        config = SynthesisConfig(
            normal_data_path=args.normal_dir or "/Users/skb/Documents/LeFusion/data/LIDC/normal",
            output_path=args.output_dir or "/Users/skb/Documents/LeFusion/data/LIDC/synthetic",
            num_synthetic_per_normal=args.num_synthetic,
            use_histogram_control=not args.no_histogram,
            multi_class=False
        )
    else:  # emidec
        config = SynthesisConfig(
            normal_data_path=args.normal_dir or "/Users/skb/Documents/LeFusion/data/EMIDEC/normal",
            output_path=args.output_dir or "/Users/skb/Documents/LeFusion/data/EMIDEC/synthetic",
            num_synthetic_per_normal=args.num_synthetic,
            use_histogram_control=False,  # EMIDEC doesn't need histogram
            multi_class=True,
            num_classes=2
        )
    
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    
    # Run pipeline
    pipeline = NormalToPathologicalPipeline(config)
    results = pipeline.run_batch_synthesis()
    
    print(f"\nSynthesis Complete!")
    print(f"Generated {results['total_synthetic_generated']} synthetic cases from {results['total_normal_cases']} normal cases")


if __name__ == "__main__":
    main()