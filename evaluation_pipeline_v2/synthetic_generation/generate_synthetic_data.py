#!/usr/bin/env python3
"""
Synthetic Data Generation Pipeline for LeFusion Paper Evaluation
Supports: LeFusion, LeFusion-H, LeFusion-H+DiffMask
With resume capability and organized folder structure
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import time
import shutil

class SyntheticDataGenerator:
    def __init__(self, config_path="configs/experiment_config.yaml"):
        """Initialize synthetic data generator with config"""
        # Get the evaluation_pipeline_v2 directory as base
        script_dir = Path(__file__).parent  # synthetic_generation/
        self.base_dir = script_dir.parent   # evaluation_pipeline_v2/
        
        # Load config with robust path resolution
        candidate_path = Path(config_path)
        if not candidate_path.is_absolute():
            candidate_path = (self.base_dir / candidate_path).resolve()
        if not candidate_path.exists():
            # Fallback to default inside evaluation_pipeline_v2/configs/
            fallback_path = (self.base_dir / "configs" / "experiment_config.yaml").resolve()
            candidate_path = fallback_path
        with open(candidate_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set output directory relative to evaluation_pipeline_v2
        self.output_dir = self.base_dir / "synthetic_data"
        self.checkpoint_file = self.output_dir / "generation_checkpoint.json"
        
        print(f"📁 Base directory: {self.base_dir}")
        print(f"🧩 Config file: {candidate_path}")
        print(f"📁 Output directory: {self.output_dir}")
        
    def load_checkpoint(self):
        """Load checkpoint for resume capability"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {}
        
    def save_checkpoint(self, checkpoint):
        """Save checkpoint for resume capability"""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
    def get_output_path(self, dataset, method, model_type, data_type):
        """Get organized output path for synthetic data"""
        # Structure: synthetic_data/dataset/model_type/method/data_type/
        path = self.output_dir / dataset / model_type / method / data_type
        return path
        
    def _resolve_path(self, maybe_rel_path: str) -> str:
        """Resolve a path relative to evaluation_pipeline_v2 directory and return absolute string"""
        p = Path(maybe_rel_path)
        if not p.is_absolute():
            p = (self.base_dir / p).resolve()
        return str(p)

    def _check_lidc_inputs(self, dataset: str) -> bool:
        """Preflight checks for LIDC inputs"""
        normal_dir = self._resolve_path(self.config['datasets'][dataset]['normal_dir'])
        test_file = self._resolve_path(self.config['datasets'][dataset]['test_file'])
        if not Path(normal_dir).exists():
            print(f"❌ Normal image dir not found: {normal_dir}")
            return False
        nii_list = list(Path(normal_dir).rglob('*.nii.gz'))
        print(f"🔎 Found {len(nii_list)} .nii.gz under {normal_dir}")
        if len(nii_list) == 0:
            print("❌ No .nii.gz files found. Please verify dataset path.")
            return False
        if not Path(test_file).exists():
            print(f"❌ test.txt not found: {test_file}")
            return False
        return True

    def _select_lidc_root_dir(self) -> str:
        """Pick LIDC root_dir. Prefer config normal_dir if it contains nii.gz, else fallback to Pathological/Image."""
        normal_dir = Path(self._resolve_path(self.config['datasets']['lidc']['normal_dir']))
        pathological_base = Path(self._resolve_path(self.config['datasets']['lidc']['pathological_dir']))
        pathological_image_dir = pathological_base / 'Image'
        normal_count = len(list(normal_dir.rglob('*.nii.gz'))) if normal_dir.exists() else 0
        if normal_count > 0:
            print(f"✅ Using LIDC normal_dir: {normal_dir} ({normal_count} files)")
            return str(normal_dir)
        # fallback
        fallback_count = len(list(pathological_image_dir.rglob('*.nii.gz'))) if pathological_image_dir.exists() else 0
        print(f"⚠️ No files in normal_dir. Fallback to pathological Image dir: {pathological_image_dir} ({fallback_count} files)")
        return str(pathological_image_dir)

    def _run_streaming(self, cmd: list) -> int:
        """Run a command and stream its output to the terminal in real-time.
        Returns the process return code.
        """
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
            process.stdout.close()
            return process.wait()
        except Exception as e:
            print(f"❌ Error launching process: {e}")
            return 1

    def _prepare_diffmask_input(self, src_img_dir: Path, src_lbl_dir: Path, staging_dir: Path) -> Path:
        """Create a staging directory with 'Image' and 'Mask' subfolders (link/copy) for DiffMask.
        Ensures staged image names use '_Vol_' and masks use '_Mask_' (as DiffMask expects).
        Supports nested Image_*/Mask_* folders and creates an empty test.txt for no filtering.
        """
        # Clean up existing staging directory to prevent duplicates
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        
        image_dir = staging_dir / 'Image'
        mask_dir = staging_dir / 'Mask'
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for img_path in Path(src_img_dir).rglob('*.nii.gz'):
            base = img_path.name
            # Rename staged image to use _Vol_ instead of _CVol_
            if '_CVol_' in base:
                staged_img_name = base.replace('_CVol_', '_Vol_')
            else:
                staged_img_name = base
            img_dst = image_dir / staged_img_name
            if not img_dst.exists():
                try:
                    os.link(img_path, img_dst)
                except OSError:
                    shutil.copy2(img_path, img_dst)

            # Determine corresponding Mask_X subdir if present
            mask_subdir = None
            parent_name = img_path.parent.name
            if parent_name.startswith('Image_'):
                mask_subdir = f"Mask_{parent_name.split('_', 1)[1]}"

            # Expected mask file name from staged image: _Vol_ -> _Mask_
            expected_mask_name = staged_img_name.replace('_Vol_', '_Mask_')

            # Locate source mask in labelsTr
            candidate_paths = []
            if mask_subdir:
                candidate_paths.append(Path(src_lbl_dir) / mask_subdir / expected_mask_name)
            candidate_paths.append(Path(src_lbl_dir) / expected_mask_name)
            src_mask_path = None
            for cand in candidate_paths:
                if cand.exists():
                    src_mask_path = cand
                    break
            if src_mask_path is None:
                hits = list(Path(src_lbl_dir).rglob(expected_mask_name))
                if hits:
                    src_mask_path = hits[0]

            if src_mask_path and src_mask_path.exists():
                mask_dst = mask_dir / expected_mask_name
                if not mask_dst.exists():
                    try:
                        os.link(src_mask_path, mask_dst)
                    except OSError:
                        shutil.copy2(src_mask_path, mask_dst)

        # Create empty test.txt to avoid filtering
        (staging_dir / 'test.txt').write_text('')
        return staging_dir

    def generate_lefusion(self, dataset, model_type, output_dir):
        """Generate synthetic data using LeFusion"""
        print(f"\n🎨 Generating LeFusion synthetic data")
        print(f"   Dataset: {dataset}")
        print(f"   Model Type: {model_type}")
        print(f"   Output: {output_dir}")
        
        # Get model path based on model type
        if model_type == "pretrained":
            model_path = self.config['model_weights']['pretrained']['lefusion'][dataset]
        else:
            model_path = self.config['model_weights']['from_scratch']['lefusion'][dataset]
        
        model_path = self._resolve_path(model_path)
        
        # Check if model exists
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        # Preflight checks and resolve dataset root
        if dataset == "lidc":
            root_dir = self._select_lidc_root_dir()
            test_file = self._resolve_path(self.config['datasets'][dataset]['test_file'])
            if not Path(test_file).exists():
                print(f"❌ test.txt not found: {test_file}")
                return False
        else:
            root_dir = self._resolve_path(self.config['datasets'][dataset]['normal_dir'])
        
        # Create output directories
        os.makedirs(output_dir / "imagesTr", exist_ok=True)
        os.makedirs(output_dir / "labelsTr", exist_ok=True)
        
        # Build command based on dataset (use absolute paths; hydra may chdir)
        py = sys.executable
        if dataset == "lidc":
            cmd = [
                py, "-u", str((self.base_dir.parent / "LeFusion" / "inference" / "inference.py").resolve()),
                f"data_type={dataset}",
                f"model_path={model_path}",
                f"dataset_root_dir={root_dir}",
                f"test_txt_dir={test_file}",
                f"target_img_path={str((output_dir / 'imagesTr').resolve())}",
                f"target_label_path={str((output_dir / 'labelsTr').resolve())}",
                "batch_size=1",
                "types=3",  # For LIDC
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=1",
                "cond_dim=16"
            ]
        elif dataset == "emidec":
            cmd = [
                py, "-u", str((self.base_dir.parent / "LeFusion" / "inference" / "inference.py").resolve()),
                f"data_type={dataset}",
                f"model_path={model_path}",
                f"dataset_root_dir={root_dir}",
                f"target_img_path={str((output_dir / 'imagesTr').resolve())}",
                f"target_label_path={str((output_dir / 'labelsTr').resolve())}",
                "batch_size=1",
                "types=1",  # For EMIDEC
                "diffusion_img_size=72",
                "diffusion_depth_size=10",
                "diffusion_num_channels=2",  # EMIDEC uses 2 channels
                "cond_dim=32"  # EMIDEC uses cond_dim=32
            ]
        else:
            print(f"❌ Unknown dataset: {dataset}")
            return False
        
        # Execute command (streaming)
        print(f"💻 Running: {' '.join(cmd)}")
        rc = self._run_streaming(cmd)
        if rc == 0:
            print(f"✅ LeFusion generation completed successfully")
            return True
        else:
            print(f"❌ LeFusion generation failed (exit {rc})")
            return False
        
    def generate_lefusion_h(self, dataset, model_type, output_dir):
        """Generate synthetic data using LeFusion-H (histogram conditioned).
        Note: inference.py always uses histogram conditioning via clusters; no extra flag needed.
        """
        print(f"\n🎨 Generating LeFusion-H synthetic data")
        print(f"   Dataset: {dataset}")
        print(f"   Model Type: {model_type}")
        print(f"   Output: {output_dir}")
        
        # LeFusion-H uses the same model but we keep the pipeline outputs separate
        if model_type == "pretrained":
            model_path = self.config['model_weights']['pretrained']['lefusion'][dataset]
        else:
            model_path = self.config['model_weights']['from_scratch']['lefusion'][dataset]
        model_path = self._resolve_path(model_path)
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        if dataset == "lidc":
            root_dir = self._select_lidc_root_dir()
            test_file = self._resolve_path(self.config['datasets'][dataset]['test_file'])
            if not Path(test_file).exists():
                print(f"❌ test.txt not found: {test_file}")
                return False
        else:
            root_dir = self._resolve_path(self.config['datasets'][dataset]['normal_dir'])
        
        os.makedirs(output_dir / "imagesTr", exist_ok=True)
        os.makedirs(output_dir / "labelsTr", exist_ok=True)
        
        py = sys.executable
        if dataset == "lidc":
            cmd = [
                py, "-u", str((self.base_dir.parent / "LeFusion" / "inference" / "inference.py").resolve()),
                f"data_type={dataset}",
                f"model_path={model_path}",
                f"dataset_root_dir={root_dir}",
                f"test_txt_dir={test_file}",
                f"target_img_path={str((output_dir / 'imagesTr').resolve())}",
                f"target_label_path={str((output_dir / 'labelsTr').resolve())}",
                "batch_size=1",
                "types=3",
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=1",
                "cond_dim=16"
            ]
        elif dataset == "emidec":
            cmd = [
                py, "-u", str((self.base_dir.parent / "LeFusion" / "inference" / "inference.py").resolve()),
                f"data_type={dataset}",
                f"model_path={model_path}",
                f"dataset_root_dir={root_dir}",
                f"target_img_path={str((output_dir / 'imagesTr').resolve())}",
                f"target_label_path={str((output_dir / 'labelsTr').resolve())}",
                "batch_size=1",
                "types=1",
                "diffusion_img_size=64",
                "diffusion_depth_size=32",
                "diffusion_num_channels=2",  # EMIDEC uses 2 channels
                "cond_dim=32"  # EMIDEC uses cond_dim=32
            ]
        else:
            print(f"❌ Unknown dataset: {dataset}")
            return False
        
        # Execute command (streaming)
        print(f"💻 Running: {' '.join(cmd)}")
        rc = self._run_streaming(cmd)
        if rc == 0:
            print(f"✅ LeFusion-H generation completed successfully")
            return True
        else:
            print(f"❌ LeFusion-H generation failed (exit {rc})")
            return False
            
    def generate_diffmask_enhancement(self, dataset, model_type, input_dir, output_dir):
        """Enhance synthetic data with DiffMask (generates masks). Copies images alongside.
        Expects input_dir with imagesTr/ and labelsTr/.
        """
        print(f"\n🎨 Enhancing with DiffMask")
        print(f"   Dataset: {dataset}")
        print(f"   Model Type: {model_type}")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        
        # Get DiffMask model path
        if model_type == "pretrained":
            model_path = self.config['model_weights']['pretrained']['diffmask'][dataset]
        else:
            model_path = self.config['model_weights']['from_scratch']['diffmask'][dataset]
        model_path = self._resolve_path(model_path)
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        src_img = Path(input_dir) / 'imagesTr'
        src_lbl = Path(input_dir) / 'labelsTr'
        # Prepare staging input in expected structure
        staging_dir = Path(output_dir).parent / '_diffmask_staging'
        staging_dir = self._prepare_diffmask_input(src_img, src_lbl, staging_dir)
        
        # Ensure output dirs exist and copy images over
        out_img = Path(output_dir) / 'imagesTr'
        out_lbl = Path(output_dir) / 'labelsTr'
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        
        # Copy all images recursively from Image_* subfolders
        print(f"📁 Copying images from {src_img} to {out_img}")
        copied_count = 0
        for img_path in src_img.rglob('*.nii.gz'):
            if img_path.is_file():
                dst = out_img / img_path.name
                if not dst.exists():
                    try:
                        os.link(img_path, dst)
                        copied_count += 1
                    except OSError:
                        shutil.copy2(img_path, dst)
                        copied_count += 1
        print(f"✅ Copied {copied_count} images to {out_img}")
        
        # Build DiffMask command using its Hydra config keys
        py = sys.executable
        lidc_test_file = str((staging_dir / 'test.txt').resolve())
        cmd = [
            py, "-u", str((self.base_dir.parent / "DiffMask" / "inference" / "inference.py").resolve()),
            f"model_path={model_path}",
            f"dataset_root_dir={str((staging_dir / 'Image').resolve())}",
            f"test_txt_path={lidc_test_file}",
            f"gen_mask_path={str(out_lbl.resolve())}/",
            "unet_num_channels=2",
            "out_dim=1"
        ]
        print(f"💻 Running: {' '.join(cmd)}")
        rc = self._run_streaming(cmd)
        if rc == 0:
            print(f"✅ DiffMask enhancement completed successfully")
            return True
        else:
            print(f"❌ DiffMask enhancement failed (exit {rc})")
            return False
            
    def generate_all(self, dataset="lidc", model_type="pretrained", methods=None, resume=False):
        """Generate all synthetic data for specified configuration"""
        
        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else {}
        
        # Default to all methods if not specified
        if methods is None:
            methods = ["lefusion", "lefusion_h", "lefusion_h_diffmask"]
            
        print(f"\n{'='*60}")
        print(f"SYNTHETIC DATA GENERATION PIPELINE")
        print(f"Dataset: {dataset}")
        print(f"Model Type: {model_type}")
        print(f"Methods: {methods}")
        print(f"Resume: {resume}")
        print(f"{'='*60}")
        
        results = {}
        
        for method in methods:
            # Check if already completed
            checkpoint_key = f"{dataset}_{model_type}_{method}"
            if resume and checkpoint.get(checkpoint_key, {}).get('completed', False):
                print(f"\n✅ Skipping {method} - already completed")
                continue
                
            print(f"\n{'='*50}")
            print(f"Processing: {method}")
            print(f"{'='*50}")
            
            start_time = time.time()
            
            if method == "lefusion":
                # Generate P+P' data
                output_dir = self.get_output_path(dataset, method, model_type, "P_P_prime")
                success = self.generate_lefusion(dataset, model_type, output_dir)
                
            elif method == "lefusion_h":
                # Generate P+P' and P+N' data
                output_dir_pp = self.get_output_path(dataset, method, model_type, "P_P_prime")
                success_pp = self.generate_lefusion_h(dataset, model_type, output_dir_pp)
                
                output_dir_pn = self.get_output_path(dataset, method, model_type, "P_N_prime")
                success_pn = self.generate_lefusion_h(dataset, model_type, output_dir_pn)
                
                success = success_pp and success_pn
                
            elif method == "lefusion_h_diffmask":
                # First generate LeFusion-H data
                temp_dir = self.get_output_path(dataset, "lefusion_h_temp", model_type, "temp")
                success_h = self.generate_lefusion_h(dataset, model_type, temp_dir)
                
                if success_h:
                    # Then enhance with DiffMask for different combinations
                    output_dir_pn = self.get_output_path(dataset, method, model_type, "P_N_prime")
                    success_pn = self.generate_diffmask_enhancement(dataset, model_type, temp_dir, output_dir_pn)
                    
                    output_dir_pn2 = self.get_output_path(dataset, method, model_type, "P_N_double_prime")
                    success_pn2 = self.generate_diffmask_enhancement(dataset, model_type, temp_dir, output_dir_pn2)
                    
                    output_dir_all = self.get_output_path(dataset, method, model_type, "P_P_prime_N_double_prime")
                    success_all = self.generate_diffmask_enhancement(dataset, model_type, temp_dir, output_dir_all)
                    
                    success = success_pn and success_pn2 and success_all
                else:
                    success = False
                    
            else:
                print(f"❌ Unknown method: {method}")
                success = False
                
            # Update checkpoint
            elapsed_time = time.time() - start_time
            checkpoint[checkpoint_key] = {
                'completed': success,
                'timestamp': datetime.now().isoformat(),
                'elapsed_time': elapsed_time
            }
            self.save_checkpoint(checkpoint)
            
            results[method] = success
            
            if success:
                print(f"✅ {method} completed in {elapsed_time:.2f} seconds")
            else:
                print(f"❌ {method} failed")
                
        # Print summary
        print(f"\n{'='*60}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*60}")
        for method, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"{method:30} {status}")
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for LeFusion paper evaluation")
    parser.add_argument("--dataset", choices=["lidc", "emidec", "all"], default="lidc",
                        help="Dataset to generate synthetic data for")
    parser.add_argument("--model-type", choices=["pretrained", "from_scratch", "all"], default="pretrained",
                        help="Model type to use")
    parser.add_argument("--methods", nargs="+", 
                        choices=["lefusion", "lefusion_h", "lefusion_h_diffmask", "all"],
                        default=["lefusion", "lefusion_h", "lefusion_h_diffmask"],
                        help="Methods to generate (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--config", default="configs/experiment_config.yaml",
                        help="Path to config file")
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(args.config)
    
    # Process datasets
    datasets = ["lidc", "emidec"] if args.dataset == "all" else [args.dataset]
    model_types = ["pretrained", "from_scratch"] if args.model_type == "all" else [args.model_type]
    
    # Process methods - expand 'all' to individual methods
    if "all" in args.methods:
        methods = ["lefusion", "lefusion_h", "lefusion_h_diffmask"]
    else:
        methods = args.methods
    
    for dataset in datasets:
        for model_type in model_types:
            generator.generate_all(
                dataset=dataset,
                model_type=model_type,
                methods=methods,
                resume=args.resume
            )

if __name__ == "__main__":
    main() 