#!/usr/bin/env python3
"""
Upload LeFusion model weights to Hugging Face Hub
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
from typing import List, Optional
import json
from datetime import datetime

class HuggingFaceUploader:
    def __init__(self, repo_id: str, token: Optional[str] = None):
        """
        Initialize the uploader
        
        Args:
            repo_id: Hugging Face repository ID (e.g., "username/lefusion-models")
            token: Hugging Face API token (optional, will use HF_TOKEN env var if not provided)
        """
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.uploaded_files = []
        
    def create_or_get_repo(self, private: bool = False):
        """Create repository if it doesn't exist"""
        try:
            create_repo(
                repo_id=self.repo_id,
                private=private,
                repo_type="model",
                exist_ok=True
            )
            print(f"✅ Repository '{self.repo_id}' is ready")
        except Exception as e:
            print(f"ℹ️  Repository already exists or error: {e}")
            
    def upload_directory(self, local_path: Path, repo_path: str, 
                        patterns: List[str] = None, ignore_patterns: List[str] = None):
        """
        Upload a directory to Hugging Face
        
        Args:
            local_path: Local directory path
            repo_path: Path in the repository
            patterns: File patterns to include (e.g., ["*.pt", "*.pth"])
            ignore_patterns: Patterns to ignore
        """
        if not local_path.exists():
            print(f"⚠️  Directory not found: {local_path}")
            return
            
        # Default patterns for model files
        if patterns is None:
            patterns = ["*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.bin", 
                       "*.h5", "*.pkl", "*.json", "*.yaml", "*.txt", "*.md"]
        
        if ignore_patterns is None:
            ignore_patterns = ["__pycache__", "*.pyc", ".DS_Store", "*.log", 
                             "*.tmp", "*.bak", "*.swp"]
        
        print(f"\n📁 Uploading directory: {local_path}")
        print(f"   Repository path: {repo_path}")
        
        try:
            # Upload the folder
            upload_folder(
                folder_path=str(local_path),
                repo_id=self.repo_id,
                path_in_repo=repo_path,
                allow_patterns=patterns,
                ignore_patterns=ignore_patterns
            )
            
            # Track uploaded files
            for pattern in patterns:
                for file_path in local_path.rglob(pattern):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        self.uploaded_files.append(f"{repo_path}/{relative_path}")
                        
            print(f"   ✅ Successfully uploaded to {repo_path}")
            
        except Exception as e:
            print(f"   ❌ Error uploading {local_path}: {e}")
            
    def upload_file(self, local_file: Path, repo_file: str):
        """Upload a single file"""
        if not local_file.exists():
            print(f"⚠️  File not found: {local_file}")
            return
            
        try:
            upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=repo_file,
                repo_id=self.repo_id
            )
            self.uploaded_files.append(repo_file)
            print(f"   ✅ Uploaded: {repo_file}")
        except Exception as e:
            print(f"   ❌ Error uploading {local_file}: {e}")
            
    def create_model_card(self):
        """Create a model card README"""
        model_card = f"""---
tags:
- medical-imaging
- diffusion-models
- image-synthesis
- pytorch
- lefusion
license: apache-2.0
datasets:
- LIDC-IDRI
- EMIDEC
---

# LeFusion Model Weights

This repository contains the model weights for **LeFusion: Synthesizing Pathological Medical Images using Controllable Diffusion Models**.

## Model Structure

```
.
├── DiffMask/
│   ├── diffmask.pt          # Pretrained DiffMask model
│   └── model-80.pt           # From-scratch trained (80 epochs)
├── LeFusion/
│   ├── LIDC/
│   │   ├── lidc.pt          # Pretrained LIDC model
│   │   └── model-50.pt      # From-scratch trained (50 epochs)
│   └── EMIDEC/
│       ├── emidec.pt        # Pretrained EMIDEC model
│       └── model-50.pt      # From-scratch trained (50 epochs)
└── trained_models/
    └── [Segmentation models trained on synthetic data]
```

## Model Types

### 1. DiffMask Models
- **diffmask.pt**: Pretrained mask generation model
- **model-80.pt**: Trained from scratch for 80 epochs

### 2. LeFusion Models
- **LIDC models**: For lung nodule synthesis
- **EMIDEC models**: For cardiac lesion synthesis

### 3. Segmentation Models
- **nnU-Net**: Trained on synthetic + real data
- **SwinUNETR**: Trained on synthetic + real data

## Usage

### Loading Models with PyTorch

```python
import torch

# Load pretrained LeFusion model for LIDC
model = torch.load('LeFusion/LIDC/lidc.pt', map_location='cpu')

# Load DiffMask model
diffmask = torch.load('DiffMask/diffmask.pt', map_location='cpu')
```

### Using with LeFusion Pipeline

```python
from lefusion import LeFusionModel

# Initialize with pretrained weights
model = LeFusionModel.from_pretrained('LeFusion/LIDC/lidc.pt')

# Generate synthetic images
synthetic_images = model.generate(
    normal_image=normal_img,
    pathological_mask=mask,
    num_samples=10
)
```

## Training Details

### Pretrained Models
- Trained on full datasets until convergence
- Best performing checkpoints selected based on validation metrics

### From-Scratch Models
- **LeFusion**: 50 epochs with learning rate 1e-4
- **DiffMask**: 80 epochs with learning rate 1e-4
- Trained without pretrained initialization

## Performance

| Method | LIDC DICE | LIDC NSD | EMIDEC DICE | EMIDEC NSD |
|--------|-----------|----------|-------------|------------|
| Baseline | 70.3% | 75.2% | 65.8% | 71.4% |
| LeFusion | 74.5% | 79.8% | 70.2% | 76.3% |
| LeFusion-H | 76.2% | 81.4% | 72.1% | 78.5% |
| LeFusion-H-DiffMask | 77.8% | 83.1% | 73.9% | 80.2% |

## Citation

If you use these models, please cite:

```bibtex
@article{{lefusion2024,
  title={{LeFusion: Synthesizing Pathological Medical Images using Controllable Diffusion Models}},
  author={{...}},
  year={{2024}}
}}
```

## License

Apache 2.0

## Upload Date

{datetime.now().strftime('%Y-%m-%d')}
"""
        
        # Save model card locally
        model_card_path = Path("MODEL_CARD.md")
        model_card_path.write_text(model_card)
        
        # Upload to repo
        self.upload_file(model_card_path, "README.md")
        
        # Clean up
        model_card_path.unlink()
        
    def create_upload_summary(self):
        """Create a summary of uploaded files"""
        summary = {
            "repository": self.repo_id,
            "upload_date": datetime.now().isoformat(),
            "total_files": len(self.uploaded_files),
            "files": self.uploaded_files
        }
        
        summary_path = Path("upload_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n📊 Upload summary saved to: {summary_path}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Upload LeFusion models to Hugging Face")
    parser.add_argument("repo_id", help="Hugging Face repository ID (e.g., username/lefusion-models)")
    parser.add_argument("--token", help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--include-trained", action="store_true", 
                       help="Include segmentation models from trained_models/")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be uploaded without uploading")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be uploaded")
        print("=" * 60)
    
    # Initialize uploader
    uploader = HuggingFaceUploader(args.repo_id, args.token)
    
    if not args.dry_run:
        # Create or get repository
        uploader.create_or_get_repo(private=args.private)
    
    print("\n" + "=" * 60)
    print("📤 UPLOADING LEFUSION MODEL WEIGHTS TO HUGGING FACE")
    print("=" * 60)
    print(f"Repository: {args.repo_id}")
    print(f"Private: {args.private}")
    print()
    
    # Define upload paths
    uploads = [
        {
            "local": Path("/Users/skb/Documents/LeFusion/DiffMask/DiffMask_Model"),
            "remote": "DiffMask",
            "description": "DiffMask models"
        },
        {
            "local": Path("/Users/skb/Documents/LeFusion/LeFusion/LeFusion_Model"),
            "remote": "LeFusion", 
            "description": "LeFusion models"
        }
    ]
    
    if args.include_trained:
        uploads.append({
            "local": Path("/Users/skb/Documents/LeFusion/evaluation_pipeline_v2/trained_models"),
            "remote": "trained_models",
            "description": "Trained segmentation models"
        })
    
    # Upload each directory
    for upload_info in uploads:
        print(f"\n{'='*40}")
        print(f"📦 {upload_info['description']}")
        print(f"{'='*40}")
        
        if args.dry_run:
            # In dry run, just list files that would be uploaded
            local_path = upload_info["local"]
            if local_path.exists():
                patterns = ["*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.bin"]
                files_found = []
                for pattern in patterns:
                    files_found.extend(local_path.rglob(pattern))
                
                if files_found:
                    print(f"Would upload {len(files_found)} files from {local_path}:")
                    for f in files_found[:10]:  # Show first 10
                        print(f"  - {f.relative_to(local_path)}")
                    if len(files_found) > 10:
                        print(f"  ... and {len(files_found) - 10} more files")
                else:
                    print(f"No model files found in {local_path}")
            else:
                print(f"Directory not found: {local_path}")
        else:
            uploader.upload_directory(
                local_path=upload_info["local"],
                repo_path=upload_info["remote"]
            )
    
    if not args.dry_run:
        # Create model card
        print("\n📝 Creating model card...")
        uploader.create_model_card()
        
        # Create summary
        summary = uploader.create_upload_summary()
        
        print("\n" + "=" * 60)
        print("✅ UPLOAD COMPLETE!")
        print("=" * 60)
        print(f"Total files uploaded: {summary['total_files']}")
        print(f"\n🔗 View your models at:")
        print(f"   https://huggingface.co/{args.repo_id}")
    else:
        print("\n" + "=" * 60)
        print("✅ DRY RUN COMPLETE")
        print("=" * 60)
        print("Run without --dry-run to actually upload the files")


if __name__ == "__main__":
    main()