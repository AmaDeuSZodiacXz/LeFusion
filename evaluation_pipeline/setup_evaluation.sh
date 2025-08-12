#!/bin/bash

# LeFusion Evaluation Pipeline Setup Script
# This script installs all required dependencies for the evaluation pipeline

echo "=========================================================="
echo "LEFUSION EVALUATION PIPELINE SETUP"
echo "Installing all required dependencies"
echo "=========================================================="

# Check if we're in the evaluation_pipeline directory
if [ ! -f "run_comprehensive_paper_evaluation.py" ]; then
    echo "Error: Please run this script from the evaluation_pipeline directory"
    echo "Current directory: $(pwd)"
    echo "Expected files: run_comprehensive_paper_evaluation.py"
    exit 1
fi

echo "✓ Running from correct directory: $(pwd)"

# Install core dependencies from requirements.txt
echo "📦 Installing core dependencies..."
pip install -r requirements.txt

# Install surface_distance library
echo "📦 Installing surface_distance library..."
cd DiffTumor/STEP3.SegmentationModel/external/surface-distance
pip install -e .
cd ../../../

# Install additional dependencies that might be missing
echo "📦 Installing additional dependencies..."
pip install torchvision
pip install tensorboard>=2.7.0
pip install omegaconf>=2.1.0
pip install hydra-core>=1.1.0
pip install blobfile>=0.13.0
pip install einops-exts>=0.0.4
pip install rotary-embedding-torch>=0.2.0

# Test surface_distance installation
echo "🧪 Testing surface_distance library..."
python -c 'from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance; print("✓ surface_distance library is working!")'

# Test other key dependencies
echo "🧪 Testing key dependencies..."
python -c 'import torch; print(f"✓ PyTorch version: {torch.__version__}")'
python -c 'import monai; print(f"✓ MONAI version: {monai.__version__}")'
python -c 'import nibabel; print("✓ Nibabel is working!")'
python -c 'import omegaconf; print("✓ OmegaConf is working!")'

echo "=========================================================="
echo "✅ SETUP COMPLETED SUCCESSFULLY!"
echo "=========================================================="
echo ""
echo "📋 Next steps:"
echo "1. Prepare your dataset in nnU-Net format"
echo "2. Run: ./run_paper_pipeline.sh"
echo "3. Or run specific methods: ./run_paper_pipeline.sh method lefusion_h"
echo ""
echo "📁 Directory structure created:"
echo "   paper_experiments/"
echo "   ├── synthetic/"
echo "   ├── training/"
echo "   └── evaluation_results/"
echo ""
echo "📊 Expected results format:"
echo "   comprehensive_paper_results.csv"
echo "==========================================================" 