#!/bin/bash

# NeuralSynth Conda Environment Setup Script
# This script creates a clean conda environment with compatible versions

echo "=========================================="
echo "NeuralSynth Conda Environment Setup"
echo "=========================================="

# Environment name
ENV_NAME="neuralsynth"

# Remove existing environment if it exists
echo "Checking for existing environment..."
conda env list | grep $ENV_NAME > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Environment '$ENV_NAME' exists. Removing..."
    conda deactivate 2>/dev/null
    conda env remove -n $ENV_NAME -y
fi

# Create new environment with Python 3.10
echo "Creating new conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate the environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch with CUDA 11.8 (adjust if needed)
echo "Installing PyTorch with CUDA support..."
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install essential packages via conda (more stable)
echo "Installing essential packages via conda..."
conda install -c conda-forge numpy scipy pandas matplotlib tqdm -y

# Install medical imaging packages
echo "Installing medical imaging packages..."
pip install --no-cache-dir \
    monai==1.2.0 \
    nibabel==5.1.0 \
    SimpleITK==2.2.1 \
    scikit-image==0.21.0

# Install diffusion model packages with compatible versions
echo "Installing diffusion model packages..."
pip install --no-cache-dir \
    diffusers==0.21.4 \
    transformers==4.30.0 \
    accelerate==0.20.3 \
    einops==0.6.1 \
    omegaconf==2.3.0

# Install additional required packages
echo "Installing additional packages..."
pip install --no-cache-dir \
    opencv-python==4.8.0.74 \
    tensorboard==2.13.0 \
    Pillow==10.0.0 \
    h5py==3.9.0 \
    pyyaml==6.0.1

# Optional: Install development tools
echo "Installing development tools (optional)..."
pip install --no-cache-dir \
    jupyter \
    ipykernel \
    pytest \
    black \
    flake8

# Create kernel for Jupyter (optional)
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python (neuralsynth)"

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate this environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify the installation, run:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\")'"
echo ""
echo "To start training, run:"
echo "  cd synthetic_training"
echo "  python train_lidc.py --help"
echo "=========================================="