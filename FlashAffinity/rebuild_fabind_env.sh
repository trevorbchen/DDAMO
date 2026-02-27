#!/bin/bash
# Script to rebuild fabind_h100 environment with PyTorch 2.4 + CUDA 11.8

set -e

ENV_NAME="fabind_h100"
BACKUP_FILE="/tmp/fabind_h100_packages_backup.txt"

echo "=== Rebuilding $ENV_NAME environment with PyTorch 2.4 + CUDA 11.8 ==="

# Backup current environment
echo "1. Backing up current package list..."
conda run -n $ENV_NAME pip list > $BACKUP_FILE 2>/dev/null || echo "Environment doesn't exist or backup failed"

# Remove old environment
echo "2. Removing old environment (if exists)..."
conda env remove -n $ENV_NAME -y || echo "Environment didn't exist"

# Create new environment with Python 3.9 (compatible with PyTorch 2.4)
echo "3. Creating new conda environment with Python 3.9..."
conda create -n $ENV_NAME python=3.9 -y

# Activate and install PyTorch 2.4 with CUDA 11.8
echo "4. Installing PyTorch 2.4 + CUDA 11.8..."
conda run -n $ENV_NAME pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyG dependencies for PyTorch 2.4 + CUDA 11.8
echo "5. Installing PyTorch Geometric and sparse dependencies..."
conda run -n $ENV_NAME pip install torch-geometric
conda run -n $ENV_NAME pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

# Install FABind dependencies
echo "6. Installing FABind+ dependencies..."
conda run -n $ENV_NAME pip install \
    biopython \
    rdkit \
    fair-esm \
    lmdb \
    joblib \
    tqdm \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn

# Install additional ML packages
echo "7. Installing additional packages..."
conda run -n $ENV_NAME pip install \
    transformers \
    accelerate \
    einops \
    timm \
    wandb

# Verify installation
echo "8. Verifying installation..."
conda run -n $ENV_NAME python -c "
import torch
import torch_geometric
import torch_sparse
import torch_cluster
import esm
from Bio import SeqIO
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'PyTorch Geometric: {torch_geometric.__version__}')
print('All imports successful!')
"

echo ""
echo "=== Environment rebuild complete! ==="
echo "Package backup saved to: $BACKUP_FILE"
echo "Activate with: conda activate $ENV_NAME"
