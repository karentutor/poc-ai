#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment from environment.yml
echo "🔧 Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate the environment
echo "✅ Environment created. Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate XENEX

# Verify installation
echo "🔍 Verifying installation..."
python -c "import fitz; import PIL; import numpy; import cv2; import pytesseract; import rapidfuzz; from tqdm import tqdm; print('✅ All dependencies successfully installed!')"

echo """
🎉 Setup complete! To activate the environment, run:
    conda activate XENEX

""" 