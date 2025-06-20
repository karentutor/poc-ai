#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook)"

echo "🔄 Deactivating current environment..."
conda deactivate

echo "🗑️  Removing existing XENEX environment..."
rm -rf ~/miniconda3/envs/XENEX

echo "🔧 Creating new XENEX environment from environment.yml..."
conda env create -f environment.yml

echo "✅ Activating XENEX environment..."
conda activate XENEX

echo "📦 Installing specific versions of Hugging Face Hub and Sentence Transformers..."
# Uninstall any existing versions first to be safe
pip uninstall huggingface-hub sentence-transformers -y || echo "Pip packages not found, proceeding."

# Install huggingface-hub and verify version
pip install huggingface-hub==0.8.1 --force-reinstall
python -c "import huggingface_hub; print('huggingface_hub version after initial install:', huggingface_hub.__version__)"

# Install sentence-transformers and verify huggingface-hub version again
pip install sentence-transformers==2.2.2 --force-reinstall
python -c "import huggingface_hub; print('huggingface_hub version after sentence-transformers:', huggingface_hub.__version__)"

# Ensure pytesseract is installed
pip install pytesseract --force-reinstall

# FINAL: Always force correct version of huggingface-hub
pip install huggingface-hub==0.8.1 --force-reinstall
python -c "import huggingface_hub; print('huggingface_hub version after final force:', huggingface_hub.__version__)"

echo "🔍 Verifying installation..."
python -c "import fitz; import PIL; import numpy; import cv2; import pytesseract; import rapidfuzz; from tqdm import tqdm; from sentence_transformers import SentenceTransformer; import huggingface_hub; print(f'huggingface_hub version: {huggingface_hub.__version__}'); print(f'pytesseract version: {pytesseract.__version__}'); print('✅ All critical dependencies successfully installed!')"

echo """
🎉 Environment recreation complete! The XENEX environment is now active.

To activate this environment in the future, run:
    conda activate XENEX
"""

# Do NOT reinstall numpy or faiss-gpu, use the versions from environment.yml

# FINAL: Force correct versions of transformers and huggingface-hub for compatibility
pip install --force-reinstall transformers==4.20.1
pip install --force-reinstall huggingface-hub==0.8.1

# Verify all final versions
echo "🔍 Final version verification:"
python -c "import numpy; print(f'numpy version: {numpy.__version__}')"
python -c "import transformers; print(f'transformers version: {transformers.__version__}')"
python -c "import huggingface_hub; print(f'huggingface_hub version: {huggingface_hub.__version__}')"

# Create a test script for faiss
echo "🔍 Testing FAISS import and basic functionality..."
cat > test_faiss.py << EOL
import numpy as np
import faiss

try:
    print(f"FAISS version: {faiss.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Create a small test index
    d = 64                           # dimension
    nb = 100                         # database size
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    
    index = faiss.IndexFlatL2(d)     # build the index
    print(f"Index trained: {index.is_trained}")
    index.add(xb)                    # add vectors to the index
    print(f"Index size: {index.ntotal}")
    
    k = 4                            # we want to see 4 nearest neighbors
    nq = 1                           # let's query 1 vector
    xq = np.random.random((nq, d)).astype('float32')
    
    distances, indices = index.search(xq, k)  # search
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
    
    print("✅ FAISS import and test successful!")
except Exception as e:
    print(f"❌ FAISS test failed: {str(e)}")
EOL

python test_faiss.py 