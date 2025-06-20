dnf install -y python3-devel mesa-libGL-devel

#!/bin/bash
ENV_NAME="xenex"

echo "[*] Deleting old env if exists"
conda remove -n "$ENV_NAME" --all -y

echo "[*] Creating new env: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.11 numpy=1.26 -y
conda activate "$ENV_NAME"

conda install -c conda-forge numpy=1.24.4
conda install -c conda-forge pytorch

echo "[*] Installing pip packages"
pip install \
    # pymupdf \
    sentence-transformers \
    opencv-python-headless \
    pytesseract \
    rapidfuzz \
    tqdm
    
pip install fastapi
pip install uvicorn

pip install python-multipart
# pip install pymupdf
# pip install pymupdf==1.21.1
pip install pymupdf==1.22.3

pip install paddleocr==2.6.1.3
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR()"

pip install numpy==1.24.4 scipy==1.10.1    
pip install sentence-transformers
# pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install paddlepaddle==2.5.2 -f https://www.paddlepaddle.org.cn/whl/linux/mavl.html

conda install -y -c conda-forge \
        pytorch==2.1.0 torchvision==0.16.0 \
        faiss-cpu sentence-transformers transformers

conda install -c conda-forge faiss-cpu -y
conda install -c conda-forge numpy=1.24.4 -y
conda install -c conda-forge pytorch -y
conda install -c tqdm -y
conda install -c conda-forge opencv -y

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu
pip install numpy<2.0.0

echo "[*] Activating env"
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "[✓] Environment '$ENV_NAME' is ready."
conda activate xenex