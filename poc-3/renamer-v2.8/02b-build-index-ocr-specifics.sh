#!/bin/bash

# Available models
AVAILABLE_MODELS=(
    "minilm"    # sentence-transformers/all-MiniLM-L6-v2
    "mpnet"     # sentence-transformers/all-mpnet-base-v2
    "intfloat"  # intfloat/multilingual-e5-large
    "baai"      # BAAI/bge-large-en-v1.5
    "e5"        # intfloat/e5-large-v2
)


# Function to show help
show_help() {
    echo "Usage: $0 [model_name] [similarity_metric] [threshold] [--gpu]"
    echo
    echo "Available models:"
    echo "  minilm    - sentence-transformers/all-MiniLM-L6-v2 (default)"
    echo "  mpnet     - sentence-transformers/all-mpnet-base-v2"
    echo "  intfloat  - intfloat/multilingual-e5-large"
    echo "  baai      - BAAI/bge-large-en-v1.5"
    echo "  e5        - intfloat/e5-large-v2"
}

# Check if help is requested
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
    exit 0
fi

# Default model if not provided
MODEL=${1:-"minilm"}


# Check for GPU flag
GPU_FLAG=""
if [[ "$*" == *"--gpu"* ]]; then
    GPU_FLAG="--gpu"
fi

# Check if it's a predefined model
if [[ " ${AVAILABLE_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "Using predefined model: $MODEL"
else
    echo "Using custom model path: $MODEL"
fi

python ./scripts/02_build_index.py --catalog catalog_ocr_specifics.json --out widget_ocr_specifics --model "$MODEL" $GPU_FLAG