#!/bin/bash
# Default model if not provided
MODEL=${1:-"sentence-transformers/all-mpnet-base-v2"}
echo "Using model: $MODEL"
python 02_build_index.py widget_catalog.csv --out widget --model "$MODEL"