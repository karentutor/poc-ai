#!/bin/bash
# Path to the finetuned model
MODEL=${1:-"./models/finetuned_minilm"}
echo "Using finetuned model: $MODEL"
python 02_build_index.py widget_catalog.csv --out widget --model "$MODEL" --similarity-metric cosine --use-gpu