#!/usr/bin/env python
"""
This script scans the current directory for all '.faiss' files 
and, for each one, generates a corresponding JSON configuration file
(e.g., for 'widget-baai.faiss', it creates 'widget-baai_config.json')
with a sample configuration. You can then modify these JSON files to
specify the correct SentenceTransformer model used to build that index.
"""

import glob
import os
import json

# Note: We're no longer importing faiss to make this script more broadly usable
# import faiss

def detect_dimension_from_filename(faiss_path):
    """Attempt to detect the dimension based on the filename patterns"""
    filename = os.path.basename(faiss_path).lower()
    
    # Use filename patterns to guess dimensions
    if "baai" in filename or "mpnet" in filename:
        return 768
    elif "intfloat" in filename or "e5" in filename:
        return 1024
    elif "minilm" in filename or "mini" in filename:
        return 384
    
    # If we can't determine from filename, suggest common dimensions
    return None

def suggest_model(dimension, faiss_path=""):
    """Suggest a possible model based on dimension"""
    common_models = {
        384: "sentence-transformers/all-MiniLM-L6-v2",
        768: [
            "sentence-transformers/all-mpnet-base-v2",  # or many BERT-based models
            "BAAI/bge-base-en-v1.5"  # BAAI base model with 768 dimensions
        ],
        1024: [
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "intfloat/e5-large-v2"  # intfloat model with 1024 dimensions
        ]
    }
    
    model_suggestion = common_models.get(dimension, "your-model-name-here")
    if isinstance(model_suggestion, list):
        if "baai" in os.path.basename(faiss_path).lower():
            return model_suggestion[1]  # Return BAAI model if "baai" is in filename
        elif "intfloat" in os.path.basename(faiss_path).lower() or "e5" in os.path.basename(faiss_path).lower():
            return model_suggestion[1] if dimension == 1024 else model_suggestion
        return f"Possible options: {', '.join(model_suggestion)}"
    return model_suggestion

def main():
    # Search for all .faiss files in the current directory
    faiss_files = glob.glob("*.faiss")
    
    if not faiss_files:
        print("No .faiss files found in the current directory.")
        return
    
    for faiss_file in faiss_files:
        prefix = faiss_file[:-6]  # Remove the '.faiss' extension
        config_file = f"{prefix}_config.json"
        
        if os.path.exists(config_file):
            print(f"Config file already exists: {config_file}")
        else:
            # Try to detect the dimension from filename patterns
            dimension = detect_dimension_from_filename(faiss_file)
            suggested_model = suggest_model(dimension, faiss_file) if dimension else "your-model-name-here"
            
            # Create a configuration with detected/suggested settings
            sample_config = {
                "model": suggested_model,
                "dimension": dimension if dimension else "unknown - please specify (e.g., 384, 768, or 1024)"
            }
            with open(config_file, "w") as f:
                json.dump(sample_config, f, indent=4)
            print(f"Created config file: {config_file} (Detected dimension: {dimension or 'unknown - please specify manually'})")
            if dimension is None:
                print(f"  Note: Could not detect dimension for {faiss_file}. Please edit the config file manually.")

if __name__ == "__main__":
    main() 