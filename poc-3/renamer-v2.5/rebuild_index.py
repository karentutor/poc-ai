#!/usr/bin/env python
import os
import pandas as pd
import torch
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import json

def build_index(model_path, csv_path, out="widget", use_gpu=False):
    # Check GPU availability
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for indexing")
    
    # Load the model
    print(f"Loading model from {model_path}")
    embedder = SentenceTransformer(model_path, device=device)
    
    # Read and filter the CSV
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    
    df = df[df["context"].notna() & (df["context"].str.strip() != "")]
    print(f"Rows after filtering empty contexts: {len(df)}")
    
    if len(df) == 0:
        raise ValueError("No valid contexts found in the CSV file")
    
    # Prepare data
    texts = df["context"].tolist()
    names = df["widgetName"].tolist()
    print(f"Number of texts to encode: {len(texts)}")
    
    # Create embeddings
    print("Creating embeddings...")
    vecs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    print(f"Shape of encoded vectors: {vecs.shape}")
    
    # Create and save index
    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    
    # Save files
    print(f"Saving index to {out}.faiss")
    faiss.write_index(index, f"{out}.faiss")
    
    print(f"Saving names to {out}_names.pkl")
    pickle.dump(names, open(f"{out}_names.pkl", "wb"))
    
    print(f"Saving contexts to {out}_ctx.pkl")
    pickle.dump(texts, open(f"{out}_ctx.pkl", "wb"))
    
    # Save configuration
    config = {
        "model_path": model_path,
        "csv_file": csv_path,
        "num_entries": len(names),
        "embedding_dimension": vecs.shape[1],
        "device_used": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    }
    
    config_file = f"{out}_config.json"
    print(f"Saving configuration to {config_file}")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅  {len(names)} widgets → {out}.faiss")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Path to the CSV file")
    ap.add_argument("--model", required=True, help="Path to the model directory")
    ap.add_argument("--out", default="widget", help="Prefix for output files")
    ap.add_argument("--gpu", action="store_true", help="Use GPU for indexing")
    args = ap.parse_args()
    
    build_index(args.model, args.csv_path, args.out, args.gpu) 