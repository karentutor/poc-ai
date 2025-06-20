#!/usr/bin/env python
# 02_build_index.py
import pandas as pd, faiss, pickle, argparse, torch, json
from sentence_transformers import SentenceTransformer

# Available models for indexing
AVAILABLE_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "intfloat": "intfloat/multilingual-e5-large",
    "baai": "BAAI/bge-large-en-v1.5",
    "e5": "intfloat/e5-large-v2"
}

# Available similarity metrics and their corresponding FAISS index types
SIMILARITY_METRICS = {
    "cosine": "IP",  # Inner Product (dot product) for cosine similarity
    "euclidean": "L2",  # L2 distance
    "dot": "IP",  # Inner Product for dot product
    "cosine_sklearn": "IP"  # Inner Product for sklearn's cosine similarity
}

def build(csv_path, out="widget", model="minilm", use_gpu=False, similarity_metric="cosine"):
    # Check GPU availability
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for indexing")
    
    # Load the model
    model_path = AVAILABLE_MODELS.get(model, model)  # Allow custom model paths
    print(f"Loading model: {model_path}")
    embedder = SentenceTransformer(model_path, device=device)
    
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")
    
    df = df[df["context"].notna() & (df["context"].str.strip() != "")]
    print(f"Rows after filtering empty contexts: {len(df)}")
    
    if len(df) == 0:
        raise ValueError("No valid contexts found in the CSV file")
    
    texts = df["context"].tolist()
    names = df["widgetName"].tolist()
    print(f"Number of texts to encode: {len(texts)}")
    
    vecs = embedder.encode(texts, convert_to_numpy=True).astype("float32")
    print(f"Shape of encoded vectors: {vecs.shape}")
    
    # Normalize vectors for cosine similarity and dot product
    if similarity_metric in ["cosine", "dot", "cosine_sklearn"]:
        faiss.normalize_L2(vecs)
    
    # Create appropriate index type based on similarity metric
    index_type = SIMILARITY_METRICS[similarity_metric]
    if index_type == "L2":
        index = faiss.IndexFlatL2(vecs.shape[1])
    else:  # IP (Inner Product)
        index = faiss.IndexFlatIP(vecs.shape[1])
    
    index.add(vecs)
    
    # Save index and metadata
    faiss.write_index(index, f"{out}.faiss")
    pickle.dump(names, open(f"{out}_names.pkl", "wb"))
    pickle.dump(texts, open(f"{out}_ctx.pkl", "wb"))
    
    # Save configuration
    config = {
        "model": model_path,
        "similarity_metric": similarity_metric,
        "index_type": index_type,
        "vector_dimension": vecs.shape[1],
        "num_vectors": len(vecs)
    }
    with open(f"{out}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅  {len(names)} widgets → {out}.faiss")
    print(f"Using {similarity_metric} similarity with {index_type} index type")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")             # <— rename here
    p.add_argument("--out", default="widget")
    p.add_argument("--model", default="minilm",
                  help="Model to use for indexing. Can be one of the predefined models (minilm, mpnet, intfloat, baai, e5) or a path to a custom model.")
    p.add_argument("--use-gpu", action="store_true", help="Use GPU for indexing")
    p.add_argument("--similarity-metric", default="cosine", choices=list(SIMILARITY_METRICS.keys()),
                  help="Similarity metric to use (cosine, euclidean, dot, cosine_sklearn)")
    args = p.parse_args()
    build(**vars(args))   