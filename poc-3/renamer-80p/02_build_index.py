#!/usr/bin/env python
# 02_build_index.py
import pandas as pd, faiss, pickle, argparse
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL)

def build(csv_path, out="widget"):
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
    
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, f"{out}.faiss")
    pickle.dump(names, open(f"{out}_names.pkl", "wb"))
    pickle.dump(texts, open(f"{out}_ctx.pkl", "wb"))
    print(f"✅  {len(names)} widgets → {out}.faiss")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")             # <— rename here
    p.add_argument("--out", default="widget")
    args = p.parse_args()
    build(**vars(args))   