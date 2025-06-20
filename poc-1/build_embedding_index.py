
"""build_embedding_index.py
--------------------------------
Encode label context from widgets.json with SentenceTransformers
and build a FAISS (or hnswlib fallback) ANN index.

Usage
-----
$ pip install sentence-transformers faiss-cpu hnswlib joblib rich
$ python build_embedding_index.py widgets.json

Outputs
-------
- embeddings.npy        (N x 384 float32)
- ids.json              (maps row index ➜ field_name)
- faiss.index           (if FAISS available)
  OR labels.hnsw        (if hnswlib fallback)

Author: ChatGPT
Created: 2025-05-01
"""

import argparse, json, re, string, sys, numpy as np, joblib
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer

_PUNCT_TABLE = str.maketrans('', '', string.punctuation)
def norm(txt: str) -> str:
    txt = txt.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", txt).strip()

def feature(rec: dict) -> str:
    parts: List[str] = [
        rec.get("label_before", ""),
        rec.get("text_after", ""),
        rec.get("page_heading", ""),
        rec.get("tooltip", ""),
    ]
    return norm(" ".join(p for p in parts if p))

def main(json_path: Path):
    data = json.loads(json_path.read_text())
    texts = [feature(r) for r in data]
    ids   = [r["field_name"] for r in data]

    print(f"Loaded {len(texts)} samples.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = emb.astype('float32')
    np.save("embeddings.npy", emb)
    Path("ids.json").write_text(json.dumps(ids))
    print("Saved embeddings.npy & ids.json")

    # Try FAISS
    try:
        import faiss

        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)  # cosine because vectors normalized
        index.add(emb)
        faiss.write_index(index, "faiss.index")
        print("FAISS index written ➜ faiss.index")

    except ModuleNotFoundError:
        print("FAISS not installed, falling back to hnswlib")
        import hnswlib
        dim = emb.shape[1]
        p = hnswlib.Index(space='cosine', dim=dim)
        p.init_index(max_elements=len(emb), ef_construction=200, M=16)
        p.add_items(emb, ids=np.arange(len(emb)))
        p.save_index("labels.hnsw")
        print("hnswlib index written ➜ labels.hnsw")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path, help="widgets.json path")
    args = parser.parse_args()
    main(args.json_path)
