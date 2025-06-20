#!/usr/bin/env python
import pickle, faiss, json, argparse
from sentence_transformers import SentenceTransformer
from extract_fields import extract                      # ← our step-3 extractor

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def load_index(prefix="forms"):
    return (faiss.read_index(f"{prefix}.faiss"),
            pickle.load(open(f"{prefix}_meta.pkl", "rb")))

def match(fields, prefix="forms", k=3):
    index, texts = load_index(prefix)
    out = {}
    for f in fields:
        if not f["label"]:
            continue
        q = model.encode([f["label"]], convert_to_numpy=True).astype("float32")
        D, I = index.search(q, k)
        out[f["name"]] = [texts[i] for i in I[0]]
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pdf")                 # ← takes the PDF, not a CSV
    p.add_argument("--idx", default="forms")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("-o", "--out")
    a = p.parse_args()

    matches = match(extract(a.pdf), a.idx, a.k)
    if a.out:
        json.dump(matches, open(a.out, "w"), indent=2, ensure_ascii=False)
        print(f"✅  wrote matches → {a.out}")
    else:
        print(json.dumps(matches, indent=2, ensure_ascii=False))
