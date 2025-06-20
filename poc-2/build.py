#!/usr/bin/env python
# build_index.py
# python "00 build.py" forms.csv --out forms

import pandas as pd, pickle, faiss, argparse
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)          # ≈ 80 MB

def build(csv_path: str, out_prefix: str = "forms"):
    df = pd.read_csv(csv_path)                   # needs textId, shortDesc, longDesc
    corpus = [f"{r.textId}: {r.shortDesc}. {r.longDesc}"
              for r in df.itertuples(index=False)]

    vecs = model.encode(corpus,
                        convert_to_numpy=True, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])     # cosine / inner-product
    index.add(vecs)

    faiss.write_index(index, f"{out_prefix}.faiss")
    pickle.dump(corpus, open(f"{out_prefix}_meta.pkl", "wb"))
    print(f"✅  {len(corpus)} entries → {out_prefix}.faiss")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--out", default="forms")
    args = p.parse_args()
    build(args.csv, args.out)
