#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline PDF form‑field renamer leveraging a SentenceTransformer
embedding + ANN index (FAISS or hnswlib).

1. Build index once with build_embedding_index.py
2. Run this script to rename widgets in PDFs under PDF_DIR.

Requirements:
    pip install fitz sentence-transformers faiss-cpu hnswlib numpy rich
"""

import os, json, re
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

# ─── PATHS ───────────────────────────────────────────────────────────────
PDF_DIR      = "./pdfs/forms"
OUTPUT_DIR   = "./pdfs/forms/renamed"
INDEX_PATH   = "./faiss.index"      # or labels.hnsw if using hnswlib
IDS_PATH     = "./ids.json"
MODEL_NAME   = "all-MiniLM-L6-v2"
DIST_TH      = 0.15

# ─── LOAD ENCODER & INDEX ────────────────────────────────────────────────
print("[*] Loading encoder and ANN index …")
encoder = SentenceTransformer(MODEL_NAME)

try:
    import faiss
    index = faiss.read_index(INDEX_PATH)
    backend = "faiss"
    def query(v): return index.search(v, k=1)  # (D, I)
except Exception:
    import hnswlib
    backend = "hnswlib"
    index = hnswlib.Index(space='cosine', dim=384)
    index.load_index("labels.hnsw", max_elements=20000)
    def query(v): return index.knn_query(v, k=1)  # (I, D)

ids = json.load(open(IDS_PATH, encoding="utf-8"))
print(f"→ backend={backend}, vectors={len(ids)}")

# ─── HELPERS ─────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", text.strip().lower()).strip("_")

def text_blocks(page):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            txt = " ".join(sp.get("text","") for sp in l.get("spans", []))
            if txt.strip():
                out.append((fitz.Rect(*l["bbox"]), txt.strip()))
    return out

def find_label_after(wrect, blocks):
    lbl=after=""; best_l=best_a=float('inf')
    wy=(wrect.y0+wrect.y1)/2
    for rect, txt in blocks:
        cy=(rect.y0+rect.y1)/2
        if abs(cy-wy)<8:
            if rect.x1 < wrect.x0:
                d=wrect.x0-rect.x1
                if d<best_l: lbl, best_l = txt,d
            elif rect.x0 > wrect.x1:
                d=rect.x0-wrect.x1
                if d<best_a: after,best_a = txt,d
    return lbl, after

def doc_header(doc):
    if not doc.page_count: return ""
    p0=doc.load_page(0)
    hdr=[]
    for b in p0.get_text("dict")["blocks"]:
        if b.get("number")==0:
            for l in b.get("lines", []):
                hdr.append(" ".join(sp.get("text","") for sp in l["spans"]).strip())
    return " ".join(hdr[:2])

def page_heading(page):
    bl=text_blocks(page)
    return bl[0][1] if bl else ""

# ─── MAIN ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in sorted(os.listdir(PDF_DIR)):
        if not fname.lower().endswith('.pdf'): continue
        src=os.path.join(PDF_DIR,fname)
        dst=os.path.join(OUTPUT_DIR,fname.replace('.pdf','_renamed.pdf'))
        print(f"[+] {fname}")
        doc=fitz.open(src)
        hdr=doc_header(doc)
        for p in range(doc.page_count):
            page=doc.load_page(p)
            heading=page_heading(page)
            blocks=text_blocks(page)
            for w in page.widgets():
                lbl,aft = find_label_after(w.rect, blocks)
                feat=f"{hdr} — {heading} — {lbl} — {aft}"
                vec=encoder.encode([feat], normalize_embeddings=True).astype('float32')
                if backend=='faiss':
                    D,I=query(vec); dist,idx=D[0][0],I[0][0]
                else:
                    I,D=query(vec); idx,dist=I[0][0],D[0][0]
                sugg = lbl if dist> DIST_TH else ids[idx]
                new=normalize(sugg)
                if new != (w.field_name or ''):
                    if (new == ''):
                        new = 'unknown'
                    w.field_name=new
                    w.update()
                    print(f"   p{p+1} '{lbl}' → {new} (d={dist:.3f})")
        doc.save(dst)
        doc.close()
        print(f"   ✔ Saved {dst}\\n")

if __name__=='__main__':
    main()