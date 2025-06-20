#!/usr/bin/env python
# extract_fields.py
from typing import List, Dict
import fitz  # PyMuPDF
import argparse, json, pathlib

MARGIN = 20          # px for label-search window
MAX_CHARS = 60

def _label(page, rect):
    words = page.get_text("words")
    return " ".join(w[4] for w in words if fitz.Rect(w[:4]).intersects(rect))[:MAX_CHARS]

def extract(pdf) -> List[Dict]:
    doc, out = fitz.open(pdf), []
    for pno in range(doc.page_count):
        page = doc[pno]
        for w in page.widgets():
            if not w.field_name:
                continue
            r = w.rect
            lab = _label(page, fitz.Rect(r.x0-MARGIN, r.y0-MARGIN,
                                         r.x1+MARGIN, r.y1+MARGIN))
            out.append({"page": pno, "name": w.field_name,
                        "ftype": w.field_type, "label": lab})
    doc.close()
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    res = extract(a.pdf)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"✅  wrote {len(res)} fields → {a.out}")
    else:
        print(json.dumps(res, indent=2, ensure_ascii=False))
