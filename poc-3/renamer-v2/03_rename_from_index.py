#!/usr/bin/env python
# 03_rename_from_index.py
import fitz, faiss, pickle, argparse, json, sys
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL)

MARGIN = 30
def surrounding_text(page, rect):
    words = page.get_text("words")
    return " ".join(w[4] for w in words if fitz.Rect(w[:4]).intersects(rect))

def _rename_widget(doc, widg, new_name):
    """rename regardless of PyMuPDF version"""
    if hasattr(widg, "set_field_name"):        # 1.25+
        widg.set_field_name(new_name)
    elif hasattr(widg, "set_name"):            # 1.22–1.24
        widg.set_name(new_name)
    else:                                      # ≤1.21 fallback
        try:
            widg.field_name = new_name
        except AttributeError:
            # brute-force: overwrite /T in the widget’s object
            doc.xref_set_key(widg.xref, "T", fitz.PDF_NAME(new_name))

def load_index(prefix="widget"):
    return (
        faiss.read_index(f"{prefix}.faiss"),
        pickle.load(open(f"{prefix}_names.pkl", "rb"))
    )

def propose_names(pdf_in, prefix="widget", k=1):
    index, names = load_index(prefix)
    doc, mapping = fitz.open(pdf_in), {}
    for page in doc:
        for w in page.widgets():
            ctx = surrounding_text(
                page,
                fitz.Rect(w.rect.x0-MARGIN, w.rect.y0-MARGIN,
                          w.rect.x1+MARGIN, w.rect.y1+MARGIN)
            )
            if not ctx.strip():
                continue
            q_vec = embedder.encode([ctx], convert_to_numpy=True).astype("float32")
            _, idx = index.search(q_vec, k)
            mapping[w.field_name] = names[idx[0][0]]
    doc.close()
    return mapping

def rename(pdf_in, pdf_out, mapping):
    doc = fitz.open(pdf_in)
    changed = 0
    for page in doc:
        for w in page.widgets():
            if w.field_name in mapping and mapping[w.field_name] != w.field_name:
                _rename_widget(doc, w, mapping[w.field_name])
                w.update()
                changed += 1
    doc.save(pdf_out)
    doc.close()
    print(f"✅  saved → {pdf_out}  (renamed {changed} widgets)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_in")
    ap.add_argument("pdf_out")
    ap.add_argument("--idx", default="widget")
    ap.add_argument("-k", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="just print mapping")
    args = ap.parse_args()

    m = propose_names(args.pdf_in, args.idx, args.k)
    if args.dry:
        print(json.dumps(m, indent=2))
        sys.exit()
    rename(args.pdf_in, args.pdf_out, m)
