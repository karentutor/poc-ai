# app/renamer.py
import faiss, pickle, fitz, tempfile, io
from sentence_transformers import SentenceTransformer

EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
INDEX    = faiss.read_index("widget.faiss")
NAMES    = pickle.load(open("widget_names.pkl", "rb"))
MARGIN   = 30

def _context(page, rect):
    words = page.get_text("words")
    return " ".join(w[4] for w in words if fitz.Rect(w[:4]).intersects(rect))

def _rename_widget(doc, widg, new_name):
    if hasattr(widg, "set_field_name"):
        widg.set_field_name(new_name)
    elif hasattr(widg, "set_name"):
        widg.set_name(new_name)
    else:
        widg.field_name = new_name  # fallback pre-1.22

def rename_pdf(binary: bytes, k: int = 1) -> bytes:
    doc = fitz.open(stream=binary, filetype="pdf")
    total_widgets = 0
    mapping_info = []
    changed_widgets = 0
    for p in doc:
        for w in p.widgets():
            ctx = _context(p, fitz.Rect(w.rect.x0-MARGIN, w.rect.y0-MARGIN,
                                        w.rect.x1+MARGIN, w.rect.y1+MARGIN))
            if not ctx.strip():
                continue
            vec = EMBEDDER.encode([ctx], convert_to_numpy=True).astype("float32")
            _, idx = INDEX.search(vec, k)
            canonical = NAMES[idx[0][0]]
            original_name = w.field_name
            if canonical != w.field_name:
                _rename_widget(doc, w, canonical)
                w.update()
                changed_widgets += 1
            mapping_info.append({
                "page": p.number + 1,
                "original_name": original_name,
                "new_name": canonical,
                "context": ctx
            })
            total_widgets += 1
            
    print(f"Total widgets: {total_widgets}")
    print(f"Total pages: {len(doc)}")
    print(f"Total fields: {len(NAMES)}")
    
    # Calculate overall accuracy as percentage of fields changed
    unchanged_widgets = total_widgets - changed_widgets
    accuracy_rate = 0
    if total_widgets > 0:
        accuracy_rate = round((unchanged_widgets / total_widgets) * 100, 2)
    
    buf = io.BytesIO()
    doc.save(buf)          # writes the entire PDF into buf
    doc.close()
    buf.seek(0)
    return buf.read(), {
        "mapping_info": mapping_info,
        "total_widgets": total_widgets,
        "changed_widgets": changed_widgets,
        "unchanged_widgets": unchanged_widgets,
        "accuracy": accuracy_rate
    }
