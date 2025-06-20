import fitz  # PyMuPDF
import os, glob, json
from collections import defaultdict
from openai import OpenAI

# Configuration
INPUT_DIR = "./pdfs/bondforms10"
OUTPUT_JSON = "./json/widgets_hierch.json"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.replace(",", " ").strip().split()).lower()

def summarize_text(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Summarize this PDF document briefly."},
                {"role": "user", "content": text[:3000]}
            ],
            max_tokens=200
        )
        return normalize(response.choices[0].message.content)
    except Exception as e:
        print(f"[!] Error summarizing: {e}")
        return ""

def as_device(rect, page):
    """Transform widget rect to match rotated device coords from get_text()."""
    mat = fitz.Matrix(1, 1).prerotate(page.rotation)
    return rect * mat

def page_text_blocks(page):
    """Extract text lines as (Rect, text), already in device coords."""
    blocks = page.get_text("dict")['blocks']
    lines = []
    for b in blocks:
        for l in b.get("lines", []):
            txt = " ".join(s["text"] for s in l["spans"]).strip()
            if txt:
                x0, y0, x1, y1 = l["bbox"]
                rect = as_device(fitz.Rect(x0, y0, x1, y1), page)
                lines.append((rect, txt))
    return lines

def find_label_and_after_fixed(widget_rect, blocks,
                               same_line_y_tol=15,
                               x_tol=3,
                               below_above_tol=10):
    """
    Return (label_before, text_after) for one widget.

    • same_line_y_tol : ± pts to consider text on same line.
    • x_tol           : pts to allow tiny overlaps / gaps.
    • below_above_tol : pts for accepting subtitles under / over the field.
    """
    label = after = ""
    best_left_dx  = best_right_dx = float("inf")

    wx0, wx1 = widget_rect.x0, widget_rect.x1
    wcy      = (widget_rect.y0 + widget_rect.y1) / 2

    for rect, txt in blocks:
        if not txt:
            continue

        rcy = (rect.y0 + rect.y1) / 2

        # ---------- 1. same horizontal band ----------
        if abs(rcy - wcy) <= same_line_y_tol:
            # LEFT  (touching or a gap ≤ x_tol)
            if rect.x1 <= wx0 + x_tol:
                dx = wx0 - rect.x1        # 0 == touching
                if dx < best_left_dx:
                    label, best_left_dx = txt, dx
            # RIGHT (touching or a gap ≤ x_tol)
            elif rect.x0 >= wx1 - x_tol:
                dx = rect.x0 - wx1
                if dx < best_right_dx:
                    after, best_right_dx = txt, dx

        # ---------- 2. subtitle directly below ----------
        elif 0 < rect.y0 - widget_rect.y1 <= below_above_tol:
            # must overlap horizontally at least a little
            if rect.x1 > wx0 and rect.x0 < wx1 and not label:
                label = txt

        # ---------- 3. subtitle directly above ----------
        elif 0 < widget_rect.y0 - rect.y1 <= below_above_tol:
            if rect.x1 > wx0 and rect.x0 < wx1 and not label:
                label = txt

    return normalize(label), normalize(after)

def find_label_and_after_v2(widget_rect, blocks, x_tol=3, y_line_tol=0.4):
    label_txt = after_txt = ""
    best_label_dx = best_after_dx = float("inf")
    wy0, wy1, wx0, wx1 = widget_rect.y0, widget_rect.y1, widget_rect.x0, widget_rect.x1
    w_h   = wy1 - wy0
    w_cy  = (wy0 + wy1) / 2

    for rect, txt in blocks:
        if not txt:
            continue
        ry0, ry1, rx0, rx1 = rect.y0, rect.y1, rect.x0, rect.x1
        r_h  = ry1 - ry0
        r_cy = (ry0 + ry1) / 2
        y_tol = max(w_h, r_h) * y_line_tol
        if abs(r_cy - w_cy) <= y_tol:
            if rx1 <= wx0 + x_tol:
                dx = wx0 - rx1
                if dx < best_label_dx:
                    label_txt, best_label_dx = txt, dx
            elif rx0 >= wx1 - x_tol:
                dx = rx0 - wx1
                if dx < best_after_dx:
                    after_txt, best_after_dx = txt, dx
        overlap_x = max(0, min(rx1, wx1) - max(rx0, wx0))
        if overlap_x >= 0.2 * (wx1 - wx0):
            if ry1 <= wy0 and wy0 - ry1 <= y_tol:
                dx = wy0 - ry1
                if dx < best_label_dx:
                    label_txt, best_label_dx = txt, dx
            elif ry0 >= wy1 and ry0 - wy1 <= y_tol and not after_txt:
                after_txt = txt
    return normalize(label_txt), normalize(after_txt)


def find_label_and_after_v3(widget_rect, blocks,
                            x_tol=3,
                            y_line_tol=0.4,
                            min_h_overlap=0.15):
    """
    Return (label_before, text_after) for wide-widget forms:

      • Accept labels *above* or *below* if horizontal overlap ≥ min_h_overlap
        of the *smaller* of (widget width, block width).
      • Keep the old left / right logic with small x-tolerance.
    """
    label_txt = after_txt = ""
    best_label_d = best_after_d = float("inf")

    wx0, wx1, wy0, wy1 = widget_rect.x0, widget_rect.x1, widget_rect.y0, widget_rect.y1
    w_w  = wx1 - wx0
    w_h  = wy1 - wy0
    w_cy = (wy0 + wy1) / 2

    for rect, txt in blocks:
        if not txt:
            continue
        rx0, rx1, ry0, ry1 = rect.x0, rect.x1, rect.y0, rect.y1
        r_w  = rx1 - rx0
        r_h  = ry1 - ry0
        r_cy = (ry0 + ry1) / 2

        # dynamic vertical tolerance
        y_tol = max(w_h, r_h) * y_line_tol

        # ---------- SAME LINE (left / right) ----------
        if abs(r_cy - w_cy) <= y_tol:
            if rx1 <= wx0 + x_tol:                  # left label
                dx = wx0 - rx1
                if dx < best_label_d:
                    label_txt, best_label_d = txt, dx
            elif rx0 >= wx1 - x_tol:               # right after-text
                dx = rx0 - wx1
                if dx < best_after_d:
                    after_txt, best_after_d = txt, dx
            continue                                # done, next block

        # ---------- ABOVE / BELOW ----------
        # horizontal overlap *relative to smaller width*
        overlap = max(0, min(rx1, wx1) - max(rx0, wx0))
        min_width = min(w_w, r_w)
        if min_width and overlap / min_width >= min_h_overlap:

            # label **above** the field
            if ry1 <= wy0 and wy0 - ry1 <= y_tol:
                d = wy0 - ry1
                if d < best_label_d:
                    label_txt, best_label_d = txt, d
            # label **below** the field  (common in bond forms)
            elif ry0 >= wy1 and ry0 - wy1 <= y_tol:
                d = ry0 - wy1
                if d < best_label_d:
                    label_txt, best_label_d = txt, d

    return normalize(label_txt), normalize(after_txt)


def find_doc_header(page):
    header_lines = []
    blocks = page.get_text("dict")['blocks']
    for b in blocks:
        if b.get("number") == 0:
            for l in b.get("lines", []):
                line = " ".join(span["text"] for span in l["spans"]).strip()
                if line:
                    header_lines.append(line)
    return normalize(" ".join(header_lines[:3]))

def find_page_heading(page):
    lines = page_text_blocks(page)
    return normalize(lines[0][1]) if lines else ""

def extract_full_text(doc):
    return "\n".join(page.get_text("text") for page in doc)

documents = []

for pdf_path in glob.glob(os.path.join(INPUT_DIR, "*.pdf")):
    doc = fitz.open(pdf_path)
    doc_header = find_doc_header(doc[0]) if doc.page_count else ""
    doc_name = os.path.basename(pdf_path)
    page_heading = find_page_heading(doc[0]) if doc.page_count else ""
    full_text = extract_full_text(doc)
    summary = summarize_text(full_text)
    fields = []
    field_counter = 1

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text_blocks = page_text_blocks(page)

        for w in page.widgets():
            w_rect_dev = as_device(w.rect, page)

            # Pad razor-thin widget rects
            min_size = 4
            if w_rect_dev.width < min_size:
                pad = (min_size - w_rect_dev.width) / 2
                w_rect_dev.x0 -= pad
                w_rect_dev.x1 += pad
            if w_rect_dev.height < min_size:
                pad = (min_size - w_rect_dev.height) / 2
                w_rect_dev.y0 -= pad
                w_rect_dev.y1 += pad

            name = normalize(w.field_name or "")
            tooltip = normalize(getattr(w, "field_label", ""))
            label, after = find_label_and_after_fixed(w_rect_dev, text_blocks)

            fields.append({
                "counter": field_counter,
                "field_name": name,
                "tooltip": tooltip,
                "label_before": label,
                "text_after": after
            })
            field_counter += 1
    doc.close()

    documents.append({
        "document_name": doc_name,
        "document_header": doc_header,
        "page_heading": page_heading,
        "summary": summary,
        "full_text": full_text,
        "fields": fields
    })

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print(f"✓ Done. {len(documents)} documents processed and written to {OUTPUT_JSON}")
