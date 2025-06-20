#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto‑rename PDF form fields using context from widgets.json and ChatGPT suggestions.
For each widget, we use its document context (header, page heading), label_before, and text_after
and ask ChatGPT to propose a concise snake_case field name.
"""
import os
import json
import re
import fitz  # PyMuPDF ≥1.23
from openai import OpenAI
from dotenv import load_dotenv

# ─── CONFIG & OPENAI SETUP ───────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o"

# ─── PATHS ────────────────────────────────────────────────────────────────────
WIDGETS_JSON = "./json/widgets.json"
PDF_DIR       = "./pdfs/forms"
OUTPUT_DIR    = "./pdfs/forms/renamed"

# ─── UTILITIES ────────────────────────────────────────────────────────────────
def normalize(name: str) -> str:
    """Convert text to snake_case safe for PDF field names."""
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")

def page_text_blocks(page):
    blocks = page.get_text("dict")["blocks"]
    lines = []
    for b in blocks:
        for l in b.get("lines", []):
            txt = " ".join(span.get("text","") for span in l.get("spans", [])).strip()
            if txt:
                x0, y0, x1, y1 = l["bbox"]
                lines.append((fitz.Rect(x0, y0, x1, y1), txt))
    return lines

# Label and after detection unchanged

def find_label_and_after(widget_rect, blocks):
    label, after = "", ""
    best_lbl, best_aft = float('inf'), float('inf')
    wcy = (widget_rect.y0 + widget_rect.y1) / 2
    for rect, txt in blocks:
        if not txt: continue
        bcy = (rect.y0 + rect.y1) / 2
        if abs(bcy - wcy) < 8:
            if rect.x1 < widget_rect.x0:
                d = widget_rect.x0 - rect.x1
                if d < best_lbl:
                    label, best_lbl = txt, d
            elif rect.x0 > widget_rect.x1:
                d = rect.x0 - widget_rect.x1
                if d < best_aft:
                    after, best_aft = txt, d
    return label.strip(), after.strip()

# Header and page heading extraction unchanged

def find_doc_header(doc):
    hdr_lines = []
    if doc.page_count:
        page = doc.load_page(0)
        for b in page.get_text("dict")["blocks"]:
            if b.get("number") == 0:
                for l in b.get("lines", []):
                    txt = " ".join(span.get("text","") for span in l.get("spans", [])).strip()
                    if txt:
                        hdr_lines.append(txt)
    return " ".join(hdr_lines[:2])


def find_page_heading(page):
    blocks = page_text_blocks(page)
    return blocks[0][1] if blocks else ""

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    entries = json.load(open(WIDGETS_JSON, encoding="utf-8"))
    # index by (document_name, page_heading, label, after)
    index = {(
        e.get("document_name"), e.get("page_heading",""),
        e.get("label_before",""), e.get("text_after","")
    ): e for e in entries}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in sorted(os.listdir(PDF_DIR)):
        if not fname.lower().endswith('.pdf'): continue
        in_pdf = os.path.join(PDF_DIR, fname)
        out_pdf = os.path.join(OUTPUT_DIR, fname.replace('.pdf','_renamed.pdf'))
        print(f"Processing {fname}...")
        doc = fitz.open(in_pdf)
        doc_hdr = find_doc_header(doc)
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            pg_head = find_page_heading(page)
            blocks = page_text_blocks(page)
            for widget in page.widgets():
                lbl, aft = find_label_and_after(widget.rect, blocks)
                key = (fname, pg_head, lbl, aft)
                entry = index.get(key)
                if not entry:
                    continue
                # build prompt for ChatGPT
                prompt = f"""
Suggest a concise snake_case field name for a PDF form field using the following context:
Document header: {doc_hdr}
Page heading: {pg_head}
Label before field: {lbl}
Text after field: {aft}
Current JSON entry: {entry}
Provide only the field name.
"""
                try:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role":"system","content":"You are an expert at naming PDF form fields."},
                            {"role":"user","content":prompt}
                        ],
                        max_tokens=10,
                        temperature=0
                    )
                    suggestion = resp.choices[0].message.content.strip().split()[0]
                except Exception as e:
                    print(f"[!] LLM error: {e}")
                    suggestion = lbl
                new_name = normalize(suggestion)
                print(f"  [{pno+1}] '{entry['field_name']}' → '{new_name}'")
                old = widget.field_name or ''
                widget.field_name = new_name
                widget.update()
                print(f"  [{pno+1}] '{old}' → '{new_name}'")
        doc.save(out_pdf)
        doc.close()
        print(f"✔ Saved: {out_pdf}\n")

if __name__=='__main__':
    main()
