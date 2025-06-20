#!/usr/bin/env python
# 01_harvest.py
"""
Walk through every PDF in ./templates/, extract:
    widget_name,  label_text (words around it)
Write one JSON file with widgets organized by document.
"""
import fitz, json, glob, os, argparse, time
from datetime import datetime
from tqdm import tqdm
import re

MARGIN = 30          # points around widget to grab context
MAX_WORDS = 25       # keep context short
MIN_FONT_SIZE = 11   # minimum font size to consider as heading
SAVE_INTERVAL = 100  # save progress every N documents

def natural_sort_key(s):
    """Generate a key for natural sorting of strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def save_progress(catalog, out_json, is_final=False):
    """Save current progress to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "is_final": is_final,
        "documents": catalog
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    if is_final:
        tqdm.write(f"\n✅ Successfully wrote final results to: {out_json}")
    else:
        tqdm.write(f"\n💾 Saved progress to: {out_json}")

def get_document_heading(doc):
    """Extract heading from first page of document."""
    if len(doc) == 0:
        return ""
    
    # Get text from first page
    first_page = doc[0]
    text = first_page.get_text()
    
    # Split into lines and find first non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines[0] if lines else ""

def get_section_headings(doc):
    """Extract section headings based on font size and formatting."""
    headings = []
    
    for page_num, page in enumerate(doc):
        # Get text blocks with their font information
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                if "spans" not in line:
                    continue
                    
                # Check if any span in the line has a large font
                is_heading = False
                line_text = ""
                
                for span in line["spans"]:
                    if span["size"] >= MIN_FONT_SIZE:
                        is_heading = True
                    line_text += span["text"]
                
                if is_heading and line_text.strip():
                    headings.append({
                        "text": line_text.strip(),
                        "page": page_num + 1,
                        "font_size": max(span["size"] for span in line["spans"])
                    })
    
    return headings

def get_full_text(doc):
    """Extract all text from document."""
    full_text = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text.append(text.strip())
    return "\n\n".join(full_text)

def label_text(page, rect):
    words = page.get_text("words")
    return " ".join(
        w[4] for w in words
        if fitz.Rect(w[:4]).intersects(rect)
    )[:MAX_WORDS*20]        # crude truncate

def harvest(templates_folder, out_json):
    tqdm.write(f"\n🔍 Scanning PDFs in: {os.path.abspath(templates_folder)}")
    tqdm.write(f"📝 Output will be saved to: {os.path.abspath(out_json)}\n")
    
    catalog = {}
    pdf_files = glob.glob(os.path.join(templates_folder, "*.pdf"))
    pdf_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    tqdm.write(f"📚 Found {len(pdf_files)} PDF files to process")
    
    total_widgets = 0
    total_time = 0
    processed_count = 0
    
    for pdf in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        doc_name = os.path.basename(pdf)
        doc = fitz.open(pdf)
        
        catalog[doc_name] = {
            "documentName": doc_name,
            "heading": get_document_heading(doc),
            "full_text": get_full_text(doc),
            "section_headings": get_section_headings(doc),
            "fields": []
        }
        
        doc_start_time = time.time()
        doc_widgets = 0
        
        for p in doc:
            for w in p.widgets():
                if not w.field_name:
                    continue
                r = w.rect
                text = label_text(
                    p,
                    fitz.Rect(r.x0-MARGIN, r.y0-MARGIN,
                              r.x1+MARGIN, r.y1+MARGIN)
                )
                if text.strip():
                    catalog[doc_name]["fields"].append({
                        "widgetName": w.field_name,
                        "context": text
                    })
                    doc_widgets += 1
        
        doc_time = time.time() - doc_start_time
        total_time += doc_time
        total_widgets += doc_widgets
        processed_count += 1
        
        # Save progress periodically
        if processed_count % SAVE_INTERVAL == 0:
            save_progress(catalog, out_json)
            tqdm.write(f"📊 Progress: {processed_count}/{len(pdf_files)} documents processed")
            tqdm.write(f"📊 Current widget count: {total_widgets}")
        
        doc.close()
    
    # Save final results
    save_progress(catalog, out_json, is_final=True)
    
    avg_time_per_widget = total_time / total_widgets if total_widgets > 0 else 0
    tqdm.write(f"📊 Found {total_widgets} widgets across {len(catalog)} documents")
    tqdm.write(f"⏱️ Total processing time: {total_time:.2f} seconds")
    tqdm.write(f"⏱️ Average time per widget: {avg_time_per_widget*1000:.2f} ms\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="templates")
    ap.add_argument("-o", "--out", default="widget_catalog.json")
    args = ap.parse_args()

    harvest(templates_folder=args.folder, out_json=args.out)