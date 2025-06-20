import fitz  # PyMuPDF
import os, glob, json
from collections import defaultdict
from openai import OpenAI

# Configuration
INPUT_DIR = "./pdfs/bondforms10"
OUTPUT_JSON = "./json/widgets.json"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Updated normalize function to guard against NoneType inputs

def normalize(text: str) -> str:
    """
    Trim, collapse whitespace, remove commas, lowercase.
    If text is None or empty, returns an empty string.
    """
    if not text:
        return ""
    return " ".join(text.replace(",", " ")
                     .strip()
                     .split()).lower()


def summarize_text(text: str) -> str:
    """Generate a brief summary of the given text using gpt-4o."""
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

def find_label_and_after(widget_rect, blocks):
    """
    Return the closest label before and text after widget_rect,
    using the line’s vertical midpoint and symmetric x‑checks.
    """
    label_txt, after_txt = "", ""
    best_label_dist = best_after_dist = float('inf')

    # Compute widget’s vertical center
    widget_cy = (widget_rect.y0 + widget_rect.y1) / 2

    for rect, txt in blocks:
        if not txt:
            continue

        # Compute this block’s vertical center
        block_cy = (rect.y0 + rect.y1) / 2

        # If it sits roughly on the same line (within 8 points)
        if abs(block_cy - widget_cy) < 8:
            # text to the left → candidate for label_before
            if rect.x1 < widget_rect.x0:
                dx = widget_rect.x0 - rect.x1
                if dx < best_label_dist:
                    label_txt, best_label_dist = txt, dx
            # text to the right → candidate for text_after
            elif rect.x0 > widget_rect.x1:
                dx = rect.x0 - widget_rect.x1
                if dx < best_after_dist:
                    after_txt, best_after_dist = txt, dx

    return normalize(label_txt), normalize(after_txt)



def page_text_blocks(page):
    """Extract all text lines with their bounding rects."""
    blocks = page.get_text("dict")['blocks']
    lines = []
    for b in blocks:
        for l in b.get("lines", []):
            txt = " ".join(s["text"] for s in l["spans"]).strip()
            if txt:
                x0, y0, x1, y1 = l["bbox"]
                lines.append((fitz.Rect(x0, y0, x1, y1), txt))
    return lines

def find_doc_header(page):
    """Grab and normalize the top block (header) of the first page."""
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
    """Take and normalize the first text line on each page as its heading."""
    lines = page_text_blocks(page)
    return normalize(lines[0][1]) if lines else ""

def extract_full_text(doc):
    """Concatenate all pages into one big text blob."""
    return "\n".join(page.get_text("text") for page in doc)

# Aggregate per field_name
field_map = defaultdict(lambda: {
    "tooltip": "",
    "label_before": set(),
    "text_after": set(),
    "document_header": "",
    "page_heading": "",
    "document_name": "",
    "file": "",
    "summary": ""
})

# Process each PDF
for pdf_path in glob.glob(os.path.join(INPUT_DIR, "*.pdf")):
    doc = fitz.open(pdf_path)
    doc_header = find_doc_header(doc[0]) if doc.page_count else ""
    doc_name = os.path.basename(pdf_path)
    page_heading = find_page_heading(doc[0]) if doc.page_count else ""
    full_text = extract_full_text(doc)
    summary = summarize_text(full_text)

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text_blocks = page_text_blocks(page)

        for w in page.widgets():
            name    = normalize(w.field_name or "")
            tooltip = normalize(getattr(w, "field_label", ""))
            label, after = find_label_and_after(w.rect, text_blocks)
            if not label:
                continue

            data = field_map[name]
            data["tooltip"] = tooltip or data["tooltip"]
            data["label_before"].add(label)
            if after:
                data["text_after"].add(after)
            data["document_header"] = doc_header
            data["page_heading"] = page_heading
            data["document_name"] = doc_name
            data["file"] = doc_name
            data["summary"] = summary
    doc.close()

# Prepare JSON output
results = []
for field_name, data in field_map.items():
    results.append({
        "field_name": field_name,
        "tooltip": data["tooltip"],
        "label_before": "; ".join(sorted(data["label_before"])),
        "text_after": "; ".join(sorted(data["text_after"])),
        "document_header": data["document_header"],
        "page_heading": data["page_heading"],
        "document_name": data["document_name"],
        "file": data["file"],
        "summary": data["summary"]
    })

# Write JSON
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✓ Done. {len(results)} unique field names written to {OUTPUT_JSON}")
