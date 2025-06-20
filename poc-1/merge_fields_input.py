# ─────────────────────────────────────────────────────────────────────────────
#  Auto‑rename PDF form fields using CSV + OpenAI (>=1.0) matching
# ─────────────────────────────────────────────────────────────────────────────
#    • Reads merge_fields.csv  → canonical field list (ID + description)
#    • Opens raw PDF, extracts each widget + its nearby label text
#    • First tries RapidFuzz. If confidence < 70 → asks an OpenAI model
#    • Saves a new PDF with /T set to the chosen MERGE_FIELD_ID
#
#  Prereqs:  pip install pymupdf pandas rapidfuzz python‑dotenv openai>=1.0
#
#  Files expected in same folder:
#    form1_raw.pdf
#    merge_fields.csv   (MERGE_FIELD_ID,MERGE_FIELD_NAME,MERGE_FIELD_DESCRIPTION)
# ─────────────────────────────────────────────────────────────────────────────

import os, re, json, sys
import fitz                                  # PyMuPDF ≥1.23
import pandas as pd
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------------------------------------------------------
# 0. Config & helpers
# ----------------------------------------------------------------------------
load_dotenv()
MODEL          = "gpt-4o-mini"   # change if you have a preferred model
FUZZ_THRESHOLD = 70              # ≥ this: accept fuzzy match, else LLM

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    sys.exit("⚠️  OPENAI_API_KEY missing (set env var or .env)")

# ----------------------------------------------------------------------------
# 1. CSV → lookup + JSON list for LLM
# ----------------------------------------------------------------------------

DF = pd.read_csv("./csv/merge_fields.csv")
DF["desc"] = DF.MERGE_FIELD_DESCRIPTION.str.lower().str.strip()
LOOKUP = dict(zip(DF.desc, DF.MERGE_FIELD_ID))
CHOICES_JSON = DF[["MERGE_FIELD_ID", "desc"]].to_dict(orient="records")
CHOICES_STR  = json.dumps(CHOICES_JSON, ensure_ascii=False)

# ----------------------------------------------------------------------------
# 2. Text‑cleaning and geometry helpers
# ----------------------------------------------------------------------------

def clean(txt: str) -> str:
    """Lower‑case, collapse underscores/blanks → single spaces."""
    return re.sub(r"[_\s]+", " ", txt).strip().lower()


def find_label(widget_rect, blocks):
    """Return the label text most likely describing widget_rect."""
    best_txt, best_metric = None, 1e9
    for rect, txt in blocks:
        if not txt:
            continue
        # same line, left
        if abs(rect.y1 - widget_rect.y0) < 12 and rect.x1 < widget_rect.x0:
            dx = widget_rect.x0 - rect.x1
            if dx < best_metric:
                best_txt, best_metric = txt, dx
        # directly above
        elif rect.y1 < widget_rect.y0 and abs(rect.x0 - widget_rect.x0) < 5:
            dy = widget_rect.y0 - rect.y1
            if dy < best_metric:
                best_txt, best_metric = txt, dy
    return best_txt

# ----------------------------------------------------------------------------
# 3. Ask OpenAI once per unresolved label
# ----------------------------------------------------------------------------

def llm_choose_id(label: str) -> str | None:
    prompt = (
        "You are a helpful assistant that maps raw form field labels to canonical merge field IDs.\n"
        "Output ONLY the matching MERGE_FIELD_ID from the list below, nothing else.\n\n"
        f"Label: {label!r}\n\nAvailable IDs with descriptions (JSON):\n{CHOICES_STR}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Return only a valid MERGE_FIELD_ID from the list."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
        )
        candidate = resp.choices[0].message.content.strip().split()[0]
        return candidate if candidate in DF.MERGE_FIELD_ID.values else None
    except Exception as exc:
        print("[LLM‑error]", exc)
        return None

# ----------------------------------------------------------------------------
# 4. Walk PDF, rename widgets
# ----------------------------------------------------------------------------

IN_PDF  = "./pdfs/forms/form1_blank.pdf"
OUT_PDF = "./pdfs/forms/form1_renamed.pdf"

doc = fitz.open(IN_PDF)
doc.need_appearances(True)  

for page in doc:
    raw_blocks = page.get_text("blocks")
    blocks = [
        (fitz.Rect(*b[:4]), clean(b[4] or ""))
        for b in raw_blocks if b[4]
    ]

    for widget in page.widgets():
        label_raw = find_label(widget.rect, blocks)
        if not label_raw:
            continue
        label = clean(label_raw)
        print("LABEL:",label)

        # 4a. Try RapidFuzz first
        match, score, _ = process.extractOne(label, LOOKUP.keys(), scorer=fuzz.QRatio)
        if score >= FUZZ_THRESHOLD:
            new_id = LOOKUP[match]
        else:
            new_id = llm_choose_id(label)
            if not new_id:
                print(f"⚠️ Unresolved label: {label_raw!r}")
                continue

        widget.field_name = new_id
        widget.update()

# ----------------------------------------------------------------------------
# 5. Save
# ----------------------------------------------------------------------------

doc.save(OUT_PDF)
print(f"Field names updated → {OUT_PDF}")
