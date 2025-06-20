import fitz  # PyMuPDF >= 1.23.0
import pandas as pd
from rapidfuzz import process, fuzz

# ----------------------------------------
# 1. CSV -> { label (lower‑case) : MERGE_FIELD_ID }
# ----------------------------------------
df = pd.read_csv("merge_fields.csv")
df["desc"] = df.MERGE_FIELD_DESCRIPTION.str.lower().str.strip()
LOOKUP = dict(zip(df.desc, df.MERGE_FIELD_ID))

# ----------------------------------------
# 2. Helper: find the nearest text label to a widget
# ----------------------------------------

def find_label(widget_rect, blocks):
    """Return the text block that most likely describes the widget.

    blocks = [(Rect, text_lower)] for the current page.
    We prefer a block *left on the same line*; if none, a block right above.
    """
    best_txt, best_score = None, 9999.0

    for rect, txt in blocks:
        # a) Same horizontal line, text ending just to the left of the widget
        if abs(rect.y1 - widget_rect.y0) < 12 and rect.x1 < widget_rect.x0:
            dx = widget_rect.x0 - rect.x1
            if dx < best_score:
                best_txt, best_score = txt, dx
        # b) Directly above, roughly aligned in X
        elif rect.y1 < widget_rect.y0 and abs(rect.x0 - widget_rect.x0) < 5:
            dy = widget_rect.y0 - rect.y1
            if dy < best_score:
                best_txt, best_score = txt, dy

    return best_txt

# ----------------------------------------
# 3. Open PDF, walk widgets, rename
# ----------------------------------------

doc = fitz.open("form1_raw.pdf")

for page in doc:
    # get_text("blocks") may yield image blocks where b[4] is None → filter them out
    raw_blocks = page.get_text("blocks")
    blocks = [
        (fitz.Rect(*b[:4]), b[4].strip().lower())
        for b in raw_blocks
        if b[4]  # skip blocks with None / empty text
    ]

    for widget in page.widgets():
        label = find_label(widget.rect, blocks)
        if not label:
            continue  # couldn’t find nearby text

        match, score, _ = process.extractOne(label, LOOKUP.keys(), scorer=fuzz.QRatio)
        if score < 70:
            continue  # low confidence

        widget.field_name = LOOKUP[match]
        widget.update()

# Save the renamed form
doc.save("form1_renamed.pdf")
print("Field names updated → form1_renamed.pdf")
