#!/usr/bin/env python
# 01_harvest.py
"""
Walk through every PDF in ./templates/, extract:
    widget_name,  label_text (words around it)
Write one CSV row per widget.
"""
import fitz, csv, glob, os, argparse

MARGIN = 30          # points around widget to grab context
MAX_WORDS = 25       # keep context short

def label_text(page, rect):
    words = page.get_text("words")
    return " ".join(
        w[4] for w in words
        if fitz.Rect(w[:4]).intersects(rect)
    )[:MAX_WORDS*20]        # crude truncate

def harvest(templates_folder, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["widgetName", "context"])
        for pdf in glob.glob(os.path.join(templates_folder, "*.pdf")):
            doc = fitz.open(pdf)
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
                        wr.writerow([w.field_name, text])
            doc.close()
    print(f"✅  wrote “{out_csv}”")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="templates")
    ap.add_argument("-o", "--out", default="widget_catalog.csv")
    args = ap.parse_args()

    harvest(templates_folder=args.folder, out_csv=args.out)