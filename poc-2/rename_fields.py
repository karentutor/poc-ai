#!/usr/bin/env python
# rename_fields.py
"""
Rename form fields in a PDF according to a JSON mapping:
{
    "oldName": "newName",
    ...
}
"""
import fitz           # PyMuPDF
import json, argparse, sys

def rename_widgets(pdf_in: str, pdf_out: str, mapping_file: str):
    mapping = json.load(open(mapping_file))
    doc = fitz.open(pdf_in)
    changed, skipped = 0, 0

    for page in doc:
        for w in page.widgets():
            old = w.field_name
            if old in mapping:
                new = mapping[old]
                if new != old:
                    w.set_name(new)        # PyMuPDF ≥ 1.22
                    w.update()
                    changed += 1
            else:
                skipped += 1

    doc.save(pdf_out)
    doc.close()
    print(f"✅  saved → {pdf_out}  (renamed {changed}, left {skipped} untouched)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_in")
    ap.add_argument("pdf_out")
    ap.add_argument("mapping_json")
    args = ap.parse_args()

    try:
        rename_widgets(args.pdf_in, args.pdf_out, args.mapping_json)
    except AttributeError:
        sys.exit("PyMuPDF ≥ 1.22 is required ( `pip install --upgrade pymupdf` ).")