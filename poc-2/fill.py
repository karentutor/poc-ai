#!/usr/bin/env python
# fill_pdf.py
import fitz, json, argparse

def fill(inpdf, outpdf, values):
    doc = fitz.open(inpdf)
    for page in doc:
        for w in page.widgets():
            if w.field_name in values:
                w.field_value = values[w.field_name]
                w.update()
    doc.save(outpdf)
    doc.close()
    print(f"📝  saved → {outpdf}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pdf_in")
    p.add_argument("pdf_out")
    p.add_argument("json_values")       # {"fieldName": "text", …}
    a = p.parse_args()
    vals = json.loads(open(a.json_values).read())
    fill(a.pdf_in, a.pdf_out, vals)
