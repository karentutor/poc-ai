#!/usr/bin/env python
# build_catalog_from_pdfs.py
import fitz, csv, glob, os, argparse

def scan_pdf(pdf_path):
    page = fitz.open(pdf_path)[0]           # first page is enough for most labels
    words = set(w[4].strip() for w in page.get_text("words"))
    return [w for w in words if 2 < len(w) < 60]   # crude filter

def build(folder, out_csv="forms.csv"):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["textId", "shortDesc", "longDesc"])
        for pdf in glob.glob(os.path.join(folder, "*.pdf")):
            for t in scan_pdf(pdf):
                wr.writerow([t.replace(" ", ""), t, t])   # cheap placeholders
    print(f"✅ catalog → {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_folder")
    ap.add_argument("-o", "--out", default="forms.csv")
    args = ap.parse_args()
    build(folder=args.pdf_folder, out_csv=args.out)