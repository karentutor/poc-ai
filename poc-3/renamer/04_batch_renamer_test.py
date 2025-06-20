#!/usr/bin/env python
# 03_rename_from_index.py
import fitz, faiss, pickle, argparse, json, sys, os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL)

MARGIN = 30
def surrounding_text(page, rect):
    words = page.get_text("words")
    return " ".join(w[4] for w in words if fitz.Rect(w[:4]).intersects(rect))

def load_index(prefix="widget"):
    return (
        faiss.read_index(f"{prefix}.faiss"),
        pickle.load(open(f"{prefix}_names.pkl", "rb"))
    )

def propose_names(pdf_in, prefix="widget", k=1):
    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(pdf_in))[0]
    report_csv_path = os.path.join(reports_dir, f"{base_name}.csv")
    
    index, names = load_index(prefix)
    doc, mapping = fitz.open(pdf_in), {}
    total_counter = 0
    same_counter = 0
    
    with open(report_csv_path, 'w') as report_csv:
        for page in doc:
            for w in page.widgets():
                current_name = w.field_name
                ctx = surrounding_text(
                    page,
                    fitz.Rect(w.rect.x0-MARGIN, w.rect.y0-MARGIN,
                              w.rect.x1+MARGIN, w.rect.y1+MARGIN)
                )
                if not ctx.strip():
                    continue
                q_vec = embedder.encode([ctx], convert_to_numpy=True).astype("float32")
                _, idx = index.search(q_vec, k)
                name = names[idx[0][0]]
                mapping[w.field_name] = name
                total_counter += 1
                if current_name == name:
                    same_counter += 1
                report_csv.write(f"{current_name},{name}\n")
                # print(f"Current name: {current_name}, new name: {name}")
    
    if total_counter > 0:
        percent_same = (same_counter / total_counter) * 100
        rounded_percent_same = round(percent_same, 2)
        tqdm.write(f"PDF: {base_name}.pdf Accuracy: {rounded_percent_same}% Total counter: {total_counter}, same counter: {same_counter}")
    else:
        tqdm.write(f"PDF: {base_name}.pdf Accuracy: 0% Total counter: {total_counter}, same counter: {same_counter}")    
    doc.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder_in")
    ap.add_argument("--idx", default="widget")
    ap.add_argument("-k", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="just print mapping")
    args = ap.parse_args()

    # Get list of PDF files
    pdf_files = [f for f in os.listdir(args.folder_in) if f.lower().endswith('.pdf')]
    
    # Process files with progress bar
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_in = os.path.join(args.folder_in, pdf)
        propose_names(pdf_in, args.idx, args.k)
