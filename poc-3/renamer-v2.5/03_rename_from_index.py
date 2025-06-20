#!/usr/bin/env python
# 03_rename_from_index.py
import fitz, faiss, pickle, argparse, json, sys, os, csv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import time
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Available similarity metrics
SIMILARITY_METRICS = {
    "cosine": lambda x, y: 1 - cosine(x, y),  # Cosine similarity (default)
    "euclidean": lambda x, y: 1 / (1 + euclidean(x, y)),  # Normalized Euclidean distance
    "dot": lambda x, y: np.dot(x, y),  # Dot product
    "cosine_sklearn": lambda x, y: cosine_similarity([x], [y])[0][0]  # sklearn's cosine similarity
}

# Default thresholds for different metrics
DEFAULT_THRESHOLDS = {
    "cosine": 0.7,  # Higher is better
    "euclidean": 0.3,  # Higher is better
    "dot": 0.5,  # Higher is better
    "cosine_sklearn": 0.7  # Higher is better
}

MARGIN = 30

def surrounding_text(page, rect):
    words = page.get_text("words")
    return " ".join(w[4] for w in words if fitz.Rect(w[:4]).intersects(rect))

def _rename_widget(doc, widg, new_name):
    """rename regardless of PyMuPDF version"""
    if hasattr(widg, "set_field_name"):        # 1.25+
        widg.set_field_name(new_name)
    elif hasattr(widg, "set_name"):            # 1.22–1.24
        widg.set_name(new_name)
    else:                                      # ≤1.21 fallback
        try:
            widg.field_name = new_name
        except AttributeError:
            # brute-force: overwrite /T in the widget's object
            doc.xref_set_key(widg.xref, "T", fitz.PDF_NAME(new_name))

def load_index(prefix="widget", use_gpu=False):
    try:
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for indexing")
            
        # Load model and index
        model_path = prefix if os.path.exists(prefix) else "sentence-transformers/all-MiniLM-L6-v2"
        embedder = SentenceTransformer(model_path, device=device)
        
        index = faiss.read_index(f"{prefix}.faiss")
        names = pickle.load(open(f"{prefix}_names.pkl", "rb"))
        
        return index, names, embedder
    except (FileNotFoundError, IOError) as e:
        print(f"Error: Could not load index {prefix}: {e}")
        return None, None, None

def propose_names(pdf_in, prefix="widget", k=1, threshold=None, similarity_metric="cosine", use_gpu=False):
    start_time = time.time()
    
    # Set default threshold based on similarity metric if not provided
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS[similarity_metric]
    
    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(pdf_in))[0]
    report_csv_path = os.path.join(reports_dir, f"{base_name}.csv")
    
    # Load index and model
    index, names, embedder = load_index(prefix, use_gpu)
    if index is None:
        return {}
    
    doc, mapping = fitz.open(pdf_in), {}
    total_counter = 0
    same_counter = 0
    
    with open(report_csv_path, 'w', newline='') as report_csv:
        writer = csv.writer(report_csv, quoting=csv.QUOTE_ALL)
        writer.writerow(["current_name", "proposed_name", "confidence", "index_used"])
        
        for page in tqdm(doc, desc="Processing pages"):
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
                distances, idx = index.search(q_vec, k)
                
                # Calculate similarity score based on the chosen metric
                if similarity_metric == "euclidean":
                    score = 1 / (1 + distances[0][0])
                else:
                    score = distances[0][0]
                
                best_name = names[idx[0][0]]
                
                # Append score if below threshold
                if score < threshold:
                    mapping[w.field_name] = f"{best_name}_{score:.2f}"
                else:
                    mapping[w.field_name] = best_name
                
                total_counter += 1
                if current_name == mapping[w.field_name]:
                    same_counter += 1
                
                writer.writerow([current_name, mapping[w.field_name], f"{score:.4f}", prefix])
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if total_counter > 0:
        percent_same = (same_counter / total_counter) * 100
        rounded_percent_same = round(percent_same, 2)
        print(f"PDF: {base_name}.pdf Accuracy: {rounded_percent_same}% Fields: {same_counter} out of {total_counter}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Using similarity metric: {similarity_metric} with threshold: {threshold}")
    else:
        print(f"PDF: {base_name}.pdf Accuracy: 0% Fields: {same_counter} out of {total_counter}")
    
    doc.close()
    return mapping

def rename(pdf_in, pdf_out, mapping):
    doc = fitz.open(pdf_in)
    changed = 0
    for page in doc:
        for w in page.widgets():
            if w.field_name in mapping and mapping[w.field_name] != w.field_name:
                _rename_widget(doc, w, mapping[w.field_name])
                w.update()
                changed += 1
    doc.save(pdf_out)
    doc.close()
    print(f"✅  saved → {pdf_out}  (renamed {changed} widgets)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_in")
    ap.add_argument("pdf_out")
    ap.add_argument("--idx", default="widget")
    ap.add_argument("-k", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="just print mapping")
    ap.add_argument("--threshold", type=float, help="Confidence threshold for renaming")
    ap.add_argument("--similarity-metric", default="cosine", choices=list(SIMILARITY_METRICS.keys()),
                   help="Similarity metric to use (cosine, euclidean, dot, cosine_sklearn)")
    ap.add_argument("--use-gpu", action="store_true", help="Use GPU for processing")
    args = ap.parse_args()

    m = propose_names(args.pdf_in, args.idx, args.k, args.threshold, args.similarity_metric, args.use_gpu)
    if args.dry:
        print(json.dumps(m, indent=2))
        sys.exit()
    rename(args.pdf_in, args.pdf_out, m)
