#!/usr/bin/env python
# 03_rename_from_index.py
import fitz, faiss, pickle, argparse, json, sys, os, glob
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import time
import csv  # Add csv module for proper CSV handling
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MARGIN = 30

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

def load_index(prefix="widget"):
    try:
        return (
            faiss.read_index(f"{prefix}.faiss"),
            pickle.load(open(f"{prefix}_names.pkl", "rb"))
        )
    except (FileNotFoundError, IOError) as e:
        tqdm.write(f"Warning: Could not load index {prefix}: {e}")
        return None, None

def load_all_indexes(use_gpu=False):
    # Check if GPU is available when requested
    device = None
    if use_gpu:
        if torch.cuda.is_available():
            device = "cuda"
            tqdm.write(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            tqdm.write("GPU requested but not available, falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"
        tqdm.write("Using CPU for processing")

    # Global embedder as fallback
    embedder = SentenceTransformer(MODEL, device=device)
    
    # Automatically load all indexes by scanning for files ending with '.faiss'
    faiss_files = glob.glob("*.faiss")
    loaded_indexes = []
    for faiss_file in faiss_files:
        prefix = faiss_file[:-6]  # Remove the '.faiss' extension
        index_obj, names = load_index(prefix)
        if index_obj is not None:
            # Attempt to load configuration file for this index
            config_filename = f"{prefix}_config.json"
            try:
                with open(config_filename, "r") as cf:
                    config = json.load(cf)
                model_name = config.get("model", None)
                if model_name:
                    embedder_for_index = SentenceTransformer(model_name, device=device)
                else:
                    embedder_for_index = embedder
            except Exception as e:
                tqdm.write(f"No config for {prefix} or error loading config: {e}. Using default embedder.")
                embedder_for_index = embedder
                
            # Configure GPU resources for FAISS if available and requested
            if use_gpu and device == "cuda":
                try:
                    # Try to move the index to GPU if supported by the index type
                    if hasattr(index_obj, 'getDevices') and index_obj.ntotal > 0:
                        res = faiss.StandardGpuResources()
                        index_obj = faiss.index_cpu_to_gpu(res, 0, index_obj)
                        tqdm.write(f"Successfully moved index {prefix} to GPU")
                except Exception as e:
                    tqdm.write(f"Could not move index {prefix} to GPU: {str(e)}")
                    
            loaded_indexes.append((prefix, index_obj, names, embedder_for_index))
    
    if not loaded_indexes:
        tqdm.write("Error: No valid indexes could be loaded. Exiting.")
        return None, None
        
    return loaded_indexes, device

def surrounding_text(page, rect):
    words = page.get_text("words")
    return " ".join(w[4] for w in words if fitz.Rect(w[:4]).intersects(rect))

def propose_names(pdf_in, loaded_indexes, device, k=1, threshold=None, similarity_metric="cosine"):
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
    
    doc, mapping = fitz.open(pdf_in), {}
    total_counter = 0
    same_counter = 0
    index_usage_counts = {prefix: 0 for prefix, _, _, _ in loaded_indexes}
    
    with open(report_csv_path, 'w', newline='') as report_csv:
        writer = csv.writer(report_csv, quoting=csv.QUOTE_ALL)  # Quote all fields
        writer.writerow(["current_name", "proposed_name", "confidence", "index_used"])
        
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
                
                best_score = float('-inf')  # Initialize with negative infinity for similarity scores
                best_name = None
                best_index = None
                
                # Iterate through each index and generate a query vector using its specific embedder
                for prefix, index_obj, names, index_embedder in loaded_indexes:
                    q_vec = index_embedder.encode([ctx], convert_to_numpy=True).astype("float32")
                    # Check if query vector dimension matches the index dimension
                    if q_vec.shape[1] != index_obj.d:
                        tqdm.write(f"Skipping index {prefix}: Query vector dimension {q_vec.shape[1]} does not match index dimension {index_obj.d}")
                        continue
                    
                    # Get the nearest neighbors
                    distances, idx = index_obj.search(q_vec, k)
                    
                    # Calculate similarity score based on the chosen metric
                    if similarity_metric == "euclidean":
                        # For Euclidean, we need to convert distance to similarity
                        score = 1 / (1 + distances[0][0])
                    else:
                        # For other metrics, we use the distance directly
                        score = distances[0][0]
                    
                    if score > best_score:  # Higher score is better for all metrics
                        best_score = score
                        best_name = names[idx[0][0]]
                        best_index = prefix
                
                if best_name is None:
                    best_name = current_name
                    best_score = float('-inf')
                    best_index = "None"
                if best_index not in index_usage_counts:
                    index_usage_counts[best_index] = 0
                
                # Update the usage count for the selected index
                index_usage_counts[best_index] += 1
                
                # Append score if below threshold (for all metrics, higher score is better)
                if best_score < threshold:
                    mapping[w.field_name] = f"{best_name}_{best_score:.2f}"
                else:
                    mapping[w.field_name] = best_name
                
                total_counter += 1
                if current_name == mapping[w.field_name]:
                    same_counter += 1
                
                writer.writerow([current_name, mapping[w.field_name], f"{best_score:.4f}", best_index])
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if total_counter > 0:
        percent_same = (same_counter / total_counter) * 100
        rounded_percent_same = round(percent_same, 2)
        
        # Create index usage summary
        index_usage_summary = ", ".join([f"{prefix}: {count}" for prefix, count in index_usage_counts.items()])
        
        tqdm.write(f"PDF: {base_name}.pdf Accuracy: {rounded_percent_same}% Fields: {same_counter} out of {total_counter}")
        tqdm.write(f"Index usage: {index_usage_summary}")
        tqdm.write(f"Processing time: {processing_time:.2f} seconds")
        tqdm.write(f"Using similarity metric: {similarity_metric} with threshold: {threshold}")
    else:
        tqdm.write(f"PDF: {base_name}.pdf Accuracy: 0% Fields: {same_counter} out of {total_counter}")
    doc.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder_in", required=True, help="Folder containing PDFs to process")
    ap.add_argument("--idx", help="Deprecated. This argument is ignored since indexes are auto-loaded.", default="widget")
    ap.add_argument("-k", type=int, default=1, help="Number of nearest neighbors to retrieve")
    ap.add_argument("--threshold", type=float, help="Similarity threshold (higher is better)")
    ap.add_argument("--similarity", default="cosine", choices=list(SIMILARITY_METRICS.keys()),
                    help="Similarity metric to use (cosine, euclidean, dot, cosine_sklearn)")
    ap.add_argument("--dry", action="store_true", help="Just print mapping without saving")
    ap.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available", default=True)
    ap.add_argument("--skip-existing", action="store_true", help="Skip PDFs that already have reports")
    args = ap.parse_args()

    # Get list of PDF files and sort them numerically
    pdf_files = [f for f in os.listdir(args.folder_in) if f.lower().endswith('.pdf')]
    
    def natural_sort_key(filename):
        name = os.path.splitext(filename)[0]
        try:
            return int(name)
        except ValueError:
            return float('inf')  # Put non-numeric names at the end
    
    pdf_files.sort(key=natural_sort_key)
    
    # Create reports directory if it doesn't exist
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Filter out PDFs that already have reports if skip-existing is enabled
    if args.skip_existing:
        pdf_files = [
            pdf for pdf in pdf_files 
            if not os.path.exists(os.path.join(reports_dir, f"{os.path.splitext(pdf)[0]}.csv"))
        ]
        if not pdf_files:
            print("All PDFs already have reports. Nothing to process.")
            sys.exit(0)
    
    # Load all indexes once before processing PDFs
    loaded_indexes, device = load_all_indexes(args.gpu)
    if loaded_indexes is None:
        sys.exit(1)
    
    # Process files with progress bar
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_in = os.path.join(args.folder_in, pdf)
        propose_names(pdf_in, loaded_indexes, device, args.k, args.threshold, args.similarity)
