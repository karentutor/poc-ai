#!/usr/bin/env python
# 03_rename_from_index.py
import fitz, faiss, pickle, argparse, json, sys, os, csv, glob
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import time
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import the shared OCR utilities
try:
    from ocr_utils import (
        OCR_AVAILABLE, OCR_DPI, OCR_LANG, MARGIN,
        extract_text_with_ocr_fallback, list_available_languages
    )
except ImportError:
    # If import fails, check if the module is in the parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from ocr_utils import (
            OCR_AVAILABLE, OCR_DPI, OCR_LANG, MARGIN,
            extract_text_with_ocr_fallback, list_available_languages
        )
    except ImportError:
        tqdm.write("WARNING: Could not import ocr_utils module. OCR functionality will be limited.")
        # Define fallback imports and constants
        from PIL import Image
        import cv2
        OCR_AVAILABLE = False
        OCR_DPI = 400
        OCR_LANG = "en"
        MARGIN = 30

# Simple class to track OCR usage
class OCRDetector:
    def __init__(self):
        self.ocr_used = False
    
    def mark_ocr_used(self):
        self.ocr_used = True

# Model to use if none specified
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

def load_index(prefix="widget"):
    try:
        # Load the index and metadata
        index_obj = faiss.read_index(f"{prefix}.faiss")
        
        # Try to load metadata from different file formats
        try:
            # First try the names file (older format)
            metadata = pickle.load(open(f"{prefix}_names.pkl", "rb"))
            # Convert to new format if it's just a list
            if isinstance(metadata, list):
                metadata = [{"widgetName": name} for name in metadata]
        except (FileNotFoundError, IOError):
            # Then try the metadata file (newer format)
            try:
                metadata = pickle.load(open(f"{prefix}_metadata.pkl", "rb"))
            except (FileNotFoundError, IOError):
                tqdm.write(f"Warning: Could not load metadata for {prefix}")
                return None, None, None
        
        # Load configuration to get model name
        config_filename = f"{prefix}_config.json"
        try:
            with open(config_filename, "r") as cf:
                config = json.load(cf)
            model_name = config.get("model", MODEL)
        except Exception as e:
            tqdm.write(f"No config for {prefix} or error loading config: {e}. Using default model.")
            model_name = MODEL
            
        return index_obj, metadata, model_name
    except (FileNotFoundError, IOError) as e:
        tqdm.write(f"Warning: Could not load index {prefix}: {e}")
        return None, None, None

def load_all_indexes(gpu=False):
    # Check if GPU is available when requested
    device = None
    if gpu:
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
        index_obj, metadata, model_name = load_index(prefix)
        if index_obj is not None:
            # Create embedder for this index
            try:
                embedder_for_index = SentenceTransformer(model_name, device=device)
            except Exception as e:
                tqdm.write(f"Error loading model {model_name} for {prefix}, using default: {e}")
                embedder_for_index = embedder
                
            # Configure GPU resources for FAISS if available and requested
            if gpu and device == "cuda":
                try:
                    # Try to move the index to GPU if supported by the index type
                    if hasattr(index_obj, 'getDevices') and index_obj.ntotal > 0:
                        res = faiss.StandardGpuResources()
                        index_obj = faiss.index_cpu_to_gpu(res, 0, index_obj)
                        tqdm.write(f"Successfully moved index {prefix} to GPU")
                except Exception as e:
                    tqdm.write(f"Could not move index {prefix} to GPU: {str(e)}")
                    
            loaded_indexes.append((prefix, index_obj, metadata, embedder_for_index))
    
    if not loaded_indexes:
        tqdm.write("Error: No valid indexes could be loaded. Exiting.")
        return None, None
        
    return loaded_indexes, device

def propose_names(pdf_in, loaded_indexes, device, k=1, threshold=None, similarity_metric="cosine", 
                 use_ocr=False, ocr_language=OCR_LANG, save_debug_images=False, use_gpu=False):
    """Process a PDF and propose new names for form fields based on surrounding text with OCR support."""
    if use_ocr and not OCR_AVAILABLE:
        tqdm.write("WARNING: OCR requested but OCR is not available. Continuing without OCR support.")
        use_ocr = False
        
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
    ocr_counter = 0
    index_usage_counts = {prefix: 0 for prefix, _, _, _ in loaded_indexes}
    
    tqdm.write(f"Processing file: {base_name}")
    
    with open(report_csv_path, 'w', newline='') as report_csv:
        writer = csv.writer(report_csv, quoting=csv.QUOTE_ALL)
        writer.writerow(["current_name", "proposed_name", "confidence", "index_used", "context_match", "document_name", "section_context", "ocr_used"])
        
        for page in tqdm(doc, desc="Processing pages"):
            for w in page.widgets():
                current_name = w.field_name
                
                # Track if OCR is used for this context
                ocr_detector = OCRDetector() if 'OCRDetector' in globals() else None
                
                # Use the shared function for text extraction with OCR fallback
                ctx = extract_text_with_ocr_fallback(
                    page,
                    fitz.Rect(w.rect.x0-MARGIN, w.rect.y0-MARGIN,
                              w.rect.x1+MARGIN, w.rect.y1+MARGIN),
                    use_ocr=use_ocr,
                    ocr_lang=ocr_language,
                    save_debug_images=save_debug_images,
                    use_gpu=use_gpu,
                    ocr_detector=ocr_detector
                )
                
                # Track OCR usage
                ocr_used = ocr_detector.ocr_used if ocr_detector else False
                if ocr_used:
                    ocr_counter += 1
                
                if not ctx.strip():
                    continue
                
                best_score = float('-inf')  # Initialize with negative infinity for similarity scores
                best_name = None
                best_index = None
                best_metadata = None
                
                # Iterate through each index and generate a query vector using its specific embedder
                for prefix, index_obj, metadata, index_embedder in loaded_indexes:
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
                        if isinstance(metadata[idx[0][0]], dict):
                            best_metadata = metadata[idx[0][0]]
                            best_name = best_metadata.get("widgetName", f"widget_{idx[0][0]}")
                        else:
                            best_metadata = {"widgetName": metadata[idx[0][0]]}
                            best_name = metadata[idx[0][0]]
                        best_index = prefix
                
                if best_name is None:
                    best_name = current_name
                    best_score = float('-inf')
                    best_index = "None"
                
                # Track statistics
                total_counter += 1
                if best_name == current_name:
                    same_counter += 1
                    
                if best_index != "None":
                    index_usage_counts[best_index] += 1
                
                # Apply the threshold
                if best_score < threshold:
                    tqdm.write(f"WARNING: Score {best_score:.4f} below threshold {threshold} for {current_name}. Keeping original name.")
                    best_name = current_name
                
                # Get context data for this match
                context_match = best_metadata.get("context", "") if best_metadata else ""
                doc_name = best_metadata.get("documentName", "") if best_metadata else ""
                
                # Add to the mapping
                mapping[current_name] = best_name
                
                # Write to the report
                writer.writerow([
                    current_name, 
                    best_name, 
                    f"{best_score:.4f}" if best_score != float('-inf') else "N/A",
                    best_index,
                    context_match[:100] + "..." if len(context_match) > 100 else context_match,
                    doc_name,
                    ctx[:100] + "..." if len(ctx) > 100 else ctx,
                    "Yes" if ocr_used else "No"
                ])
    
    # Calculate statistics
    elapsed = time.time() - start_time
    match_rate = same_counter / total_counter if total_counter > 0 else 0
    per_field_time = elapsed / total_counter if total_counter > 0 else 0
    ocr_rate = ocr_counter / total_counter if total_counter > 0 else 0
    
    # Print summary
    tqdm.write(f"\nProcessed {len(doc)} pages and {total_counter} form fields in {elapsed:.2f} seconds")
    tqdm.write(f"Processing time per field: {per_field_time*1000:.2f} ms")
    tqdm.write(f"==> Match rate (current = proposed): {match_rate*100:.2f}%")
    
    # Report OCR usage
    if ocr_counter > 0:
        tqdm.write(f"OCR was used for {ocr_counter} fields ({ocr_rate*100:.2f}% of total)")
    else:
        tqdm.write(f"No fields required OCR assistance")
    
    tqdm.write(f"Results saved to: {report_csv_path}")
    
    # Print index usage
    tqdm.write("\nIndex Usage Statistics:")
    for idx, count in sorted(index_usage_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percent = (count / total_counter) * 100 if total_counter > 0 else 0
            tqdm.write(f"   {idx}: {count} fields ({percent:.2f}%)")
    
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
    print(f"Successfully saved → {pdf_out}  (renamed {changed} widgets)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rename PDF form fields using vector similarity matching with OCR support")
    ap.add_argument("pdf_in", help="Input PDF file")
    ap.add_argument("pdf_out", help="Output PDF file")
    ap.add_argument("--idx", help="Index prefix to use (deprecated, auto-loads all indexes now)", default="widget")
    ap.add_argument("-k", type=int, default=1, help="Number of nearest neighbors to retrieve (default: 1)")
    ap.add_argument("--dry", action="store_true", help="Just print mapping without saving")
    ap.add_argument("--threshold", type=float, help="Confidence threshold for renaming")
    ap.add_argument("--similarity", default="cosine", choices=list(SIMILARITY_METRICS.keys()),
                   help="Similarity metric to use (cosine, euclidean, dot, cosine_sklearn)")
    ap.add_argument("--gpu", action="store_true", help="Use GPU for processing")
    ap.add_argument("--ocr", action="store_true", help="Enable OCR fallback for scanned documents")
    ap.add_argument("--ocr-lang", default=OCR_LANG, help=f"OCR language (default: {OCR_LANG})")
    ap.add_argument("--save-debug-images", action="store_true", help="Save preprocessed OCR images for debugging")
    ap.add_argument("--list-langs", action="store_true", help="List available OCR languages and exit")
    args = ap.parse_args()

    # If requested, list available languages and exit
    if args.list_langs:
        langs = list_available_languages()
        print("Available OCR languages:")
        for lang in langs:
            print(f"  {lang}")
        sys.exit(0)

    # Show OCR status
    if args.ocr:
        if OCR_AVAILABLE:
            tqdm.write("OCR support is enabled for scanned documents")
        else:
            tqdm.write("WARNING: OCR was requested but dependencies are not available")
            tqdm.write("WARNING: Install OCR dependencies first")
            args.ocr = False
            
    # Load all available indexes
    loaded_indexes, device = load_all_indexes(args.gpu)
    if loaded_indexes is None:
        sys.exit(1)

    # Process the PDF with all loaded indexes
    m = propose_names(
        args.pdf_in, 
        loaded_indexes, 
        device, 
        k=args.k, 
        threshold=args.threshold, 
        similarity_metric=args.similarity,
        use_ocr=args.ocr,
        ocr_language=args.ocr_lang,
        save_debug_images=args.save_debug_images,
        use_gpu=args.gpu
    )
    
    if args.dry:
        print(json.dumps(m, indent=2))
        sys.exit()
        
    rename(args.pdf_in, args.pdf_out, m)
