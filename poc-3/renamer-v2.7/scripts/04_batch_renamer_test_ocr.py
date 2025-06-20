#!/usr/bin/env python
# 04_batch_renamer_test_ocr.py
"""
Process multiple PDFs and test the accuracy of widget renaming with OCR support.

Usage:
    python 04_batch_renamer_test_ocr.py --folder_in test_pdfs
    python 04_batch_renamer_test_ocr.py --folder_in test_pdfs --ocr --copy-ocr-pdfs ocr_needed
"""
import fitz, faiss, pickle, argparse, json, sys, os, glob, io
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import time
import csv  # Add csv module for proper CSV handling
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import shutil

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

def load_index(prefix="widget"):
    try:
        # Load the index and metadata
        index_obj = faiss.read_index(f"{prefix}.faiss")
        metadata = pickle.load(open(f"{prefix}_metadata.pkl", "rb"))
        
        # Load configuration
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

def copy_ocr_pdfs(source_folder, ocr_folder, ocr_files):
    """Copy PDFs that required OCR to a separate folder."""
    if not ocr_files:
        tqdm.write("\nNo documents required OCR assistance, nothing to copy.")
        return
        
    # Create the destination folder if it doesn't exist
    os.makedirs(ocr_folder, exist_ok=True)
    
    tqdm.write(f"\nCopying {len(ocr_files)} OCR-requiring PDFs to: {os.path.abspath(ocr_folder)}")
    
    # Copy each file that used OCR
    for pdf_name in ocr_files:
        source_path = os.path.join(source_folder, pdf_name)
        dest_path = os.path.join(ocr_folder, pdf_name)
        
        try:
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            tqdm.write(f"Error copying {pdf_name}: {e}")
    
    tqdm.write(f"Successfully copied {len(ocr_files)} PDFs to {ocr_folder}")
    tqdm.write("\nDocuments that required OCR:")
    for doc in ocr_files:
        tqdm.write(f"   - {doc}\n")

def propose_names(pdf_in, loaded_indexes, device, k=1, threshold=None, similarity_metric="cosine", 
                 use_ocr=False, ocr_language=OCR_LANG, save_debug_images=False, use_gpu=False):
    """Process a PDF and propose new names for form fields based on surrounding text."""
    if use_ocr and not OCR_AVAILABLE:
        tqdm.write("WARNING: OCR requested but PaddleOCR is not installed. Install with: pip install paddleocr")
        tqdm.write("WARNING: Continuing without OCR support.")
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
    
    tqdm.write(f"\nProcessing file: {base_name}")
    
    with open(report_csv_path, 'w', newline='') as report_csv:
        writer = csv.writer(report_csv, quoting=csv.QUOTE_ALL)  # Quote all fields
        writer.writerow(["current_name", "proposed_name", "confidence", "index_used", "context_match", "document_name", "section_context", "ocr_used"])
        
        for page in doc:
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
                        best_metadata = metadata[idx[0][0]]
                        best_name = best_metadata["widgetName"]
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
                
                # Add to the mapping
                mapping[current_name] = best_name
    
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
    
    return mapping, ocr_counter > 0

# Helper function for natural sorting of filenames
def natural_sort_key(filename):
    """Generate a key for natural sorting of strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test widget naming accuracy using vector similarity matching")
    ap.add_argument("--folder_in", required=True, help="Folder containing PDFs to process")
    ap.add_argument("-k", type=int, default=1, help="Number of nearest neighbors to retrieve (default: 1)")
    ap.add_argument("--metric", choices=SIMILARITY_METRICS.keys(), default="cosine",
                    help="Similarity metric to use (default: cosine)")
    ap.add_argument("--threshold", type=float, 
                    help=f"Minimum similarity threshold (defaults based on metric: cosine={DEFAULT_THRESHOLDS['cosine']}, euclidean={DEFAULT_THRESHOLDS['euclidean']})")
    ap.add_argument("--gpu", action="store_true", help="Use GPU for FAISS and OCR if available")
    ap.add_argument("--ocr", action="store_true", help="Enable OCR fallback for scanned documents")
    ap.add_argument("--ocr-lang", default=OCR_LANG, help=f"OCR language (default: {OCR_LANG})")
    ap.add_argument("--save-debug-images", action="store_true", help="Save preprocessed OCR images for debugging")
    ap.add_argument("--list-langs", action="store_true", help="List available OCR languages and exit")
    ap.add_argument("--skip-existing", action="store_true", help="Skip PDFs that already have reports")
    ap.add_argument("--similarity", default="cosine", help="Alias for --metric for backward compatibility")
    ap.add_argument("--copy-ocr-pdfs", metavar="FOLDER", default="hard_ocr_needed", help="Copy PDFs that required OCR to this folder")
    args = ap.parse_args()

    # If requested, list available languages and exit
    if args.list_langs:
        langs = list_available_languages()
        print("Available OCR languages:")
        for lang in langs:
            print(f"  {lang}")
        sys.exit(0)

    # Check for backward compatibility
    if args.similarity != "cosine" and args.metric == "cosine":
        args.metric = args.similarity

    # Load the pre-computed indexes
    loaded_indexes, device = load_all_indexes(args.gpu)
    if loaded_indexes is None:
        sys.exit(1)
        
    # Get list of PDF files and sort them naturally
    pdf_files = [f for f in os.listdir(args.folder_in) if f.lower().endswith('.pdf')]
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
    
    # Statistics to track across all files
    total_widgets = 0
    ocr_widgets = 0
    ocr_files = []
    total_processing_time = 0
    
    # Process files with progress bar
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_in = os.path.join(args.folder_in, pdf)
        
        # Track widget count before processing this file
        previous_widget_count = total_widgets
        
        # Process the PDF
        start_time = time.time()
        mapping, ocr_used_in_pdf = propose_names(
            pdf_in, 
            loaded_indexes, 
            device, 
            k=args.k, 
            threshold=args.threshold, 
            similarity_metric=args.metric,
            use_ocr=args.ocr,
            ocr_language=args.ocr_lang,
            save_debug_images=args.save_debug_images,
            use_gpu=args.gpu
        )
        file_time = time.time() - start_time
        total_processing_time += file_time
        
        # Read the CSV to count widgets and check OCR usage
        csv_path = os.path.join(reports_dir, f"{os.path.splitext(os.path.basename(pdf_in))[0]}.csv")
        file_widget_count = 0
        file_ocr_count = 0
        
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    file_widget_count += 1
                    if len(row) >= 8 and row[7] == "Yes":  # Check OCR usage column
                        file_ocr_count += 1
            
            # Update stats
            total_widgets += file_widget_count
            ocr_widgets += file_ocr_count
            if file_ocr_count > 0:
                ocr_files.append(pdf)
    
    # Print summary of all files
    tqdm.write("\n" + "="*80)
    tqdm.write(f"SUMMARY: Processed {len(pdf_files)} PDFs with {total_widgets} total widgets")
    tqdm.write(f"Total processing time: {total_processing_time:.2f} seconds")
    
    # OCR usage statistics
    if ocr_widgets > 0:
        tqdm.write(f"\nOCR was used for {ocr_widgets}/{total_widgets} widgets ({ocr_widgets/total_widgets*100:.2f}%)")
        tqdm.write(f"OCR was needed in {len(ocr_files)}/{len(pdf_files)} files ({len(ocr_files)/len(pdf_files)*100:.2f}%)")
        if len(ocr_files) <= 10:
            tqdm.write("Files requiring OCR:")
            for file in ocr_files:
                tqdm.write(f"   - {file}\n")
                
        # Copy OCR-requiring PDFs if requested
        if args.copy_ocr_pdfs and ocr_files:
            copy_ocr_pdfs(args.folder_in, args.copy_ocr_pdfs, ocr_files)
    else:
        tqdm.write("\nNo widgets required OCR assistance")
