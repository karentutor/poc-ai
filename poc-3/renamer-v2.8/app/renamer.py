#!/usr/bin/env python
# app/renamer.py
import fitz, faiss, pickle, json, sys, os, glob, io
from sentence_transformers import SentenceTransformer
import torch
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Add scripts directory to Python path
scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.append(scripts_dir)

# Import the shared OCR utilities
try:
    import ocr_utils
    OCR_AVAILABLE = ocr_utils.OCR_AVAILABLE
    OCR_DPI = ocr_utils.OCR_DPI
    OCR_LANG = ocr_utils.OCR_LANG
    MARGIN = ocr_utils.MARGIN
    extract_text_with_ocr_fallback = ocr_utils.extract_text_with_ocr_fallback
    list_available_languages = ocr_utils.list_available_languages
except ImportError as e:
    print(f"Error importing ocr_utils: {e}")
    print(f"Looking for ocr_utils.py in: {scripts_dir}")
    # Define fallback constants
    OCR_AVAILABLE = False
    OCR_DPI = 400
    OCR_LANG = "en"
    MARGIN = 30
    
    # Fallback implementation of extract_text_with_ocr_fallback
    def extract_text_with_ocr_fallback(page, rect, use_ocr=False, ocr_lang=OCR_LANG, save_debug_images=False, use_gpu=False, ocr_detector=None):
        """Fallback implementation that only uses PyMuPDF's text extraction."""
        try:
            # Try to extract text using PyMuPDF's built-in text extraction
            try:
                words = page.get_text("words")
            except (TypeError, AttributeError):
                # For older PyMuPDF versions
                words = page.getText("words")
            
            # If words is None or empty, try alternative methods
            if not words:
                try:
                    text = page.get_text()
                    return text.strip()
                except (TypeError, AttributeError):
                    text = page.getText()
                    return text.strip()
            
            # Convert the rect to a compatible format if needed
            if not isinstance(rect, fitz.Rect):
                try:
                    rect = fitz.Rect(rect)
                except:
                    # If conversion fails, use the page rect
                    rect = page.rect
            
            # Filter words that intersect with the provided rectangle
            filtered_words = []
            for w in words:
                try:
                    # Try accessing by index (newer versions)
                    if fitz.Rect(w[:4]).intersects(rect):
                        filtered_words.append(w)
                except (TypeError, IndexError):
                    # Try accessing by word attributes (older versions)
                    if hasattr(w, 'bbox') and fitz.Rect(w.bbox).intersects(rect):
                        filtered_words.append(w)
                    elif hasattr(w, 'rect') and w.rect.intersects(rect):
                        filtered_words.append(w)
            
            # Format the text from the words
            text = ""
            for w in filtered_words:
                try:
                    # Try accessing by index (newer versions)
                    text += w[4] + " "
                except (TypeError, IndexError):
                    # Try accessing by word attributes (older versions)
                    if hasattr(w, 'text'):
                        text += w.text + " "
                
            return text.strip()
            
        except Exception as e:
            print(f"WARNING: Error extracting text: {str(e)}")
            return ""
    
    # Fallback implementation of list_available_languages
    def list_available_languages():
        return ["OCR not available. Install paddleocr or pytesseract first."]

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
    "cosine": 0.5,  # Higher is better
    "euclidean": 0.3,  # Higher is better
    "dot": 0.5,  # Higher is better
    "cosine_sklearn": 0.7  # Higher is better
}

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
                print(f"Warning: Could not load metadata for {prefix}")
                return None, None, None
        
        # Load configuration to get model name
        config_filename = f"{prefix}_config.json"
        try:
            with open(config_filename, "r") as cf:
                config = json.load(cf)
            model_name = config.get("model", MODEL)
        except Exception as e:
            print(f"No config for {prefix} or error loading config: {e}. Using default model.")
            model_name = MODEL
            
        return index_obj, metadata, model_name
    except (FileNotFoundError, IOError) as e:
        print(f"Warning: Could not load index {prefix}: {e}")
        return None, None, None

def load_all_indexes(gpu=False):
    # Check if GPU is available when requested
    device = None
    if gpu:
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU requested but not available, falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"
        print("Using CPU for processing")

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
                print(f"Error loading model {model_name} for {prefix}, using default: {e}")
                embedder_for_index = embedder
                
            # Configure GPU resources for FAISS if available and requested
            if gpu and device == "cuda":
                try:
                    # Try to move the index to GPU if supported by the index type
                    if hasattr(index_obj, 'getDevices') and index_obj.ntotal > 0:
                        res = faiss.StandardGpuResources()
                        index_obj = faiss.index_cpu_to_gpu(res, 0, index_obj)
                        print(f"Successfully moved index {prefix} to GPU")
                except Exception as e:
                    print(f"Could not move index {prefix} to GPU: {str(e)}")
                    
            loaded_indexes.append((prefix, index_obj, metadata, embedder_for_index))
    
    if not loaded_indexes:
        print("Error: No valid indexes could be loaded.")
        return None, None
        
    return loaded_indexes, device

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

def rename_pdf(binary: bytes, k: int = 1, threshold: float = None, 
              similarity_metric: str = "cosine", use_ocr: bool = False, 
              ocr_language: str = OCR_LANG, use_gpu: bool = False) -> tuple[bytes, dict]:
    """
    Rename PDF form fields using vector similarity matching with OCR support.
    
    Args:
        binary: PDF file contents as bytes
        k: Number of nearest neighbors to retrieve
        threshold: Confidence threshold for renaming
        similarity_metric: Similarity metric to use
        use_ocr: Enable OCR fallback for scanned documents
        ocr_language: OCR language to use
        use_gpu: Use GPU for processing
        
    Returns:
        tuple: (processed_pdf_bytes, stats_dict)
    """
    # Load all available indexes
    loaded_indexes, device = load_all_indexes(use_gpu)
    if loaded_indexes is None:
        raise ValueError("No valid indexes could be loaded. Please run the indexing script first.")
    
    # Set default threshold based on similarity metric if not provided
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS[similarity_metric]
    
    # Check OCR availability
    if use_ocr and not OCR_AVAILABLE:
        print("WARNING: OCR requested but OCR is not available. Continuing without OCR support.")
        use_ocr = False
    
    doc = fitz.open(stream=binary, filetype="pdf")
    total_widgets = 0
    changed_widgets = 0
    ocr_counter = 0
    index_usage_counts = {prefix: 0 for prefix, _, _, _ in loaded_indexes}
    mapping_info = []
    
    for page in doc:
        for w in page.widgets():
            current_name = w.field_name
            
            # Extract context with OCR support if enabled
            ctx = extract_text_with_ocr_fallback(
                page,
                fitz.Rect(w.rect.x0-MARGIN, w.rect.y0-MARGIN,
                          w.rect.x1+MARGIN, w.rect.y1+MARGIN),
                use_ocr=use_ocr,
                ocr_lang=ocr_language,
                save_debug_images=False,
                use_gpu=use_gpu
            )
            
            if not ctx.strip():
                continue
            
            best_score = float('-inf')
            best_name = None
            best_index = None
            best_metadata = None
            
            # Try each index
            for prefix, index_obj, metadata, index_embedder in loaded_indexes:
                q_vec = index_embedder.encode([ctx], convert_to_numpy=True).astype("float32")
                if q_vec.shape[1] != index_obj.d:
                    print(f"Skipping index {prefix}: Query vector dimension {q_vec.shape[1]} does not match index dimension {index_obj.d}")
                    continue
                
                distances, idx = index_obj.search(q_vec, k)
                
                # Calculate similarity score based on the chosen metric
                if similarity_metric == "euclidean":
                    score = float(1 / (1 + distances[0][0]))  # Convert to Python float
                else:
                    score = float(distances[0][0])  # Convert to Python float
                
                if score > best_score:
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
            total_widgets += 1
            if best_name == current_name:
                unchanged_widgets = total_widgets - changed_widgets
            else:
                changed_widgets += 1
            
            if best_index != "None":
                index_usage_counts[best_index] += 1
            
            # Apply the threshold
            # if best_score < threshold:
            #     print(f"WARNING: Score {best_score:.4f} below threshold {threshold} for {current_name}. Keeping original name.")
            #     best_name = current_name
            
            # Get context data for this match
            context_match = best_metadata.get("context", "") if best_metadata else ""
            doc_name = best_metadata.get("documentName", "") if best_metadata else ""
            
            # Add to mapping info
            mapping_info.append({
                "page": page.number + 1,
                "original_name": current_name,
                "new_name": best_name,
                "confidence": float(best_score) if best_score != float('-inf') else None,  # Convert to Python float
                "index_used": best_index,
                "context_match": context_match,
                "document_name": doc_name,
                "section_context": ctx,
                "context": ctx
            })
            
            # Rename if different
            if best_name != current_name:
                _rename_widget(doc, w, best_name)
                w.update()
    
    # Calculate statistics
    unchanged_widgets = total_widgets - changed_widgets
    accuracy_rate = (unchanged_widgets / total_widgets * 100) if total_widgets > 0 else 0
    
    # Save the modified PDF
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    
    # Prepare statistics - ensure all values are JSON serializable
    stats = {
        "mapping_info": mapping_info,
        "total_widgets": total_widgets,
        "changed_widgets": changed_widgets,
        "unchanged_widgets": unchanged_widgets,
        "accuracy": round(float(accuracy_rate), 2),  # Convert to Python float
        "index_usage": {str(k): int(v) for k, v in index_usage_counts.items()}  # Convert to string keys and int values
    }
    
    return buf.read(), stats

# Simple class to track OCR usage
class OCRDetector:
    def __init__(self):
        self.ocr_used = False
    
    def mark_ocr_used(self):
        self.ocr_used = True
