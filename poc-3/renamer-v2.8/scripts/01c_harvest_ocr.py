#!/usr/bin/env python
# 01c_harvest_ocr.py
"""
Walk through every PDF in ./templates/, extract:
    widget_name,  label_text (words around it)
With OCR fallback for scanned/image PDFs.

Usage:
    python 01c_harvest_ocr.py --folder templates --out widget_catalog.json --ocr
    python 01c_harvest_ocr.py --folder templates --out widget_catalog.json --ocr --copy-ocr-pdfs ocr_needed
"""
import fitz, json, glob, os, argparse, time, sys
from datetime import datetime
from tqdm import tqdm
import re
import shutil

# Add the current directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the shared OCR utilities
try:
    # Try Tesseract first
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("INFO: Tesseract OCR available")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("WARNING: Tesseract not found, will try PaddleOCR")

try:
    import ocr_utils
    OCR_AVAILABLE = ocr_utils.OCR_AVAILABLE
    OCR_DPI = ocr_utils.OCR_DPI
    OCR_LANG = "eng" if TESSERACT_AVAILABLE else "en"  # Use eng for Tesseract, en for PaddleOCR
    MARGIN = ocr_utils.MARGIN
    extract_text_with_ocr_fallback = ocr_utils.extract_text_with_ocr_fallback
    list_available_languages = ocr_utils.list_available_languages
except ImportError as e:
    print(f"Importing ocr_utils: {str(e)}")
    print(f"Looking for ocr_utils in: {script_dir}")
    # Define fallback constants
    OCR_AVAILABLE = TESSERACT_AVAILABLE  # Use Tesseract availability
    OCR_DPI = 400
    OCR_LANG = "eng" if TESSERACT_AVAILABLE else "en"
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
            tqdm.write(f"WARNING: Error extracting text: {str(e)}")
            return ""
    
    # Fallback implementation of list_available_languages
    def list_available_languages():
        return ["OCR not available. Install paddleocr or pytesseract first."]

# Simple class to track OCR usage
class OCRDetector:
    def __init__(self):
        self.ocr_used = False
    
    def mark_ocr_used(self):
        self.ocr_used = True

# Constants
MAX_WORDS = 25       # keep context short
MIN_FONT_SIZE = 11   # minimum font size to consider as heading
SAVE_INTERVAL = 100  # save progress every N documents

def natural_sort_key(s):
    """Generate a key for natural sorting of strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def save_progress(catalog, out_json, is_final=False):
    """Save current progress to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "is_final": is_final,
        "documents": catalog
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    if is_final:
        tqdm.write(f"\nSuccessfully wrote final results to: {out_json}")
    else:
        tqdm.write(f"\nSaved progress to: {out_json}")

def get_document_heading(doc):
    """Extract heading from first page of document."""
    if len(doc) == 0:
        return ""
    
    # Get text from first page
    first_page = doc[0]
    text = first_page.get_text()
    
    # Split into lines and find first non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines[0] if lines else ""

def get_section_headings(doc):
    """Extract section headings based on font size and formatting."""
    headings = []
    
    for page_num, page in enumerate(doc):
        # Get text blocks with their font information
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                if "spans" not in line:
                    continue
                    
                # Check if any span in the line has a large font
                is_heading = False
                line_text = ""
                
                for span in line["spans"]:
                    if span["size"] >= MIN_FONT_SIZE:
                        is_heading = True
                    line_text += span["text"]
                
                if is_heading and line_text.strip():
                    headings.append({
                        "text": line_text.strip(),
                        "page": page_num + 1,
                        "font_size": max(span["size"] for span in line["spans"])
                    })
    
    return headings

def get_full_text(doc):
    """Extract all text from document."""
    full_text = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            full_text.append(text.strip())
    return "\n\n".join(full_text)

def copy_ocr_pdfs(templates_folder, ocr_folder, ocr_used_files):
    """Copy PDFs that required OCR to a separate folder."""
    if not ocr_used_files:
        tqdm.write("\nNo documents required OCR assistance, nothing to copy.")
        return
        
    # Create the destination folder if it doesn't exist
    os.makedirs(ocr_folder, exist_ok=True)
    
    tqdm.write(f"\nCopying {len(ocr_used_files)} OCR-requiring PDFs to: {os.path.abspath(ocr_folder)}")
    
    # Copy each file that used OCR
    for pdf_name in ocr_used_files:
        source_path = os.path.join(templates_folder, pdf_name)
        dest_path = os.path.join(ocr_folder, pdf_name)
        
        try:
            shutil.copy2(source_path, dest_path)
        except Exception as e:
            tqdm.write(f"Error copying {pdf_name}: {e}")
    
    tqdm.write(f"Successfully copied {len(ocr_used_files)} PDFs to {ocr_folder}")
    tqdm.write("\nDocuments that required OCR:")
    for doc in ocr_used_files:
        tqdm.write(f"   - {doc}\n")

def harvest(templates_folder, out_json, use_ocr=False, ocr_language=OCR_LANG, ocr_dpi=OCR_DPI, save_debug_images=False, use_gpu=False, copy_ocr_pdfs_folder=None):
    """Process PDFs, extract widgets and context, with optional OCR."""
    if use_ocr and not OCR_AVAILABLE:
        tqdm.write("WARNING: OCR requested but Tesseract is not installed. Install with:")
        tqdm.write("  - For Amazon Linux: sudo dnf install -y tesseract tesseract-langpack-eng")
        tqdm.write("  - For Ubuntu/Debian: sudo apt-get install -y tesseract-ocr tesseract-ocr-eng")
        tqdm.write("  - For RHEL/CentOS: sudo yum install -y tesseract tesseract-langpack-eng")
        tqdm.write("WARNING: Continuing without OCR support.")
        use_ocr = False
    
    tqdm.write(f"\nScanning PDFs in: {os.path.abspath(templates_folder)}")
    tqdm.write(f"Output will be saved to: {os.path.abspath(out_json)}")
    if use_ocr:
        gpu_str = "with GPU acceleration" if use_gpu else "without GPU acceleration"
        tqdm.write(f"OCR support is enabled ({ocr_dpi} DPI, language: {ocr_language}, {gpu_str})\n")
        if save_debug_images:
            tqdm.write(f"Debug images will be saved to the temp directory\n")
        if copy_ocr_pdfs_folder:
            tqdm.write(f"OCR-requiring PDFs will be copied to: {os.path.abspath(copy_ocr_pdfs_folder)}\n")
    else:
        tqdm.write("OCR support is disabled\n")
    
    catalog = {}
    pdf_files = glob.glob(os.path.join(templates_folder, "*.pdf"))
    pdf_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    tqdm.write(f"Found {len(pdf_files)} PDF files to process")
    
    total_widgets = 0
    total_time = 0
    processed_count = 0
    ocr_used_files = []
    
    for pdf in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        doc_name = os.path.basename(pdf)
        tqdm.write(f"\nProcessing file: {doc_name}")
        doc = fitz.open(pdf)
        
        catalog[doc_name] = {
            "documentName": doc_name,
            "heading": get_document_heading(doc),
            "full_text": get_full_text(doc),
            "section_headings": get_section_headings(doc),
            "fields": []
        }
        
        doc_start_time = time.time()
        doc_widgets = 0
        doc_ocr_used = False
        
        for p in doc:
            for w in p.widgets():
                if not w.field_name:
                    continue
                r = w.rect
                
                # Track if OCR is used for this widget
                ocr_detector = OCRDetector()
                
                # Use the shared function for text extraction with OCR fallback
                text = extract_text_with_ocr_fallback(
                    p,
                    fitz.Rect(r.x0-MARGIN, r.y0-MARGIN,
                              r.x1+MARGIN, r.y1+MARGIN),
                    use_ocr=use_ocr,
                    ocr_lang=ocr_language,
                    save_debug_images=save_debug_images,
                    use_gpu=use_gpu,
                    ocr_detector=ocr_detector
                )
                
                # Check if OCR was used
                if ocr_detector.ocr_used:
                    doc_ocr_used = True
                
                # Limit text length if needed
                if text and len(text) > MAX_WORDS * 20:
                    text = text[:MAX_WORDS * 20]  # crude truncate
                
                if text.strip():
                    catalog[doc_name]["fields"].append({
                        "widgetName": w.field_name,
                        "context": text,
                        "page": p.number + 1,
                        "ocr_used": ocr_detector.ocr_used
                    })
                    doc_widgets += 1
        
        doc_time = time.time() - doc_start_time
        total_time += doc_time
        total_widgets += doc_widgets
        processed_count += 1
        
        if doc_ocr_used:
            ocr_used_files.append(doc_name)
            tqdm.write(f"OCR was used for this document! Found {doc_widgets} widgets in {doc_time:.2f} seconds")
        else:
            tqdm.write(f"Native text extraction only. Found {doc_widgets} widgets in {doc_time:.2f} seconds")
        
        # Save progress periodically
        if processed_count % SAVE_INTERVAL == 0:
            save_progress(catalog, out_json)
            tqdm.write(f"Progress: {processed_count}/{len(pdf_files)} documents processed")
            tqdm.write(f"Current widget count: {total_widgets}")
        
        doc.close()
    
    # Save final results
    save_progress(catalog, out_json, is_final=True)
    
    avg_time_per_widget = total_time / total_widgets if total_widgets > 0 else 0
    tqdm.write(f"\nFound {total_widgets} widgets across {len(catalog)} documents")
    tqdm.write(f"Total processing time: {total_time:.2f} seconds")
    tqdm.write(f"Average time per widget: {avg_time_per_widget*1000:.2f} ms")
    
    # Output summary of OCR usage
    if ocr_used_files:
        tqdm.write(f"\nOCR was used for {len(ocr_used_files)}/{len(pdf_files)} documents:")
        for file in ocr_used_files:
            tqdm.write(f"   - {file}\n")
        
        # Copy OCR-requiring PDFs if requested
        if copy_ocr_pdfs_folder:
            copy_ocr_pdfs(templates_folder, copy_ocr_pdfs_folder, ocr_used_files)
    else:
        tqdm.write(f"\nNo documents required OCR assistance")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Harvest widget contexts from PDFs with optional OCR")
    ap.add_argument("--folder", default="templates", help="Input folder containing PDFs")
    ap.add_argument("-o", "--out", default="widget_catalog.json", help="Output JSON file")
    ap.add_argument("--ocr", action="store_true", help="Enable OCR fallback for scanned documents")
    ap.add_argument("--ocr-lang", default=OCR_LANG, help=f"OCR language (default: {OCR_LANG})")
    ap.add_argument("--ocr-dpi", type=int, default=OCR_DPI, help=f"DPI for OCR processing (default: {OCR_DPI})")
    ap.add_argument("--list-langs", action="store_true", help="List available OCR languages and exit")
    ap.add_argument("--save-debug-images", action="store_true", help="Save preprocessed OCR images for debugging")
    ap.add_argument("--gpu", action="store_true", help="Use GPU acceleration for OCR if available")
    ap.add_argument("--copy-ocr-pdfs", metavar="FOLDER",  default="hard_ocr_needed", help="Copy PDFs that required OCR to this folder")
    args = ap.parse_args()

    # If requested, list available languages and exit
    if args.list_langs:
        langs = list_available_languages()
        print("Available OCR languages:")
        for lang in langs:
            print(f"  {lang}")
        sys.exit(0)

    harvest(templates_folder=args.folder, out_json=args.out, use_ocr=args.ocr,
            ocr_language=args.ocr_lang, ocr_dpi=args.ocr_dpi, 
            save_debug_images=args.save_debug_images, use_gpu=args.gpu,
            copy_ocr_pdfs_folder=args.copy_ocr_pdfs)