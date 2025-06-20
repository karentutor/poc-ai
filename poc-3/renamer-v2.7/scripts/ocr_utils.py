#!/usr/bin/env python
# ocr_utils.py
"""
Shared OCR utility functions for PDF processing.
This module provides high-quality OCR capabilities using PaddleOCR (preferred)
with fallback to Tesseract OCR if needed for backward compatibility.
"""
import os
import io
import sys
import tempfile
import re
import time
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Try to import fitz (PyMuPDF) and check its version
try:
    import fitz
    PYMUPDF_VERSION = fitz.version
    
    # Handle different version formats
    if isinstance(PYMUPDF_VERSION, tuple):
        # New version format (tuple)
        version_str = PYMUPDF_VERSION[0]
        PYMUPDF_LEGACY = False  # Recent versions don't need legacy mode
    else:
        # Old string format
        version_str = PYMUPDF_VERSION
        version_parts = version_str.split('.')
        PYMUPDF_LEGACY = int(version_parts[0]) < 1 or (int(version_parts[0]) == 1 and int(version_parts[1]) < 18)
    
    tqdm.write(f"Using PyMuPDF version {version_str}" + (" (legacy mode)" if PYMUPDF_LEGACY else ""))
    
except ImportError:
    # If fitz is not installed, set a flag to indicate its absence
    tqdm.write("Warning: PyMuPDF (fitz) is not installed. PDF processing functions will not work.")
    PYMUPDF_VERSION = None
    PYMUPDF_LEGACY = True
except Exception as e:
    # Handle any other errors in version detection
    tqdm.write(f"Warning: Error detecting PyMuPDF version: {str(e)}. Using legacy mode.")
    PYMUPDF_VERSION = "unknown"
    PYMUPDF_LEGACY = True

# Check for OCR options - try PaddleOCR first, then fall back to Tesseract
PADDLE_OCR_AVAILABLE = False
TESSERACT_OCR_AVAILABLE = False

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR, draw_ocr
    PADDLE_OCR_AVAILABLE = True
    tqdm.write("INFO: PaddleOCR successfully loaded (preferred OCR engine)")
except ImportError:
    tqdm.write("WARNING: PaddleOCR not found. Will try Tesseract as fallback.")
    PADDLE_OCR_AVAILABLE = False

# Try to import Tesseract as fallback
try:
    import pytesseract
    TESSERACT_OCR_AVAILABLE = True
    tqdm.write("INFO: Tesseract OCR available as fallback")
    # Try to set the tesseract command path if it's not found by default
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        # Check common paths
        common_paths = [
            # Windows
            "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            # Mac Homebrew
            "/usr/local/bin/tesseract",
            # Linux
            "/usr/bin/tesseract",
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                tqdm.write(f"Set tesseract path to: {path}")
                break
except ImportError:
    TESSERACT_OCR_AVAILABLE = False

# Set OCR availability flag based on available engines
OCR_AVAILABLE = PADDLE_OCR_AVAILABLE or TESSERACT_OCR_AVAILABLE
if not OCR_AVAILABLE:
    tqdm.write("WARNING: No OCR engines available. Install PaddleOCR or Tesseract.")

# Default OCR settings
OCR_DPI = 400        # DPI for OCR processing (higher for better quality)
OCR_LANG = "eng" if TESSERACT_OCR_AVAILABLE and not PADDLE_OCR_AVAILABLE else "en"  # Default language code
MARGIN = 30          # Points around widget to grab context

# Global PaddleOCR model instances (lazy-loaded)
_ocr_models = {}

def get_paddle_ocr_model(lang="en", use_gpu=False):
    """Get or initialize a PaddleOCR model for the specified language.
    
    Args:
        lang: Language code ('en', 'ch', etc.)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        PaddleOCR model instance
    """
    if not PADDLE_OCR_AVAILABLE:
        return None
        
    # Map common Tesseract language codes to PaddleOCR codes
    lang_map = {
        "eng": "en",
        "fra": "fr",
        "deu": "german",
        "spa": "es",
        "ita": "it",
        "por": "pt",
        "rus": "ru",
        "ara": "ar",
        "hin": "hi",
        "jpn": "japan",
        "kor": "korean",
        "chi_sim": "ch",
        "chi_tra": "chinese_cht"
    }
    
    # Handle language code mapping
    if '+' in lang:
        # For multiple languages, just use English as fallback
        paddle_lang = "en"
    else:
        paddle_lang = lang_map.get(lang, lang)
    
    # Create cache key based on language and GPU usage
    key = f"{paddle_lang}_{use_gpu}"
    
    # Load model if not already cached
    if key not in _ocr_models:
        try:
            tqdm.write(f"INFO: Initializing PaddleOCR model for language: {paddle_lang}")
            # Initialize with optimized settings
            _ocr_models[key] = PaddleOCR(
                use_angle_cls=True,  # Detect text orientation
                lang=paddle_lang,    # Language model
                use_gpu=use_gpu,     # GPU acceleration if available
                show_log=False,      # Disable verbose logging
                use_mp=True,         # Use multiprocessing
                enable_mkldnn=True,  # Use Intel MKL-DNN acceleration if available
                rec_batch_num=6      # Batch size for recognition
            )
        except Exception as e:
            tqdm.write(f"WARNING: Error loading PaddleOCR model: {str(e)}")
            # Fallback to English if specified language fails
            if paddle_lang != "en":
                tqdm.write("WARNING: Falling back to English OCR model")
                return get_paddle_ocr_model("en", use_gpu)
            return None
    
    return _ocr_models[key]

def preprocess_image_for_ocr(image):
    """Apply advanced preprocessing to improve OCR accuracy.
    
    Args:
        image: Grayscale numpy array image
        
    Returns:
        List of preprocessed images with different techniques
    """
    results = []
    
    # Original grayscale
    results.append(("original", image))
    
    # 1. Noise removal with bilateral filter (preserves edges)
    try:
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        results.append(("denoised", denoised))
    except Exception:
        pass
    
    # 2. Adaptive thresholding with different parameters
    try:
        # Normal adaptive threshold
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        results.append(("adaptive", binary))
        
        # More aggressive parameters for low contrast documents
        binary_aggressive = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 7, 2
        )
        results.append(("adaptive_aggressive", binary_aggressive))
    except Exception:
        pass
    
    # 3. Otsu's thresholding
    try:
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(("otsu", otsu))
    except Exception:
        pass
    
    # 4. Contrast enhancement with CLAHE
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        results.append(("clahe", enhanced))
        
        # Apply Otsu after CLAHE for better results on low contrast images
        _, clahe_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(("clahe_otsu", clahe_otsu))
    except Exception:
        pass
    
    # 5. Edge enhancement with unsharp masking
    try:
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        unsharp = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        results.append(("unsharp", unsharp))
    except Exception:
        pass
    
    return results

def paddle_ocr_text(img, lang="en", use_gpu=False):
    """Extract text from image using PaddleOCR.
    
    Args:
        img: NumPy array image (BGR format for colored, grayscale otherwise)
        lang: Language code for OCR
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Extracted text as string
    """
    if not PADDLE_OCR_AVAILABLE:
        return ""
    
    # Get OCR model for specified language
    ocr_model = get_paddle_ocr_model(lang, use_gpu)
    if ocr_model is None:
        return ""
    
    try:
        # Ensure image is in RGB format (PaddleOCR expects RGB)
        if len(img.shape) == 2:  # Grayscale
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:  # BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Run OCR detection
        result = ocr_model.ocr(img_rgb, cls=True)
        
        # Handle different result formats (varies by version)
        if not result or len(result) == 0:
            return ""
        
        # Extract text from results
        text = ""
        
        # Handle both list and dict type results (API changed in different versions)
        # PaddleOCR can return None or empty results for some images
        try:
            if isinstance(result, list):
                # Newer versions return a list of results for each image
                if result[0] is None:
                    return ""
                    
                for line in result[0]:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        # Get text confidence pair
                        if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                            text += line[1][0] + " "
                        elif isinstance(line[1], dict) and 'text' in line[1]:
                            text += line[1]['text'] + " "
                    elif isinstance(line, dict) and 'text' in line:
                        text += line['text'] + " "
        except (TypeError, IndexError) as e:
            # Handle cases where PaddleOCR returns unexpected structure
            tqdm.write(f"WARNING: PaddleOCR result parsing error: {str(e)}")
            return ""
        
        return text.strip()
    except Exception as e:
        tqdm.write(f"WARNING: PaddleOCR error: {str(e)}")
        return ""

def tesseract_ocr_text(img, lang="eng"):
    """Extract text from image using Tesseract OCR.
    
    Args:
        img: NumPy array image
        lang: Language code(s) for Tesseract OCR
        
    Returns:
        Extracted text as string
    """
    if not TESSERACT_OCR_AVAILABLE:
        return ""
    
    try:
        # Process with multiple PSM modes to get best results
        results = []
        
        # Try different page segmentation modes
        psm_modes = [6, 3, 4, 7]  # 6=block, 3=auto, 4=single column, 7=single line
        
        for psm in psm_modes:
            try:
                text = pytesseract.image_to_string(
                    img, 
                    config=f'--psm {psm} --oem 3',
                    lang=lang
                )
                
                if text.strip():
                    words = text.strip().split()
                    results.append({
                        'text': text.strip(),
                        'method': f'psm{psm}',
                        'confidence': len(words),
                        'word_count': len(words)
                    })
            except Exception:
                pass
        
        # Select the best result based on confidence
        if results:
            # Sort by confidence score (descending)
            results.sort(key=lambda x: x['confidence'], reverse=True)
            best_result = results[0]
            return best_result['text']
        
        return ""
    except Exception as e:
        tqdm.write(f"WARNING: Tesseract OCR error: {str(e)}")
        return ""

def ocr_text(page, rect, dpi=OCR_DPI, lang=OCR_LANG, save_images=False, use_gpu=False):
    """Extract text from image using available OCR engines with advanced preprocessing.
    
    Args:
        page: PyMuPDF page object
        rect: Rectangle area to extract text from
        dpi: DPI for OCR processing (higher values = better quality but slower)
        lang: OCR language(s) to use
        save_images: Whether to save debug images of preprocessing steps
        use_gpu: Whether to use GPU acceleration (PaddleOCR only)
        
    Returns:
        Extracted text as string
    """
    if not OCR_AVAILABLE:
        return ""
    
    # Set language based on available engines
    paddle_lang = lang
    tesseract_lang = lang
    
    # Convert between language formats if needed
    if PADDLE_OCR_AVAILABLE and lang.startswith("eng"):
        paddle_lang = "en"
    elif TESSERACT_OCR_AVAILABLE and lang == "en":
        tesseract_lang = "eng"
    
    # Log which OCR engine we're using
    if PADDLE_OCR_AVAILABLE:
        engine = "PaddleOCR"
        ocr_lang = paddle_lang
    else:
        engine = "Tesseract"
        ocr_lang = tesseract_lang
        
    tqdm.write(f"INFO: Using {engine} for page {page.number + 1} with language: {ocr_lang}")
    zoom = dpi / 72
    
    # Create a transformation matrix - using a simple zoom matrix that's compatible with all PyMuPDF versions
    matrix = fitz.Matrix(zoom, zoom)
    
    # Ensure the rectangle is valid
    if rect.is_empty or rect.is_infinite:
        # Use the entire page if rect is invalid
        rect = page.rect
    
    # Create a pixmap for the specified area
    try:
        pix = page.get_pixmap(matrix=matrix, clip=rect)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to grayscale for processing
        gray = np.array(img.convert('L'))
        
        # Validate image size - skip very small images that might cause OCR errors
        if gray.shape[0] < 20 or gray.shape[1] < 20:
            tqdm.write(f"WARNING: Image too small for OCR: {gray.shape}")
            return ""
            
        # Skip blank or nearly blank images
        if np.mean(gray) > 250 or np.mean(gray) < 5:
            tqdm.write(f"WARNING: Image appears to be blank (mean pixel value: {np.mean(gray):.1f})")
            return ""
        
        # Create debug directory if saving images
        if save_images:
            debug_dir = os.path.join(tempfile.gettempdir(), "ocr_debug")
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            page_name = f"page_{page.number+1}_{timestamp}"
            debug_page_dir = os.path.join(debug_dir, page_name)
            os.makedirs(debug_page_dir, exist_ok=True)
            tqdm.write(f"INFO: Saving debug images to: {debug_page_dir}")
            
            # Save original image
            cv2.imwrite(os.path.join(debug_page_dir, "00_original.png"), gray)
        
        # Get multiple preprocessed versions of the image
        preprocessed_images = preprocess_image_for_ocr(gray)
        
        # Save preprocessed images if debugging
        if save_images:
            for i, (name, img) in enumerate(preprocessed_images):
                cv2.imwrite(os.path.join(debug_page_dir, f"{i+1:02d}_{name}.png"), img)
        
        # Try each preprocessing method and collect results
        results = []
        
        for name, img in preprocessed_images:
            # Apply OCR with the appropriate engine
            if PADDLE_OCR_AVAILABLE:
                try:
                    # Convert to RGB for PaddleOCR (if not already)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    result_text = paddle_ocr_text(img_rgb, lang=paddle_lang, use_gpu=use_gpu)
                except Exception as e:
                    tqdm.write(f"WARNING: Error applying PaddleOCR to {name} image: {str(e)}")
                    result_text = ""
            else:
                # Use Tesseract
                result_text = tesseract_ocr_text(img, lang=tesseract_lang)
            
            # Calculate confidence based on text length and quality
            if result_text.strip():
                words = result_text.strip().split()
                confidence = len(words)  # Basic confidence based on word count
                
                # Apply bonuses for specific preprocessing methods
                if name in ['clahe_otsu', 'adaptive']:
                    confidence *= 1.2
                
                # Check for reasonable average word length
                char_count = len(''.join(words))
                avg_word_len = char_count / len(words) if words else 0
                if 3 <= avg_word_len <= 10:  # Typical English word length
                    confidence *= 1.1
                
                results.append({
                    'text': result_text.strip(),
                    'method': name,
                    'confidence': confidence,
                    'word_count': len(words)
                })
        
        # Select the best result based on confidence
        if results:
            # Sort by confidence score (descending)
            results.sort(key=lambda x: x['confidence'], reverse=True)
            best_result = results[0]
            tqdm.write(f"SUCCESS: Best OCR method: {best_result['method']} with {best_result['word_count']} words")
            return best_result['text']
        else:
            # Try one more time with original image directly
            if PADDLE_OCR_AVAILABLE:
                try:
                    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    return paddle_ocr_text(img_rgb, lang=paddle_lang, use_gpu=use_gpu)
                except Exception as e:
                    tqdm.write(f"WARNING: Error applying PaddleOCR to original image: {str(e)}")
                    # Fall back to Tesseract if PaddleOCR fails
                    if TESSERACT_OCR_AVAILABLE:
                        return tesseract_ocr_text(gray, lang=tesseract_lang)
                    return ""
            else:
                return tesseract_ocr_text(gray, lang=tesseract_lang)
            
    except Exception as e:
        error_msg = str(e)
        tqdm.write(f"WARNING: OCR error: {error_msg}")
        return ""

def extract_text_with_ocr_fallback(page, rect, use_ocr=False, ocr_lang=OCR_LANG, save_debug_images=False, use_gpu=False, ocr_detector=None):
    """Extract text around a widget with optional OCR fallback.
    
    Args:
        page: PyMuPDF page object
        rect: Rectangle area to extract text from
        use_ocr: Whether to use OCR fallback if no text is found
        ocr_lang: OCR language to use
        save_debug_images: Whether to save debug images for OCR
        use_gpu: Whether to use GPU acceleration for OCR
        ocr_detector: Optional OCRDetector object to track if OCR was used
        
    Returns:
        Extracted text as string
    """
    try:
        # Try to extract text using PyMuPDF's built-in text extraction
        # Handle different PyMuPDF versions by trying different methods
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
        # Handle different word format in different PyMuPDF versions
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
        
        # If no words found and OCR is enabled, try OCR
        if not filtered_words and use_ocr and OCR_AVAILABLE:
            ocr_result = ocr_text(page, rect, lang=ocr_lang, save_images=save_debug_images, use_gpu=use_gpu)
            if ocr_result:
                # Mark that OCR was used if we have a detector
                if ocr_detector is not None and hasattr(ocr_detector, 'mark_ocr_used'):
                    ocr_detector.mark_ocr_used()
                tqdm.write(f"SUCCESS: OCR found text: {ocr_result[:30]}...")
                return ocr_result
        
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
        # If all else fails and OCR is enabled, try OCR on the whole rect
        if use_ocr and OCR_AVAILABLE:
            ocr_result = ocr_text(page, rect, lang=ocr_lang, save_images=save_debug_images, use_gpu=use_gpu)
            # Mark that OCR was used if we have a detector
            if ocr_detector is not None and hasattr(ocr_detector, 'mark_ocr_used') and ocr_result:
                ocr_detector.mark_ocr_used()
            return ocr_result
        return ""

def list_available_languages():
    """List available OCR languages based on the available OCR engines.
    
    Returns:
        List of available language codes
    """
    if not OCR_AVAILABLE:
        return ["OCR not available. Install paddleocr or pytesseract first."]
    
    # Use Tesseract language list if available
    if TESSERACT_OCR_AVAILABLE:
        try:
            # Get tesseract command path
            tesseract_cmd = pytesseract.pytesseract.tesseract_cmd or 'tesseract'
            import subprocess
            result = subprocess.run([tesseract_cmd, '--list-langs'], capture_output=True, text=True)
            
            # Parse the output to extract language codes
            langs = []
            for line in result.stdout.strip().split('\n'):
                if line and not line.startswith('List'):
                    langs.append(line.strip())
            
            return langs
        except Exception as e:
            tqdm.write(f"Error listing Tesseract languages: {str(e)}")
    
    # Fall back to PaddleOCR language list if Tesseract is not available
    if PADDLE_OCR_AVAILABLE:
        # PaddleOCR supported languages 
        paddle_langs = [
            "ch", "en", "fr", "german", "korean", "japan",
            "chinese_cht", "ta", "te", "ka", "latin",
            "arabic", "cyrillic", "devanagari"
        ]
        
        # Map to more familiar language names for display
        lang_display = {
            "ch": "Chinese (Simplified)",
            "en": "English",
            "fr": "French",
            "german": "German",
            "korean": "Korean",
            "japan": "Japanese",
            "chinese_cht": "Chinese (Traditional)",
            "ta": "Tamil",
            "te": "Telugu",
            "ka": "Kannada",
            "latin": "Latin",
            "arabic": "Arabic",
            "cyrillic": "Cyrillic",
            "devanagari": "Devanagari"
        }
        
        # Return formatted language list
        return [f"{code} - {lang_display.get(code, code)}" for code in paddle_langs]
    
    return ["No OCR languages available"] 