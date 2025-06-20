# PDF Processing Scripts

This directory contains scripts for processing PDF forms, extracting widget information, and applying OCR (Optical Character Recognition) for better text extraction from scanned documents.

## Shared OCR Module

The `ocr_utils.py` module provides advanced OCR capabilities that are shared between multiple scripts:

- `01c_harvest_ocr.py`: Extracts widget context from PDF documents
- `04_batch_renamer_test_ocr.py`: Tests widget naming accuracy using vector similarity

### OCR Features

The shared OCR module includes:

- Automatic tesseract path detection for Windows, Linux, and macOS
- Multiple image preprocessing techniques for better text recognition:
  - Adaptive thresholding
  - Otsu's thresholding
  - CLAHE contrast enhancement
  - Bilateral filtering for noise reduction
  - Edge enhancement with unsharp masking
  - Morphological operations for thin text
- Page orientation and skew detection/correction
- Multiple page segmentation modes (PSM) for different document layouts
- Confidence scoring to select the best result among multiple attempts
- Support for multiple languages
- Debug image saving for troubleshooting

## Usage

### Extracting Widget Context

```bash
# Basic usage
python 01c_harvest_ocr.py --folder templates --out widget_catalog.json --ocr

# Advanced OCR options
python 01c_harvest_ocr.py --folder templates --out widget_catalog.json --ocr --ocr-dpi 450 --ocr-lang eng+fra

# Debug mode with image saving
python 01c_harvest_ocr.py --folder templates --out widget_catalog.json --ocr --save-debug-images

# List available languages
python 01c_harvest_ocr.py --list-langs
```

### Testing Widget Naming

```bash
# Basic usage
python 04_batch_renamer_test_ocr.py document.pdf --ocr

# Advanced OCR options
python 04_batch_renamer_test_ocr.py document.pdf --ocr --ocr-lang eng+fra --save-debug-images

# Configure similarity metrics
python 04_batch_renamer_test_ocr.py document.pdf --ocr --metric cosine --threshold 0.7
```

## Extending the OCR Module

To use the OCR utilities in a new script:

```python
# Import the OCR utilities
try:
    from ocr_utils import (
        OCR_AVAILABLE, OCR_DPI, OCR_LANG, MARGIN,
        extract_text_with_ocr_fallback, list_available_languages
    )
except ImportError:
    # Fall back to parent directory if needed
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from ocr_utils import (
            OCR_AVAILABLE, OCR_DPI, OCR_LANG, MARGIN,
            extract_text_with_ocr_fallback, list_available_languages
        )
    except ImportError:
        print("Could not import OCR utilities")
```

## Requirements

The OCR functionality requires:

- pytesseract
- Tesseract OCR (installed on the system)
- Pillow (PIL)
- OpenCV (cv2)
- NumPy

Other dependencies vary by script. 