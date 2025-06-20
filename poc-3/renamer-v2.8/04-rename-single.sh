#!/bin/bash
# run_rename_single.sh - Script to run 03_rename_from_index.py on a single PDF file

# Default parameters
THRESHOLD=0.7
SIMILARITY="cosine"
USE_OCR=false
OCR_LANG="en"
USE_GPU=false

# Display usage information
show_usage() {
    echo "Usage: $0 <input_pdf> [options]"
    echo "Options:"
    echo "  -o, --output <pdf>     Output PDF file (default: input_pdf with '_renamed' suffix)"
    echo "  -t, --threshold <val>  Similarity threshold (default: $THRESHOLD)"
    echo "  -s, --similarity <sim> Similarity metric: cosine, euclidean, dot, cosine_sklearn (default: $SIMILARITY)"
    echo "  -k <num>               Number of nearest neighbors to retrieve (default: 1)"
    echo "  --ocr                  Enable OCR support for scanned documents"
    echo "  --ocr-lang <lang>      OCR language code (default: $OCR_LANG)"
    echo "  --gpu                  Use GPU acceleration if available"
    echo "  --dry                  Just print mapping without saving the PDF"
    echo "  --debug-ocr            Save debug images for OCR preprocessing"
    echo "  --list-langs           List available OCR languages and exit"
    echo ""
    echo "Example: $0 input.pdf -o output.pdf --ocr --gpu"
    exit 1
}

# Check if we have at least one argument
if [ $# -lt 1 ]; then
    show_usage
fi

# First argument is input PDF
INPUT_PDF=$1
shift

# Set default output file with _renamed suffix
OUTPUT_PDF="${INPUT_PDF%.pdf}_renamed.pdf"

# Parse command line options
K=1
DRY_RUN=false
DEBUG_OCR=false
LIST_LANGS=false

while [ $# -gt 0 ]; do
    case "$1" in
        -o|--output)
            OUTPUT_PDF="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -s|--similarity)
            SIMILARITY="$2"
            shift 2
            ;;
        -k)
            K="$2"
            shift 2
            ;;
        --ocr)
            USE_OCR=true
            shift
            ;;
        --ocr-lang)
            OCR_LANG="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --dry)
            DRY_RUN=true
            shift
            ;;
        --debug-ocr)
            DEBUG_OCR=true
            shift
            ;;
        --list-langs)
            LIST_LANGS=true
            shift
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Validate input PDF
if [ ! -f "$INPUT_PDF" ]; then
    echo "Error: Input PDF file '$INPUT_PDF' not found!"
    exit 1
fi

# Validate output path
OUTPUT_DIR=$(dirname "$OUTPUT_PDF")
if [ ! -d "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "." ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Build command
CMD="python scripts/03_rename_from_index.py \"$INPUT_PDF\" \"$OUTPUT_PDF\" -k $K --similarity $SIMILARITY"

# Add threshold if set
if [ -n "$THRESHOLD" ]; then
    CMD="$CMD --threshold $THRESHOLD"
fi

# Add OCR if enabled
if [ "$USE_OCR" = true ]; then
    CMD="$CMD --ocr --ocr-lang $OCR_LANG"
fi

# Add GPU if enabled
if [ "$USE_GPU" = true ]; then
    CMD="$CMD --gpu"
fi

# Add dry run if enabled
if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry"
fi

# Add debug OCR if enabled
if [ "$DEBUG_OCR" = true ]; then
    CMD="$CMD --save-debug-images"
fi

# Add list languages if enabled
if [ "$LIST_LANGS" = true ]; then
    CMD="$CMD --list-langs"
fi

# Print the command to be executed
echo "Executing: $CMD"

# Execute command
eval $CMD 