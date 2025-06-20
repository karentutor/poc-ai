#!/usr/bin/env bash
# Sets up Tesseract language data and exports TESSDATA_PREFIX
set -euo pipefail

# ---------- Config ----------
LANGS=("eng")           # add more e.g. "rus" "fra" if needed
DATA_DIR="$HOME/tessdata"
RC_FILE="${HOME}/.bashrc"   # change to ~/.zshrc if you use zsh
# ----------------------------

echo "⏳ Creating tessdata directory at: $DATA_DIR"
mkdir -p "$DATA_DIR"

for L in "${LANGS[@]}"; do
    FILE="$DATA_DIR/${L}.traineddata"
    if [[ -f "$FILE" ]]; then
        echo "✅ ${L}.traineddata already present"
    else
        echo "⬇️  Downloading ${L}.traineddata …"
        curl -L "https://github.com/tesseract-ocr/tessdata/raw/main/${L}.traineddata" -o "$FILE"
        echo "✅ ${L}.traineddata downloaded"
    fi
done

# Add TESSDATA_PREFIX to shell rc if not present
if grep -q "TESSDATA_PREFIX" "$RC_FILE"; then
    echo "ℹ️  TESSDATA_PREFIX already defined in $RC_FILE"
else
    echo 'export TESSDATA_PREFIX="$HOME"' >> "$RC_FILE"
    echo "✅ Added TESSDATA_PREFIX to $RC_FILE (points to $HOME)"
fi

echo "🚀 Done.  Open a new terminal or run:  source \"$RC_FILE\""
echo "🔍 Test with:  tesseract --list-langs  (should list: ${LANGS[*]})"
