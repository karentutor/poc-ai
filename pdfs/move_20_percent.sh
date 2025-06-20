#!/bin/bash

# Set source and destination directories
SRC_DIR="./bondForms80Percent"
DEST_DIR="./bondForms20Percent"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Get list of all files in source directory (excluding directories)
files=($(find "$SRC_DIR" -maxdepth 1 -type f))
total_files=${#files[@]}

# Calculate 20% of total files (round up)
count_to_move=$(( (total_files + 4) / 5 ))

# Shuffle and select 20%
selected_files=($(printf "%s\n" "${files[@]}" | shuf | head -n $count_to_move))

# Move files
for file in "${selected_files[@]}"; do
  mv "$file" "$DEST_DIR/"
done

echo "Moved $count_to_move files to $DEST_DIR."