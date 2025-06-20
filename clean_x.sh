#!/bin/bash
# Recursively remove +x from all .py, .faiss, .pkl, and .csv files

find . -type f \( -name "*.py" -o -name "*.faiss" -o -name "*.pkl" -o -name "*.csv" -o -name "*.txt"  -o -name "*.yml" -o -name "Dockerfile" \) -exec chmod -x {} \;

echo "Removed +x from all .py, .faiss, .pkl, and .csv files recursively."
