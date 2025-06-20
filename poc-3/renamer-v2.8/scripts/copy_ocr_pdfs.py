#!/usr/bin/env python
# copy_ocr_pdfs.py
"""
Copy PDFs that required OCR processing to a separate folder.
This helps identify documents that need special processing and may require
further analysis or optimization.

Usage:
    python copy_ocr_pdfs.py --source templates --dest ocr_needed --catalog widget_catalog.json
"""
import os
import sys
import json
import argparse
import shutil
from tqdm import tqdm

def copy_ocr_pdfs(catalog_file, source_folder, dest_folder, copy_all=False):
    """
    Copy PDFs that required OCR to a separate directory.
    
    Args:
        catalog_file: Path to the widget_catalog.json file containing OCR usage data
        source_folder: Folder containing the original PDFs
        dest_folder: Destination folder for PDFs that needed OCR
        copy_all: If True, copy all fields from a document that has at least one OCR field
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Load the catalog file
    try:
        with open(catalog_file, 'r', encoding='utf-8') as f:
            catalog_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading catalog file: {e}")
        return
    
    # List of documents that needed OCR
    ocr_documents = []
    
    # Process the catalog
    documents = catalog_data.get("documents", {})
    
    for doc_name, doc_data in documents.items():
        # Check if any field in this document used OCR
        fields = doc_data.get("fields", [])
        ocr_fields = [field for field in fields if field.get("ocr_used", False)]
        
        if ocr_fields:
            ocr_documents.append(doc_name)
    
    # Copy the PDFs
    if ocr_documents:
        print(f"Found {len(ocr_documents)} documents that required OCR")
        
        for doc_name in tqdm(ocr_documents, desc="Copying files"):
            # Source and destination paths
            source_path = os.path.join(source_folder, doc_name)
            dest_path = os.path.join(dest_folder, doc_name)
            
            # Check if source file exists
            if not os.path.exists(source_path):
                print(f"Warning: Source file not found: {source_path}")
                continue
            
            # Copy the file
            try:
                shutil.copy2(source_path, dest_path)
            except Exception as e:
                print(f"Error copying {doc_name}: {e}")
        
        print(f"\nSuccessfully copied {len(ocr_documents)} OCR-requiring documents to: {dest_folder}")
        print("\nDocuments that required OCR:")
        for doc in ocr_documents:
            print(f"   - {doc}\n")
    else:
        print("No documents required OCR assistance!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Copy PDFs that require OCR to a separate folder")
    ap.add_argument("--source", required=True, help="Source folder containing original PDFs")
    ap.add_argument("--dest", required=True, help="Destination folder for PDFs that needed OCR")
    ap.add_argument("--catalog", required=True, help="Path to widget_catalog.json file")
    ap.add_argument("--all-fields", action="store_true", help="Copy all fields from documents with at least one OCR field")
    args = ap.parse_args()
    
    copy_ocr_pdfs(args.catalog, args.source, args.dest, copy_all=args.all_fields) 