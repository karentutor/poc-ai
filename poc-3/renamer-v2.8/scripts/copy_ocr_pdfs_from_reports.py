#!/usr/bin/env python
# copy_ocr_pdfs_from_reports.py
"""
Copy PDFs that required OCR processing to a separate folder based on batch renamer reports.
This helps identify documents that need special processing and may require
further analysis or optimization.

Usage:
    python copy_ocr_pdfs_from_reports.py --source forms --dest ocr_needed --reports reports
"""
import os
import sys
import csv
import argparse
import shutil
from tqdm import tqdm
import glob

def copy_ocr_pdfs_from_reports(reports_folder, source_folder, dest_folder):
    """
    Copy PDFs that required OCR to a separate directory based on the CSV reports.
    
    Args:
        reports_folder: Folder containing the CSV reports
        source_folder: Folder containing the original PDFs
        dest_folder: Destination folder for PDFs that needed OCR
    """
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Find all report CSV files
    report_files = glob.glob(os.path.join(reports_folder, "*.csv"))
    
    if not report_files:
        print(f"No report files found in {reports_folder}")
        return
    
    # Dictionary to track PDFs that needed OCR
    ocr_documents = {}
    
    # Process each report file
    for report_file in tqdm(report_files, desc="Processing reports"):
        base_name = os.path.splitext(os.path.basename(report_file))[0]
        pdf_name = f"{base_name}.pdf"
        
        try:
            with open(report_file, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                ocr_used = False
                for row in reader:
                    if len(row) >= 8 and row[7] == "Yes":  # Check OCR usage column
                        ocr_used = True
                        break
                
                if ocr_used:
                    ocr_documents[pdf_name] = True
        except Exception as e:
            print(f"Error processing report {report_file}: {e}")
    
    # Copy the PDFs
    if ocr_documents:
        print(f"Found {len(ocr_documents)} documents that required OCR")
        
        for doc_name in tqdm(ocr_documents.keys(), desc="Copying files"):
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
        for doc in ocr_documents.keys():
            print(f"   - {doc}\n")
    else:
        print("No documents required OCR assistance!")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Copy PDFs that require OCR to a separate folder based on batch processing reports")
    ap.add_argument("--source", required=True, help="Source folder containing original PDFs")
    ap.add_argument("--dest", required=True, help="Destination folder for PDFs that needed OCR")
    ap.add_argument("--reports", required=True, help="Folder containing CSV reports from batch processing")
    args = ap.parse_args()
    
    copy_ocr_pdfs_from_reports(args.reports, args.source, args.dest) 