#!/usr/bin/env python3
"""
Test script to compare context extraction between original and improved scripts.
"""
import sys
import json
import os
from pathlib import Path
import fitz

# Import both modules to test context extraction
sys.path.append(str(Path(__file__).parent))
from scripts.import_helper import load_harvest_ocr
harvest_ocr = load_harvest_ocr()
new_context_func = harvest_ocr.widget_context
# We'll manually implement the old version for testing

def old_context_func(page, wrect, use_ocr=False):
    """Recreate the old context extraction for comparison"""
    close = wrect + (-15, -15, 15, 15)
    far = wrect + (-30, -30, 30, 30)  # Original MARGIN was 30
    
    def grab(rect):
        return page.get_text("words", clip=rect)
    
    words = grab(close)
    if not words and use_ocr:
        return "Would use OCR in the original version"
    if not words:
        words = grab(far)
    
    if words:
        words.sort(key=lambda w: (w[1], w[0]))
        snippet = " ".join(w[4] for w in words).split()[:25]  # MAX_WORDS was 25
        return " ".join(snippet) if snippet else ""
    return ""

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_context.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    doc = fitz.open(pdf_path)
    results = []
    
    for page_num, page in enumerate(doc):
        for w in page.widgets():
            if not w.field_name:
                continue
            
            old_ctx = old_context_func(page, w.rect)
            new_ctx = new_context_func(page, w.rect, False)  # No OCR for basic test
            
            results.append({
                "widgetName": w.field_name,
                "page": page_num + 1,
                "oldContext": old_ctx,
                "newContext": new_ctx,
                "oldLength": len(old_ctx),
                "newLength": len(new_ctx),
                "diff": len(new_ctx) - len(old_ctx)
            })
    
    # Sort by biggest difference in context length
    results.sort(key=lambda x: -x["diff"])
    
    # Output results
    with open("context_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} widgets.")
    print(f"Results saved to context_comparison.json")
    
    # Print summary
    if results:
        avg_old_len = sum(r["oldLength"] for r in results) / len(results)
        avg_new_len = sum(r["newLength"] for r in results) / len(results)
        print(f"Average old context length: {avg_old_len:.1f} chars")
        print(f"Average new context length: {avg_new_len:.1f} chars")
        print(f"Improvement: {(avg_new_len/avg_old_len - 1)*100:.1f}%")
        
        # Print a few examples
        print("\nExamples of biggest improvements:")
        for i, r in enumerate(results[:3]):
            print(f"\n{i+1}. {r['widgetName']} (page {r['page']}):")
            print(f"   OLD ({r['oldLength']} chars): {r['oldContext'][:100]}...")
            print(f"   NEW ({r['newLength']} chars): {r['newContext'][:100]}...")

if __name__ == "__main__":
    main() 