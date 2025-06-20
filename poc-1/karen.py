import pdfplumber
with pdfplumber.open("./pdfs/BondForms10/AIA Document A310 - Bid.pdf") as pdf:
    page = pdf.pages[0]
    for annot in page.annots:
        bbox = annot["rect"]
        nearby_text = page.within_bbox((
            bbox[0] - 100, bbox[1] - 50,
            bbox[2] + 100, bbox[3] + 50
        )).extract_text()
        print(f"Field: {annot['field_name']}, Context: {near-
by_text}")