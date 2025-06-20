from itertools import count
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import (
    NameObject,
    TextStringObject,
    IndirectObject,
)

src  = "form1.pdf"
dest = "form1_blank.pdf"

reader   = PdfReader(src)
writer   = PdfWriter()
new_name = count(1)                      # Blank1, Blank2, …

def rename_field(obj):
    """Recursively rename /T in parent + child field dictionaries."""
    if isinstance(obj, IndirectObject):
        obj = obj.get_object()

    if "/T" in obj:
        obj[NameObject("/T")] = TextStringObject(f"Blank{next(new_name)}")

    if "/Kids" in obj:
        for kid in obj["/Kids"]:
            rename_field(kid)

# ── grab the (possibly indirect) AcroForm dict ──────────────────────────────
acroform_ref = reader.trailer["/Root"].get("/AcroForm")
if not acroform_ref:
    raise RuntimeError("PDF has no AcroForm / form fields")

acroform = acroform_ref.get_object()     # dereference

# ── walk every top‑level field ──────────────────────────────────────────────
for field_ref in acroform.get("/Fields", []):
    rename_field(field_ref)

# ── copy pages and the (now‑modified) AcroForm to the writer ───────────────
for page in reader.pages:
    writer.add_page(page)
writer._root_object.update({NameObject("/AcroForm"): acroform})

with open(dest, "wb") as f:
    writer.write(f)

print(f"Saved → {dest}")
