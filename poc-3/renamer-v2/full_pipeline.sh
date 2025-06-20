python 01_harvest.py --folder ../pdfs/BondForms -o widget_catalog.csv
python 02_build_index.py widget_catalog.csv --out widget
python 03_rename_from_index.py ./pdfs/forms/form1_blank.pdf ./pdfs/forms/form1_renamed.pdf