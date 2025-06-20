#!/usr/bin/env python3
"""
Helper module to import modules with numeric names.
"""
import importlib.util
import sys
from pathlib import Path

def load_harvest_ocr():
    """Dynamically import the 01b_harvest_ocr.py module"""
    script_dir = Path(__file__).parent
    module_path = script_dir / "01b_harvest_ocr.py"
    
    spec = importlib.util.spec_from_file_location("harvest_ocr_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["harvest_ocr_module"] = module
    spec.loader.exec_module(module)
    
    return module 