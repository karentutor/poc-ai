try: 
    import paddleocr
    print("PaddleOCR module imported successfully!")
    from paddleocr import PaddleOCR
    print("PaddleOCR class imported successfully!")
except ImportError as e: 
    print(f"Import error: {str(e)}") 