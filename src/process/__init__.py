"""
Process package for Legal Document Analysis PoC
"""
from .ocr_text_extractor import OCRTextExtractor
from .text_cleaner import TextCleaner

__all__ = ["OCRTextExtractor", "TextCleaner"]

