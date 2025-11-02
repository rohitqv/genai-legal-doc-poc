"""
OCR and text extraction module for legal documents
"""
import base64
from pathlib import Path
from typing import Optional, Dict, List
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from langdetect import detect
import torch
from transformers import pipeline

from ..utils.config import (
    OCR_LANGUAGE, SUPPORTED_LANGUAGES,
    TRANSLATION_MODEL_EN_FR, TRANSLATION_MODEL_FR_EN,
    TRANSLATION_MODEL_EN_DE, TRANSLATION_MODEL_DE_EN
)
from ..utils.logger import logger


class OCRTextExtractor:
    """Class for OCR and text extraction from various document formats"""
    
    def __init__(self):
        self.ocr_language = OCR_LANGUAGE
        self.translation_models = {}
        self._initialize_translation_models()
    
    def _initialize_translation_models(self):
        """Initialize translation models"""
        try:
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Using device: {'CUDA' if device == 0 else 'CPU'} for translation")
            
            # Initialize translation pipelines
            self.translation_models["fr_en"] = pipeline(
                "translation",
                model=TRANSLATION_MODEL_FR_EN,
                device=device
            )
            self.translation_models["de_en"] = pipeline(
                "translation",
                model=TRANSLATION_MODEL_DE_EN,
                device=device
            )
            logger.info("Translation models initialized")
        except Exception as e:
            logger.warning(f"Could not initialize translation models: {str(e)}")
            self.translation_models = {}
    
    def extract_from_pdf(self, file_path: str, use_ocr: bool = False) -> Dict[str, any]:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            use_ocr: Whether to use OCR for scanned PDFs
            
        Returns:
            Dictionary with text, metadata, and page info
        """
        logger.info(f"Extracting text from PDF: {file_path}")
        
        try:
            doc = fitz.open(file_path)
            full_text = []
            page_texts = []
            
            for page_num, page in enumerate(doc):
                # Try text extraction first
                text = page.get_text()
                
                # If text is sparse or empty, use OCR
                if use_ocr or len(text.strip()) < 100:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img, lang=self.ocr_language)
                    logger.debug(f"Used OCR for page {page_num + 1}")
                
                full_text.append(text)
                page_texts.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "length": len(text)
                })
            
            doc.close()
            
            result = {
                "text": "\n\n".join(full_text),
                "pages": page_texts,
                "total_pages": len(page_texts),
                "extraction_method": "OCR" if use_ocr else "native",
                "language": None  # Will be detected separately
            }
            
            logger.info(f"Extracted text from {len(page_texts)} pages")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise
    
    def extract_from_image(self, file_path: str, language: str = None) -> Dict[str, any]:
        """
        Extract text from image using OCR
        
        Args:
            file_path: Path to image file
            language: OCR language code (default: configured language)
            
        Returns:
            Dictionary with extracted text
        """
        logger.info(f"Extracting text from image: {file_path}")
        
        try:
            img = Image.open(file_path)
            lang = language or self.ocr_language
            
            text = pytesseract.image_to_string(img, lang=lang)
            
            result = {
                "text": text,
                "extraction_method": "OCR",
                "language": None,
                "ocr_language_used": lang
            }
            
            logger.info(f"Extracted {len(text)} characters from image")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from image {file_path}: {str(e)}")
            raise
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'fr', 'de')
        """
        try:
            if not text or len(text.strip()) < 50:
                return "unknown"
            
            # Use a sample for faster detection
            sample = text[:1000] if len(text) > 1000 else text
            lang = detect(sample)
            logger.debug(f"Detected language: {lang}")
            return lang
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return "unknown"
    
    def translate_to_english(self, text: str, source_language: str) -> Dict[str, any]:
        """
        Translate text to English
        
        Args:
            text: Text to translate
            source_language: Source language code
            
        Returns:
            Dictionary with translated text and metadata
        """
        if source_language == "en" or source_language not in ["fr", "de"]:
            return {
                "translated_text": text,
                "translation_applied": False,
                "source_language": source_language
            }
        
        try:
            model_key = f"{source_language}_en"
            if model_key not in self.translation_models:
                logger.warning(f"Translation model not available for {source_language} -> en")
                return {
                    "translated_text": text,
                    "translation_applied": False,
                    "source_language": source_language
                }
            
            # Translate in chunks to handle long texts
            chunk_size = 500
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            translated_chunks = []
            
            for chunk in chunks:
                result = self.translation_models[model_key](chunk)
                translated_chunks.append(result[0]["translation_text"])
            
            translated_text = " ".join(translated_chunks)
            
            logger.info(f"Translated {len(text)} characters from {source_language} to English")
            return {
                "translated_text": translated_text,
                "translation_applied": True,
                "source_language": source_language
            }
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return {
                "translated_text": text,
                "translation_applied": False,
                "source_language": source_language,
                "error": str(e)
            }
    
    def extract_text(
        self,
        file_path: str,
        file_type: str = "pdf",
        use_ocr: bool = False,
        translate: bool = True
    ) -> Dict[str, any]:
        """
        Main extraction method that handles all formats
        
        Args:
            file_path: Path to file
            file_type: Type of file (pdf, image, txt)
            use_ocr: Whether to use OCR
            translate: Whether to translate non-English text
            
        Returns:
            Dictionary with extracted text and metadata
        """
        logger.info(f"Extracting text from {file_type} file: {file_path}")
        
        if file_type.lower() == "pdf":
            result = self.extract_from_pdf(file_path, use_ocr)
        elif file_type.lower() in ["png", "jpg", "jpeg", "tiff"]:
            result = self.extract_from_image(file_path)
        elif file_type.lower() == "txt":
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            result = {
                "text": text,
                "extraction_method": "native",
                "language": None
            }
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Detect language
        if result["text"]:
            detected_lang = self.detect_language(result["text"])
            result["language"] = detected_lang
            
            # Translate if needed
            if translate and detected_lang not in ["en", "unknown"]:
                translation_result = self.translate_to_english(result["text"], detected_lang)
                result["translated_text"] = translation_result["translated_text"]
                result["translation_applied"] = translation_result["translation_applied"]
            else:
                result["translated_text"] = result["text"]
                result["translation_applied"] = False
        
        return result

