"""
Text cleaning and normalization module
"""
import re
from typing import List, Dict
from ..utils.config import CHUNK_SIZE, CHUNK_OVERLAP
from ..utils.logger import logger


class TextCleaner:
    """Class for cleaning and normalizing legal document text"""
    
    def __init__(self):
        # Common patterns to clean
        self.patterns = [
            (r'\s+', ' '),  # Multiple whitespace
            (r'\n\s*\n\s*\n+', '\n\n'),  # Multiple newlines
            (r'[^\w\s\.\,\;\:\!\?\(\)\[\]\-\'\"\/\@\#\$\%\&\*\+\=]', ''),  # Special chars
        ]
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
            aggressive: Whether to use aggressive cleaning (removes more characters)
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Apply standard cleaning patterns
        for pattern, replacement in self.patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Remove excessive whitespace
        cleaned = cleaned.strip()
        
        # Remove table-like structures (rows of repeated characters)
        if aggressive:
            lines = cleaned.split('\n')
            filtered_lines = []
            for line in lines:
                # Skip lines that look like table separators
                if not re.match(r'^[\s\-=_\+]+$', line):
                    filtered_lines.append(line)
            cleaned = '\n'.join(filtered_lines)
        
        logger.debug(f"Cleaned text: {len(text)} -> {len(cleaned)} characters")
        return cleaned
    
    def split_into_chunks(
        self,
        text: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Dict[str, any]]:
        """
        Split text into chunks for embedding
        
        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in tokens (approximate)
            chunk_overlap: Overlap between chunks in tokens
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunk_size = chunk_size or CHUNK_SIZE
        chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        
        # Simple token approximation (4 characters per token)
        char_per_token = 4
        chunk_char_size = chunk_size * char_per_token
        overlap_char_size = chunk_overlap * char_per_token
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + chunk_char_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end),
                    text.rfind('\n', start, end)
                )
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "length": len(chunk_text)
                })
                chunk_index += 1
            
            # Move start position with overlap
            start = end - overlap_char_size if end - overlap_char_size > 0 else end
            
            if start >= len(text):
                break
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def remove_tables(self, text: str) -> str:
        """
        Remove table-like structures from text
        
        Args:
            text: Text potentially containing tables
            
        Returns:
            Text with tables removed
        """
        lines = text.split('\n')
        cleaned_lines = []
        in_table = False
        
        for line in lines:
            # Detect table-like patterns
            if re.match(r'^[\s\|\+\-\=]+$', line) or re.match(r'^[\s\|\-]+$', line):
                in_table = True
                continue
            
            # Detect table rows (multiple delimiters)
            if '|' in line and line.count('|') > 2:
                in_table = True
                continue
            
            if in_table and not line.strip():
                in_table = False
                continue
            
            if not in_table:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from legal document
        
        Args:
            text: Document text
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        current_section = "PREAMBLE"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            # Detect section headers (all caps, numbered, etc.)
            section_pattern = r'^(SECTION|PART|ARTICLE|CHAPTER)\s+\d+[\.:]?\s*(.+)$'
            match = re.match(section_pattern, line, re.IGNORECASE)
            
            if match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        logger.info(f"Extracted {len(sections)} sections from document")
        return sections
    
    def normalize_legal_document(
        self,
        text: str,
        clean: bool = True,
        remove_tables: bool = True,
        extract_sections: bool = False
    ) -> Dict[str, any]:
        """
        Full normalization pipeline for legal documents
        
        Args:
            text: Raw document text
            clean: Whether to clean text
            remove_tables: Whether to remove tables
            extract_sections: Whether to extract sections
            
        Returns:
            Dictionary with normalized text and metadata
        """
        result = {
            "original_length": len(text),
            "normalized_text": text
        }
        
        if clean:
            result["normalized_text"] = self.clean_text(result["normalized_text"])
            result["cleaned"] = True
        
        if remove_tables:
            result["normalized_text"] = self.remove_tables(result["normalized_text"])
            result["tables_removed"] = True
        
        if extract_sections:
            result["sections"] = self.extract_sections(result["normalized_text"])
            result["section_count"] = len(result["sections"])
        
        result["final_length"] = len(result["normalized_text"])
        result["reduction_percent"] = (
            (result["original_length"] - result["final_length"]) / result["original_length"] * 100
            if result["original_length"] > 0 else 0
        )
        
        logger.info(
            f"Normalized document: {result['original_length']} -> {result['final_length']} "
            f"({result['reduction_percent']:.1f}% reduction)"
        )
        
        return result

