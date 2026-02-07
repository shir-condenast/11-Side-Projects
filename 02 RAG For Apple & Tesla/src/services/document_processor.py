"""
Document processor for ingesting and parsing PDF files.
Handles extraction of text and metadata from 10-K filings.
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import pdfplumber
from loguru import logger

from src.models.schemas import Document, DocumentChunk


class PDFProcessor:
    """Processes PDF documents and extracts structured content."""
    
    def __init__(self):
        """Initialize PDF processor."""
        self.section_patterns = {
            'item_1': r'Item\s+1[:\.]?\s*Business',
            'item_1a': r'Item\s+1A[:\.]?\s*Risk\s+Factors',
            'item_1b': r'Item\s+1B[:\.]?\s*Unresolved\s+Staff\s+Comments',
            'item_7': r'Item\s+7[:\.]?\s*Management.*Discussion',
            'item_8': r'Item\s+8[:\.]?\s*Financial\s+Statements',
            'item_9': r'Item\s+9[:\.]?\s*(?:Changes|Disagreements)',
        }
    
    def process_document(self, pdf_path: Path) -> Document:
        """
        Process a PDF document and extract structured content.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Document object with metadata
        """
        logger.info(f"Processing document: {pdf_path}")
        
        # Determine company from filename
        company = self._detect_company(pdf_path)
        
        # Extract text with metadata
        pages = self._extract_pages_with_metadata(pdf_path)
        
        # Create document
        doc = Document(
            doc_id=pdf_path.stem,
            title=f"{company} 10-K Filing",
            source_path=str(pdf_path),
            company=company,
            metadata={
                "total_pages": len(pages),
                "file_size": pdf_path.stat().st_size,
                "sections": self._detect_sections(pages)
            }
        )
        
        logger.info(f"Successfully processed {company} document with {len(pages)} pages")
        return doc, pages
    
    def _detect_company(self, pdf_path: Path) -> str:
        """Detect company from filename or content."""
        filename = pdf_path.name.lower()
        if 'tsla' in filename or 'tesla' in filename:
            return "Tesla"
        elif 'apple' in filename or '10-q4-2024' in filename:
            return "Apple"
        else:
            # Try to detect from first page
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    first_page_text = pdf.pages[0].extract_text()
                    if 'tesla' in first_page_text.lower():
                        return "Tesla"
                    elif 'apple' in first_page_text.lower():
                        return "Apple"
            except Exception as e:
                logger.warning(f"Could not detect company from content: {e}")
        
        return "Unknown"
    
    def _extract_pages_with_metadata(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from all pages with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page dictionaries with text and metadata
        """
        pages = []
        
        try:
            # Use PyMuPDF for better text extraction and metadata
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                
                # Clean text
                text = self._clean_text(text)
                
                if text.strip():  # Only add non-empty pages
                    pages.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'metadata': {
                            'width': page.rect.width,
                            'height': page.rect.height,
                            'rotation': page.rotation
                        }
                    })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting pages with PyMuPDF: {e}")
            # Fallback to pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        text = self._clean_text(text)
                        
                        if text.strip():
                            pages.append({
                                'page_number': page_num + 1,
                                'text': text,
                                'metadata': {}
                            })
            except Exception as e2:
                logger.error(f"Error with pdfplumber fallback: {e2}")
                raise
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep important punctuation
        text = text.replace('\x00', '')
        text = text.replace('\ufffd', '')
        
        return text.strip()
    
    def _detect_sections(self, pages: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Detect major sections in the document.
        
        Args:
            pages: List of page dictionaries
            
        Returns:
            Dictionary mapping section names to starting page numbers
        """
        sections = {}
        
        for page in pages:
            text = page['text']
            page_num = page['page_number']
            
            for section_name, pattern in self.section_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    if section_name not in sections:
                        sections[section_name] = page_num
                        logger.debug(f"Found {section_name} at page {page_num}")
        
        return sections
    
    def extract_tables(self, pdf_path: Path, page_numbers: Optional[List[int]] = None) -> List[Dict]:
        """
        Extract tables from specific pages.
        
        Args:
            pdf_path: Path to PDF file
            page_numbers: Optional list of page numbers to extract from
            
        Returns:
            List of extracted tables
        """
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_process = page_numbers if page_numbers else range(len(pdf.pages))
                
                for page_num in pages_to_process:
                    if page_num < len(pdf.pages):
                        page = pdf.pages[page_num]
                        page_tables = page.extract_tables()
                        
                        for table in page_tables:
                            tables.append({
                                'page': page_num + 1,
                                'data': table
                            })
        
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
        
        return tables