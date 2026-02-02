"""
PDF parsing and text extraction with metadata preservation.
"""
import re
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from loguru import logger


class PDFDocument:
    """Represents a parsed PDF document with metadata."""
    
    def __init__(self, filepath: Path, document_name: str):
        self.filepath = filepath
        self.document_name = document_name
        self.pages: List[Dict[str, any]] = []
        self.metadata: Dict[str, any] = {}
    
    def add_page(self, page_num: int, text: str, section: Optional[str] = None):
        """Add a page to the document."""
        self.pages.append({
            "page_number": page_num,
            "text": text,
            "section": section
        })
    
    def __len__(self):
        return len(self.pages)
    
    def __repr__(self):
        return f"PDFDocument(name={self.document_name}, pages={len(self.pages)})"


class PDFParser:
    """Parse PDF documents and extract text with metadata."""
    
    def __init__(self):
        self.section_patterns = {
            "Item 1": r"Item\s+1[\.\:]\s*Business",
            "Item 1A": r"Item\s+1A[\.\:]\s*Risk Factors",
            "Item 1B": r"Item\s+1B[\.\:]\s*Unresolved Staff Comments",
            "Item 2": r"Item\s+2[\.\:]\s*Properties",
            "Item 3": r"Item\s+3[\.\:]\s*Legal Proceedings",
            "Item 4": r"Item\s+4[\.\:]\s*Mine Safety",
            "Item 5": r"Item\s+5[\.\:]\s*Market for Registrant",
            "Item 6": r"Item\s+6[\.\:]\s*",
            "Item 7": r"Item\s+7[\.\:]\s*Management",
            "Item 7A": r"Item\s+7A[\.\:]\s*Quantitative",
            "Item 8": r"Item\s+8[\.\:]\s*Financial Statements",
            "Item 9": r"Item\s+9[\.\:]\s*",
            "Item 9A": r"Item\s+9A[\.\:]\s*Controls",
            "Item 9B": r"Item\s+9B[\.\:]\s*",
            "Item 10": r"Item\s+10[\.\:]\s*Directors",
            "Item 11": r"Item\s+11[\.\:]\s*Executive Compensation",
            "Item 12": r"Item\s+12[\.\:]\s*Security Ownership",
            "Item 13": r"Item\s+13[\.\:]\s*",
            "Item 14": r"Item\s+14[\.\:]\s*Principal Accountant",
            "Item 15": r"Item\s+15[\.\:]\s*Exhibits",
            "Item 16": r"Item\s+16[\.\:]\s*",
        }
    
    def parse(self, filepath: Path, document_name: str) -> PDFDocument:
        """
        Parse a PDF file and extract text with metadata.
        
        Args:
            filepath: Path to PDF file
            document_name: Human-readable document name (e.g., "Apple 10-K")
            
        Returns:
            PDFDocument object with extracted content
        """
        logger.info(f"Parsing PDF: {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        
        doc = PDFDocument(filepath, document_name)
        
        try:
            pdf_file = fitz.open(str(filepath))
            current_section = None
            
            for page_num in range(len(pdf_file)):
                page = pdf_file[page_num]
                text = page.get_text("text")
                
                # Clean text
                text = self._clean_text(text)
                
                # Detect section changes
                detected_section = self._detect_section(text)
                if detected_section:
                    current_section = detected_section
                    logger.debug(f"Page {page_num + 1}: Detected section {current_section}")
                
                # Add page with metadata
                doc.add_page(
                    page_num=page_num + 1,  # 1-indexed for human readability
                    text=text,
                    section=current_section
                )
            
            pdf_file.close()
            
            # Extract document metadata
            doc.metadata = self._extract_metadata(filepath)
            
            logger.info(f"Successfully parsed {len(doc)} pages from {document_name}")
            return doc
            
        except Exception as e:
            logger.error(f"Error parsing PDF {filepath}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove multiple blank lines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def _detect_section(self, text: str) -> Optional[str]:
        """
        Detect 10-K section from page text.
        
        Returns:
            Section name (e.g., "Item 8") or None
        """
        for section_name, pattern in self.section_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return section_name
        return None
    
    def _extract_metadata(self, filepath: Path) -> Dict[str, any]:
        """Extract metadata from PDF."""
        try:
            pdf_file = fitz.open(str(filepath))
            metadata = pdf_file.metadata
            pdf_file.close()
            return metadata
        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
            return {}


class MultiDocumentParser:
    """Parse multiple PDF documents."""
    
    def __init__(self):
        self.parser = PDFParser()
    
    def parse_documents(
        self,
        documents: List[Dict[str, str]]
    ) -> List[PDFDocument]:
        """
        Parse multiple documents.
        
        Args:
            documents: List of dicts with 'filepath' and 'name' keys
            
        Returns:
            List of parsed PDFDocument objects
        """
        parsed_docs = []
        
        for doc_info in documents:
            filepath = Path(doc_info['filepath'])
            name = doc_info['name']
            
            try:
                parsed_doc = self.parser.parse(filepath, name)
                parsed_docs.append(parsed_doc)
            except Exception as e:
                logger.error(f"Failed to parse {name}: {e}")
                continue
        
        return parsed_docs