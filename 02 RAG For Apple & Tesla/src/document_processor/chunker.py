"""
Text chunking with overlap and metadata preservation.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger
from src.config import get_config


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    document_name: str
    section: Optional[str]
    page_number: int
    chunk_index: int
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "document_name": self.document_name,
            "section": self.section,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index
        }
    
    def format_source(self) -> str:
        """Format source citation."""
        parts = [self.document_name]
        if self.section:
            parts.append(self.section)
        parts.append(f"p. {self.page_number}")
        return ", ".join(parts)


class TextChunker:
    """Chunk text into overlapping segments."""
    
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        config = get_config()
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        self.separators = config.chunking.separators
    
    def chunk_document(self, pdf_document) -> List[TextChunk]:
        """
        Chunk an entire PDF document.
        
        Args:
            pdf_document: PDFDocument object
            
        Returns:
            List of TextChunk objects
        """
        logger.info(f"Chunking document: {pdf_document.document_name}")
        
        all_chunks = []
        chunk_counter = 0
        
        for page_info in pdf_document.pages:
            page_text = page_info["text"]
            page_num = page_info["page_number"]
            section = page_info["section"]
            
            # Chunk the page text
            page_chunks = self._chunk_text(page_text)
            
            # Create TextChunk objects with metadata
            for chunk_text in page_chunks:
                chunk = TextChunk(
                    text=chunk_text,
                    document_name=pdf_document.document_name,
                    section=section,
                    page_number=page_num,
                    chunk_index=chunk_counter
                )
                all_chunks.append(chunk)
                chunk_counter += 1
        
        logger.info(f"Created {len(all_chunks)} chunks from {pdf_document.document_name}")
        return all_chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text with overlap.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        if not text or len(text) == 0:
            return []
        
        # Approximate tokens (4 chars per token)
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + char_chunk_size
            
            # If this isn't the last chunk, try to break at a separator
            if end < len(text):
                # Look for separator in the last portion of the chunk
                search_start = max(start, end - 200)
                
                best_break = -1
                for separator in self.separators:
                    pos = text.rfind(separator, search_start, end)
                    if pos > best_break:
                        best_break = pos
                
                if best_break != -1:
                    end = best_break + 1
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - char_overlap
            
            # Prevent infinite loop
            if start <= 0:
                start = end
        
        return chunks
    
    def chunk_documents(self, pdf_documents: List) -> List[TextChunk]:
        """
        Chunk multiple documents.
        
        Args:
            pdf_documents: List of PDFDocument objects
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in pdf_documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


class HybridChunker(TextChunker):
    """
    Advanced chunker that respects document structure.
    
    Tries to keep sections together when possible.
    """
    
    def chunk_document(self, pdf_document) -> List[TextChunk]:
        """Chunk document with section awareness."""
        logger.info(f"Hybrid chunking document: {pdf_document.document_name}")
        
        # Group pages by section
        sections = {}
        for page_info in pdf_document.pages:
            section = page_info["section"] or "Unknown"
            if section not in sections:
                sections[section] = []
            sections[section].append(page_info)
        
        all_chunks = []
        chunk_counter = 0
        
        # Process each section
        for section, pages in sections.items():
            # Combine all pages in section
            section_text = "\n\n".join(page["text"] for page in pages)
            
            # Chunk the section
            section_chunks = self._chunk_text(section_text)
            
            # Assign page numbers (use first page of chunk)
            for chunk_text in section_chunks:
                # Find which page this chunk starts in
                page_num = pages[0]["page_number"]  # Default to first page
                
                chunk = TextChunk(
                    text=chunk_text,
                    document_name=pdf_document.document_name,
                    section=section if section != "Unknown" else None,
                    page_number=page_num,
                    chunk_index=chunk_counter
                )
                all_chunks.append(chunk)
                chunk_counter += 1
        
        logger.info(f"Created {len(all_chunks)} chunks from {pdf_document.document_name}")
        return all_chunks