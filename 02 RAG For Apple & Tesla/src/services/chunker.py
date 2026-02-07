"""
Document chunking strategies for optimal retrieval.
Implements semantic, fixed, and recursive chunking.
"""
import re
from typing import List, Dict, Any
from loguru import logger

from src.models.schemas import DocumentChunk, Document
from src.config import ChunkingConfig


class DocumentChunker:
    """Handles document chunking with multiple strategies."""
    
    def __init__(self, config: ChunkingConfig):
        """
        Initialize chunker with configuration.
        
        Args:
            config: Chunking configuration
        """
        self.config = config
        self.strategy_map = {
            'fixed': self._fixed_chunking,
            'semantic': self._semantic_chunking,
            'recursive': self._recursive_chunking
        }
    
    def chunk_document(
        self,
        document: Document,
        pages: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Chunk document using configured strategy.
        
        Args:
            document: Document object
            pages: List of page dictionaries
            
        Returns:
            List of document chunks
        """
        logger.info(f"Chunking document {document.doc_id} using {self.config.strategy} strategy")
        
        strategy_func = self.strategy_map.get(self.config.strategy, self._semantic_chunking)
        chunks = strategy_func(document, pages)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _fixed_chunking(
        self,
        document: Document,
        pages: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Fixed-size chunking with overlap.
        Simple and effective for most use cases.
        """
        chunks = []
        chunk_id = 0
        
        for page in pages:
            text = page['text']
            page_num = page['page_number']
            
            # Split into sentences for better chunk boundaries
            sentences = self._split_into_sentences(text)
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                
                if current_length + sentence_length > self.config.chunk_size:
                    # Create chunk
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text.split()) >= self.config.min_chunk_size:
                            chunks.append(self._create_chunk(
                                chunk_id=chunk_id,
                                text=chunk_text,
                                document=document,
                                page_num=page_num,
                                section=self._detect_section_for_page(document, page_num)
                            ))
                            chunk_id += 1
                        
                        # Keep overlap
                        overlap_words = self.config.chunk_overlap
                        overlap_text = ' '.join(current_chunk).split()[-overlap_words:]
                        current_chunk = [' '.join(overlap_text)]
                        current_length = len(overlap_text)
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= self.config.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        document=document,
                        page_num=page_num,
                        section=self._detect_section_for_page(document, page_num)
                    ))
                    chunk_id += 1
        
        return chunks
    
    def _semantic_chunking(
        self,
        document: Document,
        pages: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Semantic chunking based on topic boundaries.
        Uses paragraph and section breaks.
        """
        chunks = []
        chunk_id = 0
        
        for page in pages:
            text = page['text']
            page_num = page['page_number']
            section = self._detect_section_for_page(document, page_num)
            
            # Split by paragraphs (double newline or section headers)
            paragraphs = self._split_into_paragraphs(text)
            
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                para_length = len(para.split())
                
                # Check if this is a section header
                is_header = self._is_section_header(para)
                
                if is_header and current_chunk:
                    # Save current chunk before starting new section
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.split()) >= self.config.min_chunk_size:
                        chunks.append(self._create_chunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            document=document,
                            page_num=page_num,
                            section=section
                        ))
                        chunk_id += 1
                    current_chunk = []
                    current_length = 0
                
                # Add paragraph to current chunk
                if current_length + para_length <= self.config.chunk_size:
                    current_chunk.append(para)
                    current_length += para_length
                else:
                    # Save current chunk
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text.split()) >= self.config.min_chunk_size:
                            chunks.append(self._create_chunk(
                                chunk_id=chunk_id,
                                text=chunk_text,
                                document=document,
                                page_num=page_num,
                                section=section
                            ))
                            chunk_id += 1
                    
                    # Start new chunk with this paragraph
                    current_chunk = [para]
                    current_length = para_length
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= self.config.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        document=document,
                        page_num=page_num,
                        section=section
                    ))
                    chunk_id += 1
        
        return chunks
    
    def _recursive_chunking(
        self,
        document: Document,
        pages: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Recursive chunking that tries multiple separators.
        Falls back to character-level splitting if needed.
        """
        separators = ["\n\n", "\n", ". ", " "]
        chunks = []
        chunk_id = 0
        
        for page in pages:
            text = page['text']
            page_num = page['page_number']
            section = self._detect_section_for_page(document, page_num)
            
            page_chunks = self._recursive_split(
                text=text,
                separators=separators,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            for chunk_text in page_chunks:
                if len(chunk_text.split()) >= self.config.min_chunk_size:
                    chunks.append(self._create_chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        document=document,
                        page_num=page_num,
                        section=section
                    ))
                    chunk_id += 1
        
        return chunks
    
    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Recursively split text using different separators."""
        if not separators:
            # Base case: split by characters
            words = text.split()
            chunks = []
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk)
            return chunks
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        splits = text.split(separator)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split.split())
            
            if current_length + split_length <= chunk_size:
                current_chunk.append(split)
                current_length += split_length
            else:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                
                # If split is too large, recursively split it
                if split_length > chunk_size:
                    sub_chunks = self._recursive_split(
                        split, remaining_separators, chunk_size, chunk_overlap
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk = [split]
                    current_length = split_length
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks
    
    def _create_chunk(
        self,
        chunk_id: int,
        text: str,
        document: Document,
        page_num: int,
        section: str
    ) -> DocumentChunk:
        """Create a document chunk with metadata."""
        return DocumentChunk(
            chunk_id=f"{document.doc_id}_chunk_{chunk_id}",
            text=text,
            metadata={
                'document': document.title,
                'company': document.company,
                'page_number': page_num,
                'section': section,
                'doc_id': document.doc_id,
                'word_count': len(text.split())
            }
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text is likely a section header."""
        # Check for "Item X:" pattern or all caps short text
        if re.match(r'^\s*Item\s+\d+[A-Z]?[\.:]\s*', text, re.IGNORECASE):
            return True
        if len(text.split()) < 10 and text.isupper():
            return True
        return False
    
    def _detect_section_for_page(self, document: Document, page_num: int) -> str:
        """Detect which section a page belongs to."""
        sections = document.metadata.get('sections', {})
        
        current_section = "Unknown"
        for section_name, section_page in sorted(sections.items(), key=lambda x: x[1]):
            if page_num >= section_page:
                current_section = section_name
        
        return current_section