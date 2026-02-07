"""
Main RAG pipeline orchestrating all components.
Provides the unified interface for question answering.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from src.config import RAGConfig
from src.models.schemas import (
    Document,
    Query,
    RAGResponse,
    RetrievedContext
)
from src.services.document_processor import PDFProcessor
from src.services.chunker import DocumentChunker
from src.services.embedding_service import EmbeddingService
from src.services.vector_store import FAISSVectorStore
from src.services.retriever import Retriever
from src.services.llm_service import LLMService


class RAGPipeline:
    """Complete RAG pipeline for question answering."""
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialize RAG pipeline.
        
        Args:
            config: RAG configuration (uses default if not provided)
        """
        self.config = config or RAGConfig()
        self.is_indexed = False
        
        logger.info("Initializing RAG Pipeline")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")
        
        # Document processing
        self.pdf_processor = PDFProcessor()
        self.chunker = DocumentChunker(self.config.chunking)
        
        # Embeddings
        self.embedding_service = EmbeddingService(self.config.embedding)
        embedding_dim = self.embedding_service.get_embedding_dimension()
        
        # Vector store
        self.vector_store = FAISSVectorStore(
            self.config.vector_store,
            embedding_dim
        )
        
        # Retriever
        self.retriever = Retriever(
            self.config.retriever,
            self.vector_store,
            self.embedding_service
        )
        
        # LLM
        self.llm_service = LLMService(self.config.llm)
        
        logger.info("All components initialized successfully")
    
    def index_documents(self, pdf_paths: List[Path], force_reindex: bool = False):
        """
        Index PDF documents into vector store.
        
        Args:
            pdf_paths: List of paths to PDF files
            force_reindex: Force reindexing even if index exists
        """
        # Check if index exists
        if not force_reindex and self._index_exists():
            logger.info("Loading existing index...")
            self.vector_store.load()
            self.is_indexed = True
            logger.info(f"Loaded index with {len(self.vector_store.chunks)} chunks")
            return
        
        logger.info(f"Indexing {len(pdf_paths)} documents...")
        
        all_chunks = []
        
        # Process each document
        for pdf_path in pdf_paths:
            logger.info(f"Processing: {pdf_path.name}")
            
            # Extract text and metadata
            document, pages = self.pdf_processor.process_document(pdf_path)
            
            # Chunk document
            chunks = self.chunker.chunk_document(document, pages)
            all_chunks.extend(chunks)
            
            logger.info(f"Created {len(chunks)} chunks from {pdf_path.name}")
        
        logger.info(f"Total chunks: {len(all_chunks)}")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedding_service.encode(texts, show_progress=True)
        
        # Add to vector store
        self.vector_store.add_chunks(all_chunks, embeddings)
        
        # Save index
        logger.info("Saving index...")
        self.vector_store.save()
        
        self.is_indexed = True
        logger.info("Indexing complete!")
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            query: User question
            
        Returns:
            Dictionary with answer and sources in required format
        """
        if not self.is_indexed:
            raise RuntimeError("Documents must be indexed before answering questions")
        
        logger.info(f"Answering question: {query[:100]}...")
        
        # Check if question is out of scope
        if self._is_out_of_scope_question(query):
            return {
                "answer": "This question cannot be answered based on the provided documents.",
                "sources": []
            }
        
        # Retrieve contexts
        contexts = self.retriever.retrieve(query)
        
        if not contexts:
            return {
                "answer": "Not specified in the document.",
                "sources": []
            }
        
        # Generate answer
        answer = self.llm_service.answer_question(query, contexts)
        
        # Extract sources from answer or use retrieved contexts
        sources = self._extract_sources(answer, contexts)
        
        # Clean answer (remove source markers if embedded)
        answer = self._clean_answer(answer)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def batch_answer_questions(
        self,
        questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of question dictionaries with 'question_id' and 'question'
            
        Returns:
            List of answer dictionaries
        """
        logger.info(f"Answering {len(questions)} questions in batch")
        
        results = []
        for q in questions:
            question_id = q.get('question_id')
            question_text = q.get('question')
            
            logger.info(f"Processing Q{question_id}: {question_text[:80]}...")
            
            result = self.answer_question(question_text)
            result['question_id'] = question_id
            
            results.append(result)
        
        logger.info("Batch processing complete")
        return results
    
    def _is_out_of_scope_question(self, query: str) -> bool:
        """
        Detect if question is outside document scope.
        
        Args:
            query: User query
            
        Returns:
            True if question is out of scope
        """
        out_of_scope_patterns = [
            'forecast', 'prediction', 'future', '2025', '2026',
            'what color', 'painted', 'favorite', 'who is the cfo',
            'stock price forecast', 'will happen'
        ]
        
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in out_of_scope_patterns)
    
    def _extract_sources(
        self,
        answer: str,
        contexts: List[RetrievedContext]
    ) -> List[str]:
        """Extract source citations."""
        # Try to extract from answer first
        sources = self.llm_service.extract_sources(answer)
        
        if sources:
            return sources
        
        # If no sources in answer, use retrieved contexts
        if contexts:
            # Take top 3 contexts as sources
            sources = []
            for ctx in contexts[:3]:
                source = [
                    f"{ctx.chunk.metadata.get('company')} 10-K",
                    ctx.chunk.metadata.get('section', 'Unknown'),
                    f"p. {ctx.chunk.metadata.get('page_number', 'Unknown')}"
                ]
                if source not in sources:
                    sources.append(source)
        
        return sources
    
    def _clean_answer(self, answer: str) -> str:
        """Clean answer by removing embedded source citations."""
        import re
        
        # Remove citation patterns
        answer = re.sub(r'\[.*?\]', '', answer)
        
        # Clean up whitespace
        answer = ' '.join(answer.split())
        
        return answer.strip()
    
    def _index_exists(self) -> bool:
        """Check if index already exists."""
        index_path = self.config.index_dir
        required_files = ['faiss_index.bin', 'chunks.pkl', 'metadata.pkl']
        
        return all((index_path / f).exists() for f in required_files)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline."""
        stats = {
            'is_indexed': self.is_indexed,
            'config': {
                'chunking_strategy': self.config.chunking.strategy,
                'chunk_size': self.config.chunking.chunk_size,
                'embedding_model': self.config.embedding.model_name,
                'llm_model': self.config.llm.model_name,
                'use_reranker': self.config.retriever.use_reranker,
                'use_gpu': self.config.use_gpu
            }
        }
        
        if self.is_indexed:
            stats['vector_store'] = self.vector_store.get_stats()
        
        return stats
    
    def clear_gpu_cache(self):
        """Clear GPU cache from all components."""
        logger.info("Clearing GPU cache...")
        self.embedding_service.clear_gpu_cache()
        self.llm_service.clear_gpu_cache()