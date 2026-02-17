"""
Retrieval pipeline with optional re-ranking.
Handles query processing and context retrieval.
"""
from typing import List, Tuple
import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder

from src.models.schemas import DocumentChunk, RetrievedContext
from src.config import RetrieverConfig
from src.services.vector_store import FAISSVectorStore
from src.services.embedding_service import EmbeddingService

from rank_bm25 import BM25Okapi



class Retriever:
    """Retriever with vector search and optional re-ranking."""
    
    def __init__(
        self,
        config: RetrieverConfig,
        vector_store: FAISSVectorStore,
        embedding_service: EmbeddingService
    ):
        """
        Initialize retriever.
        
        Args:
            config: Retriever configuration
            vector_store: Vector store for similarity search
            embedding_service: Service for encoding queries
        """
        self.config = config
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.reranker = None
        
        if config.use_reranker:
            self._initialize_reranker()
    
    def _initialize_reranker(self):
        """Initialize cross-encoder reranker."""
        logger.info(f"Loading reranker model: {self.config.reranker_model}")
        try:
            self.reranker = CrossEncoder(
                self.config.reranker_model,
                max_length=512
            )
            logger.info("Reranker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            logger.warning("Continuing without reranking")
            self.config.use_reranker = False
    
    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> List[RetrievedContext]:
        """
        Retrieve relevant contexts for a query.
        
        Args:
            query: Query text
            top_k: Number of contexts to retrieve (defaults to config)
            
        Returns:
            List of retrieved contexts with scores
        """
        top_k = top_k or self.config.top_k
        
        logger.debug(f"Retrieving contexts for query: {query[:100]}...")
        
        # Encode query
        query_embedding = self.embedding_service.encode_query(query)
        
        # Initial retrieval with higher k if reranking
        initial_k = top_k * 3 if self.config.use_reranker else top_k
        results = self.vector_store.search(query_embedding, initial_k)
        
        # Filter by similarity threshold
        results = [
            (chunk, score) for chunk, score in results
            if score <= self.config.similarity_threshold
        ]
        
        if not results:
            logger.warning("No results above similarity threshold")
            return []
        
        # Rerank if configured
        if self.config.use_reranker and len(results) > 1:
            results = self._rerank(query, results)
            results = results[:self.config.rerank_top_k]
        else:
            results = results[:top_k]
        
        # Convert to RetrievedContext objects
        contexts = []
        for rank, (chunk, score) in enumerate(results, 1):
            contexts.append(RetrievedContext(
                chunk=chunk,
                score=score,
                rank=rank
            ))
        
        logger.debug(f"Retrieved {len(contexts)} contexts")
        return contexts
    
    def _rerank(
        self,
        query: str,
        results: List[Tuple[DocumentChunk, float]]
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Query text
            results: Initial retrieval results
            
        Returns:
            Reranked results
        """
        logger.debug(f"Reranking {len(results)} results")
        
        # Prepare pairs for reranker
        pairs = [(query, chunk.text) for chunk, _ in results]
        
        # Get reranker scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine with chunks
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(results, rerank_scores)
        ]
        
        # Sort by reranker score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked
    
    def retrieve_with_metadata(
        self,
        query: str,
        filter_company: str = None,
        filter_section: str = None
    ) -> List[RetrievedContext]:
        """
        Retrieve contexts with metadata filtering.
        
        Args:
            query: Query text
            filter_company: Filter by company (Apple/Tesla)
            filter_section: Filter by section (e.g., 'item_8')
            
        Returns:
            Filtered retrieved contexts
        """
        # Get all contexts
        contexts = self.retrieve(query, top_k=self.config.top_k * 2)
        
        # Apply filters
        if filter_company:
            contexts = [
                ctx for ctx in contexts
                if ctx.chunk.metadata.get('company', '').lower() == filter_company.lower()
            ]
        
        if filter_section:
            contexts = [
                ctx for ctx in contexts
                if ctx.chunk.metadata.get('section', '').lower() == filter_section.lower()
            ]
        
        # Re-rank after filtering
        contexts = sorted(contexts, key=lambda x: x.score, reverse=True)
        contexts = contexts[:self.config.rerank_top_k]
        
        # Update ranks
        for rank, ctx in enumerate(contexts, 1):
            ctx.rank = rank
        
        return contexts


class HybridRetriever(Retriever):
    """Hybrid retriever combining dense and sparse retrieval."""
    
    def __init__(
        self,
        config: RetrieverConfig,
        vector_store: FAISSVectorStore,
        embedding_service: EmbeddingService
    ):
        """Initialize hybrid retriever with BM25."""
        super().__init__(config, vector_store, embedding_service)
        
        try:
            self.BM25 = BM25Okapi
            self.bm25_index = None
            logger.info("BM25 support enabled")
        except ImportError:
            logger.warning("BM25 not available. Using dense retrieval only.")
            self.BM25 = None
    
    def build_bm25_index(self):
        """Build BM25 index from chunks."""
        if not self.BM25:
            return
        
        logger.info("Building BM25 index")
        
        # Tokenize all chunks
        tokenized_chunks = [
            chunk.text.lower().split()
            for chunk in self.vector_store.chunks
        ]
        
        self.bm25_index = self.BM25(tokenized_chunks)
        logger.info("BM25 index built")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        dense_weight: float = 0.7
    ) -> List[RetrievedContext]:
        """
        Hybrid retrieval combining dense and sparse methods.
        
        Args:
            query: Query text
            top_k: Number of contexts to retrieve
            dense_weight: Weight for dense retrieval (1 - dense_weight for BM25)
            
        Returns:
            Combined and reranked contexts
        """
        top_k = top_k or self.config.top_k
        
        # Dense retrieval
        dense_contexts = super().retrieve(query, top_k * 2)
        
        # BM25 retrieval if available
        if self.bm25_index:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top BM25 results
            top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
            
            # Normalize scores
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            
            # Combine scores
            combined_scores = {}
            
            # Add dense scores
            for ctx in dense_contexts:
                chunk_id = ctx.chunk.chunk_id
                combined_scores[chunk_id] = {
                    'chunk': ctx.chunk,
                    'score': dense_weight * ctx.score
                }
            
            # Add BM25 scores
            sparse_weight = 1.0 - dense_weight
            for idx in top_indices:
                chunk = self.vector_store.chunks[idx]
                chunk_id = chunk.chunk_id
                bm25_score = bm25_scores[idx] / max_bm25
                
                if chunk_id in combined_scores:
                    combined_scores[chunk_id]['score'] += sparse_weight * bm25_score
                else:
                    combined_scores[chunk_id] = {
                        'chunk': chunk,
                        'score': sparse_weight * bm25_score
                    }
            
            # Sort by combined score
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )[:top_k]
            
            # Convert to RetrievedContext
            contexts = [
                RetrievedContext(
                    chunk=item['chunk'],
                    score=item['score'],
                    rank=rank
                )
                for rank, item in enumerate(sorted_results, 1)
            ]
            
            return contexts
        
        else:
            # Fall back to dense only
            return dense_contexts[:top_k]