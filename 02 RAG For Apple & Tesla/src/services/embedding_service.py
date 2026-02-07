"""
Embedding service with GPU acceleration.
Handles conversion of text to vector embeddings.
"""
import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm

from src.config import EmbeddingConfig


class EmbeddingService:
    """Service for generating embeddings with GPU support."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding service.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info(f"Loading embedding model: {config.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = SentenceTransformer(config.model_name, device=str(self.device))
        
        # Optimize for GPU if available
        if torch.cuda.is_available():
            self.model.cuda()
            logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        logger.info("Embedding model loaded successfully")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding (defaults to config)
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = batch_size or self.config.batch_size
        
        logger.debug(f"Encoding {len(texts)} texts with batch size {batch_size}")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True
            )
            
            return embeddings
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"GPU OOM. Falling back to CPU and smaller batch size")
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Retry with CPU and smaller batch
                self.model.to('cpu')
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size // 2,
                    show_progress_bar=show_progress,
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=True
                )
                
                # Move back to GPU
                if self.config.device == 'cuda':
                    self.model.to(self.device)
                
                return embeddings
            else:
                raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.encode(query, show_progress=False)[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score
        """
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def batch_similarity(
        self,
        query_emb: np.ndarray,
        doc_embs: np.ndarray
    ) -> np.ndarray:
        """
        Calculate similarities between query and multiple documents.
        
        Args:
            query_emb: Query embedding
            doc_embs: Document embeddings
            
        Returns:
            Array of similarity scores
        """
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
        
        # Normalize if not already
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(query_norm, doc_norms.T).flatten()
        
        return similarities
    
    def clear_gpu_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")