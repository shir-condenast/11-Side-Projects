"""
Generate embeddings for text chunks using sentence transformers.
"""
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from tqdm import tqdm
from src.config import get_config


class EmbeddingGenerator:
    """Generate embeddings for text using sentence transformers."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        config = get_config()
        self.model_name = model_name or config.embedding.model_name
        self.device = device or config.embedding.device
        self.batch_size = batch_size or config.embedding.batch_size
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        return embedding
    
    def encode_chunks(self, chunks: List, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for TextChunk objects.
        
        Args:
            chunks: List of TextChunk objects
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk.text for chunk in chunks]
        return self.encode(texts, show_progress=show_progress)
    
    def encode_batch_generator(self, texts: List[str], batch_size: Optional[int] = None):
        """
        Generate embeddings in batches (generator for memory efficiency).
        
        Args:
            texts: List of text strings
            batch_size: Batch size (uses config default if None)
            
        Yields:
            Batches of embeddings
        """
        batch_size = batch_size or self.batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
            batch = texts[i:i + batch_size]
            embeddings = self.encode(batch, show_progress=False)
            yield embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def __repr__(self):
        return f"EmbeddingGenerator(model={self.model_name}, dim={self.embedding_dim})"


class CachedEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator with caching to avoid recomputing embeddings.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode with caching."""
        if text in self._cache:
            return self._cache[text]
        
        embedding = super().encode_single(text)
        self._cache[text] = embedding
        return embedding
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    def cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._cache)