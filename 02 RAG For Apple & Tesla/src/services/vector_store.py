"""
Vector store implementation using FAISS with GPU support.
Handles storage and retrieval of embeddings.
"""
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss
from loguru import logger

from src.models.schemas import DocumentChunk
from src.config import VectorStoreConfig


class FAISSVectorStore:
    """FAISS-based vector store with GPU acceleration."""
    
    def __init__(self, config: VectorStoreConfig, embedding_dim: int):
        """
        Initialize FAISS vector store.
        
        Args:
            config: Vector store configuration
            embedding_dim: Dimension of embeddings
        """
        self.config = config
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.use_gpu = config.use_gpu and faiss.get_num_gpus() > 0
        
        if self.use_gpu:
            logger.info(f"FAISS GPU support available. Using {faiss.get_num_gpus()} GPU(s)")
        else:
            logger.info("Using FAISS CPU index")
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index based on configuration."""
        logger.info(f"Initializing FAISS index with dimension {self.embedding_dim}")
        
        # Create appropriate index based on similarity metric
        if self.config.similarity_metric == "cosine":
            # Normalize embeddings and use inner product for cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.config.similarity_metric == "l2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:  # inner product
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Move to GPU if available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Could not move FAISS to GPU: {e}. Using CPU.")
                self.use_gpu = False
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """
        Add chunks with their embeddings to the index.
        
        Args:
            chunks: List of document chunks
            embeddings: Array of embeddings
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Normalize embeddings for cosine similarity
        if self.config.similarity_metric == "cosine":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Convert to float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Add to index
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
        logger.info(f"Total chunks in index: {len(self.chunks)}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Normalize query for cosine similarity
        if self.config.similarity_metric == "cosine":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Ensure correct shape and dtype
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid index
                chunk = self.chunks[idx]
                
                # Convert distance to similarity score
                if self.config.similarity_metric == "cosine":
                    score = float(dist)  # Already similarity for IP
                elif self.config.similarity_metric == "l2":
                    score = 1.0 / (1.0 + float(dist))  # Convert L2 to similarity
                else:
                    score = float(dist)
                
                results.append((chunk, score))
        
        return results
    
    def save(self, path: Optional[Path] = None):
        """
        Save index and chunks to disk.
        
        Args:
            path: Path to save (defaults to config path)
        """
        save_path = path or self.config.index_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving vector store to {save_path}")
        
        # Save FAISS index (move to CPU first if on GPU)
        index_file = save_path / "faiss_index.bin"
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_file))
        else:
            faiss.write_index(self.index, str(index_file))
        
        # Save chunks
        chunks_file = save_path / "chunks.pkl"
        with open(chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save metadata
        metadata_file = save_path / "metadata.pkl"
        metadata = {
            'embedding_dim': self.embedding_dim,
            'num_chunks': len(self.chunks),
            'similarity_metric': self.config.similarity_metric,
        }
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Vector store saved successfully")
    
    def load(self, path: Optional[Path] = None):
        """
        Load index and chunks from disk.
        
        Args:
            path: Path to load from (defaults to config path)
        """
        load_path = path or self.config.index_path
        
        logger.info(f"Loading vector store from {load_path}")
        
        # Load metadata
        metadata_file = load_path / "metadata.pkl"
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        # Verify embedding dimension
        if metadata['embedding_dim'] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: {metadata['embedding_dim']} vs {self.embedding_dim}"
            )
        
        # Load FAISS index
        index_file = load_path / "faiss_index.bin"
        self.index = faiss.read_index(str(index_file))
        
        # Move to GPU if configured
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Could not move FAISS to GPU: {e}")
                self.use_gpu = False
        
        # Load chunks
        chunks_file = load_path / "chunks.pkl"
        with open(chunks_file, 'rb') as f:
            self.chunks = pickle.load(f)
        
        logger.info(f"Loaded {len(self.chunks)} chunks from vector store")
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        return {
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'use_gpu': self.use_gpu,
            'similarity_metric': self.config.similarity_metric
        }