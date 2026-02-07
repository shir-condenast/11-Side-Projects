"""
Configuration management for RAG system.
Handles all system parameters and settings.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_embeddings: bool = True


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_size: int = 100
    strategy: str = "semantic"  # semantic, fixed, recursive


@dataclass
class VectorStoreConfig:
    """Configuration for vector database."""
    store_type: str = "faiss"  # faiss, chroma, qdrant
    index_path: Path = Path("data/vector_index")
    use_gpu: bool = torch.cuda.is_available()
    similarity_metric: str = "cosine"  # cosine, l2, ip


@dataclass
class RetrieverConfig:
    """Configuration for retrieval pipeline."""
    top_k: int = 5
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3
    similarity_threshold: float = 0.3


@dataclass
class LLMConfig:
    """Configuration for LLM."""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    load_in_4bit: bool = True  # Use 4-bit quantization for efficiency
    use_flash_attention: bool = True


@dataclass
class RAGConfig:
    """Master configuration for RAG system."""
    # Sub-configurations
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Data paths
    data_dir: Path = Path("data")
    pdf_dir: Path = Path("data/pdfs")
    index_dir: Path = Path("data/vector_index")
    cache_dir: Path = Path("data/cache")
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = Path("logs/rag_system.log")
    
    # System
    seed: int = 42
    use_gpu: bool = torch.cuda.is_available()
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


def get_default_config() -> RAGConfig:
    """Get default configuration."""
    return RAGConfig()


def get_gpu_optimized_config() -> RAGConfig:
    """Get GPU-optimized configuration."""
    config = RAGConfig()
    
    if torch.cuda.is_available():
        # Optimize for GPU
        config.embedding.batch_size = 64
        config.embedding.device = "cuda"
        config.vector_store.use_gpu = True
        config.llm.device = "cuda"
        config.llm.load_in_4bit = True
        config.llm.use_flash_attention = True
        
        # Larger chunks for GPU processing
        config.chunking.chunk_size = 768
        config.chunking.chunk_overlap = 192
        
    return config