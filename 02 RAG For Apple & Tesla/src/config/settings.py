"""
Configuration settings for the RAG system.
"""
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "cpu"  # or "cuda" if GPU available


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "])


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k: int = 10  # Initial retrieval
    top_n: int = 5   # After re-ranking
    similarity_threshold: float = 0.5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LLMConfig(BaseModel):
    """LLM configuration."""
    model_name: str = "llama3.2"  # Ollama model
    temperature: float = 0.1
    max_tokens: int = 500
    timeout: int = 60  # seconds


class PathConfig(BaseModel):
    """File path configuration."""
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Optional[Path] = None
    raw_data_dir: Optional[Path] = None
    processed_data_dir: Optional[Path] = None
    vector_store_path: Optional[Path] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set default paths relative to project root
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.raw_data_dir is None:
            self.raw_data_dir = self.data_dir / "raw"
        if self.processed_data_dir is None:
            self.processed_data_dir = self.data_dir / "processed"
        if self.vector_store_path is None:
            self.vector_store_path = self.processed_data_dir / "vector_store"
        
        # Create directories if they don't exist
        for path in [self.data_dir, self.raw_data_dir, self.processed_data_dir]:
            path.mkdir(parents=True, exist_ok=True)


class RAGConfig(BaseModel):
    """Main RAG system configuration."""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    
    # Document-specific settings
    apple_10k_filename: str = "10-Q4-2024-As-Filed.pdf"
    tesla_10k_filename: str = "tsla-20231231-gen.pdf"
    
    # Refusal message
    refusal_message: str = "This question cannot be answered based on the provided documents."
    
    class Config:
        arbitrary_types_allowed = True


# Singleton instance
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get or create the configuration singleton."""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def reset_config():
    """Reset configuration (useful for testing)."""
    global _config
    _config = None