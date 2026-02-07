"""
Core data models for RAG system.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v
    
    class Config:
        arbitrary_types_allowed = True


class Document(BaseModel):
    """Represents a source document."""
    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    source_path: str = Field(..., description="Path to source file")
    doc_type: str = Field(default="10-K", description="Type of document")
    company: str = Field(..., description="Company name (Apple or Tesla)")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="Document chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processed_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class RetrievedContext(BaseModel):
    """Represents retrieved context for a query."""
    chunk: DocumentChunk
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(..., ge=1, description="Rank in results")
    
    class Config:
        arbitrary_types_allowed = True


class RAGResponse(BaseModel):
    """Response from the RAG system."""
    question_id: Optional[int] = Field(None, description="Question ID")
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default_factory=list, description="Source citations")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    retrieved_contexts: Optional[List[RetrievedContext]] = Field(None, description="Retrieved contexts")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        arbitrary_types_allowed = True
        
    def to_json_output(self) -> Dict[str, Any]:
        """Convert to required JSON output format."""
        return {
            "question_id": self.question_id,
            "answer": self.answer,
            "sources": self.sources
        }


class Query(BaseModel):
    """Represents a user query."""
    text: str = Field(..., description="Query text")
    question_id: Optional[int] = Field(None, description="Question ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional query metadata")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query text cannot be empty")
        return v


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating RAG performance."""
    question_id: int
    exact_match: bool
    rouge_1: float
    rouge_l: float
    bleu: float
    answer_relevancy: float
    context_precision: float
    faithfulness: float
    has_hallucination: bool
    hallucination_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question_id": self.question_id,
            "exact_match": self.exact_match,
            "rouge_1": self.rouge_1,
            "rouge_l": self.rouge_l,
            "bleu": self.bleu,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "faithfulness": self.faithfulness,
            "has_hallucination": self.has_hallucination,
            "hallucination_details": self.hallucination_details
        }


class PromptTemplate(BaseModel):
    """Template for LLM prompts."""
    system_prompt: str = Field(..., description="System prompt")
    user_template: str = Field(..., description="User message template")
    
    def format(self, **kwargs) -> str:
        """Format the prompt with given variables."""
        return self.user_template.format(**kwargs)