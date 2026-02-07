"""
Main interface for RAG system.
Provides the required answer_question() function.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

from src.config import RAGConfig, get_gpu_optimized_config
from src.pipeline.rag_pipeline import RAGPipeline


# Global pipeline instance
_pipeline = None


def initialize_pipeline(config: RAGConfig = None, pdf_dir: Path = None):
    """
    Initialize the RAG pipeline with documents.
    
    Args:
        config: Optional RAG configuration
        pdf_dir: Directory containing PDF files
    """
    global _pipeline
    
    if config is None:
        config = get_gpu_optimized_config()
    
    _pipeline = RAGPipeline(config)
    
    # Get PDF files
    if pdf_dir is None:
        pdf_dir = Path("data/pdfs")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        logger.info("Please place your 10-K PDF files in the data/pdfs directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Index documents
    _pipeline.index_documents(pdf_files)
    
    logger.info("Pipeline initialized and ready!")


def answer_question(query: str) -> dict:
    """
    Answers a question using the RAG pipeline.
    
    This is the required interface function that must be implemented.
    
    Args:
        query (str): The user question about Apple or Tesla 10-K filings.
    
    Returns:
        dict: {
            "answer": "Answer text or 'This question cannot be answered based on the provided documents.'",
            "sources": ["Apple 10-K", "Item 8", "p. 28"]  # Empty list if refused
        }
    """
    global _pipeline
    
    if _pipeline is None:
        raise RuntimeError(
            "Pipeline not initialized. Call initialize_pipeline() first."
        )
    
    result = _pipeline.answer_question(query)
    return result


def answer_questions_batch(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Answer multiple questions in batch.
    
    Args:
        questions: List of question dictionaries with 'question_id' and 'question'
    
    Returns:
        List of answer dictionaries with 'question_id', 'answer', and 'sources'
    """
    global _pipeline
    
    if _pipeline is None:
        raise RuntimeError(
            "Pipeline not initialized. Call initialize_pipeline() first."
        )
    
    results = _pipeline.batch_answer_questions(questions)
    return results


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """
    Save results to JSON file.
    
    Args:
        results: List of answer dictionaries
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """Main execution function."""
    # Configure logging
    logger.add(
        "logs/rag_system.log",
        rotation="100 MB",
        retention="10 days",
        level="INFO"
    )
    
    logger.info("="*80)
    logger.info("RAG System - Starting")
    logger.info("="*80)
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    initialize_pipeline()
    
    # Load test questions
    test_questions = [
        {"question_id": 1, "question": "What was Apples total revenue for the fiscal year ended September 28, 2024?"},
        {"question_id": 2, "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
        {"question_id": 3, "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
        {"question_id": 4, "question": "On what date was Apples 10-K report for 2024 signed and filed with the SEC?"},
        {"question_id": 5, "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
        {"question_id": 6, "question": "What was Teslas total revenue for the year ended December 31, 2023?"},
        {"question_id": 7, "question": "What percentage of Teslas total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
        {"question_id": 8, "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
        {"question_id": 9, "question": "What types of vehicles does Tesla currently produce and deliver?"},
        {"question_id": 10, "question": "What is the purpose of Teslas 'lease pass-through fund arrangements'?"},
        {"question_id": 11, "question": "What is Teslas stock price forecast for 2025?"},
        {"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
        {"question_id": 13, "question": "What color is Teslas headquarters painted?"}
    ]
    
    # Answer questions
    logger.info(f"Answering {len(test_questions)} test questions...")
    results = answer_questions_batch(test_questions)
    
    # Save results
    output_path = Path("outputs/rag_results.json")
    save_results(results, output_path)
    
    # Print summary
    logger.info("="*80)
    logger.info("Results Summary")
    logger.info("="*80)
    
    for result in results:
        logger.info(f"Q{result['question_id']}: {result['answer'][:100]}...")
    
    logger.info("="*80)
    logger.info("Processing complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()