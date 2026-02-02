"""
Main entry point for the RAG system.

Usage:
    python main.py --mode index
    python main.py --mode query --question "What was Apple's revenue?"
    python main.py --mode evaluate --output results.json
"""
import argparse
import json
import sys
from pathlib import Path
from loguru import logger
# from src.rag_pipeline import RAGPipeline
from src.config import get_config


# Test questions from assignment
TEST_QUESTIONS = [
    {"question_id": 1, "question": "What was Apple's total revenue for the fiscal year ended September 28, 2024?"},
    {"question_id": 2, "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
    {"question_id": 3, "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
    {"question_id": 4, "question": "On what date was Apple's 10-K report for 2024 signed and filed with the SEC?"},
    {"question_id": 5, "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
    {"question_id": 6, "question": "What was Tesla's total revenue for the year ended December 31, 2023?"},
    {"question_id": 7, "question": "What percentage of Tesla's total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
    {"question_id": 8, "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
    {"question_id": 9, "question": "What types of vehicles does Tesla currently produce and deliver?"},
    {"question_id": 10, "question": "What is the purpose of Tesla's 'lease pass-through fund arrangements'?"},
    {"question_id": 11, "question": "What is Tesla's stock price forecast for 2025?"},
    {"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
    {"question_id": 13, "question": "What color is Tesla's headquarters painted?"}
]


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level
    )


def mode_index(args):
    """Index documents."""
    logger.info("=" * 60)
    logger.info("INDEXING MODE")
    logger.info("=" * 60)
    
    config = get_config()
    
    # Check PDFs exist
    apple_pdf = config.paths.raw_data_dir / config.apple_10k_filename
    tesla_pdf = config.paths.raw_data_dir / config.tesla_10k_filename
    
    if not apple_pdf.exists():
        logger.error(f"Apple PDF not found: {apple_pdf}")
        logger.info(f"Please place the PDF in: {config.paths.raw_data_dir}")
        return 1
    
    if not tesla_pdf.exists():
        logger.error(f"Tesla PDF not found: {tesla_pdf}")
        logger.info(f"Please place the PDF in: {config.paths.raw_data_dir}")
        return 1
    
    logger.info(f"Apple PDF: {apple_pdf}")
    logger.info(f"Tesla PDF: {tesla_pdf}")
    
    # Create pipeline
    pipeline = RAGPipeline(
        use_reranking=not args.no_rerank,
        use_hybrid_chunking=not args.simple_chunking
    )
    
    # Index
    try:
        pipeline.index_documents(
            apple_pdf_path=apple_pdf,
            tesla_pdf_path=tesla_pdf,
            force_reindex=args.force
        )
        logger.success("✓ Indexing complete!")
        return 0
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return 1


def mode_query(args):
    """Answer a single question."""
    logger.info("=" * 60)
    logger.info("QUERY MODE")
    logger.info("=" * 60)
    
    if not args.question:
        logger.error("--question is required for query mode")
        return 1
    
    # Create pipeline
    pipeline = RAGPipeline(
        use_reranking=not args.no_rerank
    )
    
    # Answer question
    logger.info(f"Question: {args.question}")
    logger.info("-" * 60)
    
    try:
        result = pipeline.answer_question(args.question)
        
        print("\nANSWER:")
        print(result["answer"])
        print("\nSOURCES:")
        if result["sources"]:
            for i, source in enumerate(result["sources"], 1):
                print(f"  {i}. {source}")
        else:
            print("  (none)")
        
        return 0
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return 1


def mode_evaluate(args):
    """Evaluate on all test questions."""
    logger.info("=" * 60)
    logger.info("EVALUATION MODE")
    logger.info("=" * 60)
    
    # Create pipeline
    pipeline = RAGPipeline(
        use_reranking=not args.no_rerank
    )
    
    # Answer all questions
    logger.info(f"Processing {len(TEST_QUESTIONS)} questions...")
    
    try:
        results = pipeline.batch_answer(TEST_QUESTIONS)
        
        # Save to file
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.success(f"✓ Results saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        refusals = sum(
            1 for r in results
            if "cannot be answered" in r["answer"].lower()
        )
        
        print(f"Total questions: {len(results)}")
        print(f"Answered: {len(results) - refusals}")
        print(f"Refused: {refusals}")
        
        # Show sample answers
        print("\nSample Answers:")
        for i in range(min(3, len(results))):
            r = results[i]
            print(f"\nQ{r['question_id']}: {TEST_QUESTIONS[i]['question'][:60]}...")
            print(f"A: {r['answer'][:100]}...")
        
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


def mode_interactive(args):
    """Interactive Q&A mode."""
    logger.info("=" * 60)
    logger.info("INTERACTIVE MODE")
    logger.info("=" * 60)
    logger.info("Type 'quit' or 'exit' to end session\n")
    
    # Create pipeline
    pipeline = RAGPipeline(
        use_reranking=not args.no_rerank
    )
    
    while True:
        try:
            question = input("\n💬 Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("Goodbye!")
                break
            
            if not question:
                continue
            
            # Answer
            result = pipeline.answer_question(question)
            
            print(f"\n🤖 Answer: {result['answer']}")
            if result['sources']:
                print(f"\n📚 Sources: {', '.join(result['sources'][:3])}")
            
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System for Financial Document Q&A"
    )
    
    parser.add_argument(
        "--mode",
        choices=["index", "query", "evaluate", "interactive"],
        required=True,
        help="Operation mode"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        help="Question to answer (for query mode)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file for evaluation results"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if index exists"
    )
    
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable re-ranking"
    )
    
    parser.add_argument(
        "--simple-chunking",
        action="store_true",
        help="Use simple chunking instead of hybrid"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Route to appropriate mode
    if args.mode == "index":
        return mode_index(args)
    elif args.mode == "query":
        return mode_query(args)
    elif args.mode == "evaluate":
        return mode_evaluate(args)
    elif args.mode == "interactive":
        return mode_interactive(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())