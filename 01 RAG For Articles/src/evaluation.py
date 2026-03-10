"""
RAGAS Evaluation for Interior Design RAG System

Evaluates the RAG pipeline on:
- Faithfulness: Is the response grounded in retrieved context?
- Answer Relevancy: Is the response relevant to the query?
- Context Precision: Are retrieved articles relevant to the query?
- Context Recall: Does context contain needed information?
"""

import os
import sys
from datetime import datetime

sys.path.append('.')

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.data_loader import load_articles
from src.rag_loader import HybridRAG


# Ground Truth Test Cases
GROUND_TRUTH = [
    {
        "question": "What are some ideas for pink dining rooms?",
        "ground_truth": "Blush pink walls with brass fixtures and velvet chairs create elegance. Millennial pink pairs well with crystal chandeliers. Pastel pink works for Scandinavian minimalist style.",
        "relevant_titles": ["Blush Pink Dining Rooms That Exude Elegance", "Millennial Pink Formal Dining Rooms", "Pastel Pink Scandinavian Dining Rooms"]
    },
    {
        "question": "How can I create a navy blue living room?",
        "ground_truth": "Deep navy blue walls create dramatic living spaces. Layer with cognac leather and brass accessories for sophistication.",
        "relevant_titles": ["Navy Blue Living Room Sophistication"]
    },
    {
        "question": "What colors work well for a bedroom retreat?",
        "ground_truth": "Emerald green creates luxurious bedroom retreats with gold fixtures. Blush pink bedrooms offer romantic atmospheres with white linens.",
        "relevant_titles": ["Emerald Green Bedroom Sanctuaries", "Blush Bedroom Retreats"]
    },
    {
        "question": "How do I design a modern kitchen?",
        "ground_truth": "Terracotta tiles bring Mediterranean charm paired with white cabinetry. Copper accents add warmth to kitchen spaces.",
        "relevant_titles": ["Terracotta Kitchen Design Ideas", "Copper Accent Kitchens"]
    },
    {
        "question": "What are trending dining chair styles?",
        "ground_truth": "Pink velvet dining chairs are statement pieces that mix well with glass or marble tables.",
        "relevant_titles": ["Pink Velvet Dining Chairs Trend"]
    },
    {
        "question": "How can I create a dramatic dining room?",
        "ground_truth": "Deep magenta accent walls transform traditional dining rooms. Black accent walls add drama balanced with light furniture.",
        "relevant_titles": ["Bold Magenta Dining Spaces", "Black Accent Wall Dining Spaces"]
    },
    {
        "question": "What are ideas for a home office?",
        "ground_truth": "Charcoal gray creates focused professional spaces with natural wood and plants. Pink home offices inspire creativity.",
        "relevant_titles": ["Charcoal Gray Home Offices", "Pink Home Office Creativity Boost"]
    },
    {
        "question": "How do I design a luxurious bathroom?",
        "ground_truth": "Carrara marble transforms bathrooms into spa-like sanctuaries. Add brass fixtures for warmth.",
        "relevant_titles": ["White Marble Bathroom Luxury"]
    },
    {
        "question": "What pink shades work for nurseries?",
        "ground_truth": "Modern pink nurseries use geometric patterns and mixed metals, moving beyond traditional designs.",
        "relevant_titles": ["Pink Nursery Design Trends"]
    },
    {
        "question": "How can I make a bold entryway?",
        "ground_truth": "Pink entryway walls make bold first impressions. Add console tables with marble tops.",
        "relevant_titles": ["Pink Entryway Statements"]
    },
    {
        "question": "What green colors work for dining rooms?",
        "ground_truth": "Sage green dining rooms feel fresh and organic with natural wood tables. Emerald green pairs with pink for a fresh palette.",
        "relevant_titles": ["Sage Green Dining Rooms", "Pink and Green Dining Room Palette"]
    },
    {
        "question": "How do I add gold accents to a room?",
        "ground_truth": "Cream walls with gold accents create timeless dining rooms. Rose gold lighting fixtures add warmth to neutral spaces.",
        "relevant_titles": ["Cream and Gold Dining Elegance", "Rose Gold Accents in Dining Areas"]
    },
]


def run_evaluation():
    """Run RAGAS evaluation on the RAG system."""
    print("=" * 60)
    print("RAGAS Evaluation - Interior Design RAG")
    print("=" * 60)
    
    # Initialize RAG
    print("\n[1/4] Loading RAG system...")
    articles = load_articles()
    rag = HybridRAG(articles)
    rag.populate_database()
    
    # Generate responses for each test case
    print("[2/4] Generating RAG responses...")
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for i, test_case in enumerate(GROUND_TRUTH):
        query = test_case["question"]
        print(f"  Processing {i+1}/{len(GROUND_TRUTH)}: {query[:40]}...")
        
        # Get RAG results
        results = rag.hybrid_search(query, top_k=5, alpha=0.5)
        
        if results:
            # Generate conversational response
            intro = rag.generate_conversation_intro(query, results)
            recommendations = [rag.generate_recommendation(a, query) for a in results[:3]]
            answer = intro + "\n\n" + "\n\n".join(recommendations)
            contexts = [f"{a['title']}: {a['content']}" for a in results]
        else:
            answer = rag.generate_no_results_response(query)
            contexts = ["No relevant articles found."]
        
        eval_data["question"].append(query)
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(test_case["ground_truth"])
    
    # Create dataset
    print("[3/4] Running RAGAS evaluation...")
    dataset = Dataset.from_dict(eval_data)
    
    # Run evaluation
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    print("[4/4] Evaluation complete!")
    return results


if __name__ == "__main__":
    results = run_evaluation()
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(results)

