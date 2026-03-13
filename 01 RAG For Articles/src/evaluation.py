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
import json
from datetime import datetime
from pathlib import Path

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

# Import OpenAI for question generation
from openai import OpenAI


def generate_test_questions(articles, num_questions=12, save_to_file=True):
    """
    Generate test questions based on the actual articles in the dataset.

    Args:
        articles: List of article dictionaries
        num_questions: Number of test questions to generate
        save_to_file: Whether to save generated questions to a JSON file

    Returns:
        List of test case dictionaries with question, ground_truth, and relevant_titles
    """
    print("=" * 60)
    print("GENERATING TEST QUESTIONS FROM NEW DATA")
    print("=" * 60)
    print(f"\nTotal articles available: {len(articles)}")
    print(f"Target questions: {num_questions}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Sample diverse articles for question generation
    import random
    sampled_articles = random.sample(articles, min(20, len(articles)))

    # Create a summary of available articles
    article_summaries = []
    for i, article in enumerate(sampled_articles[:10], 1):
        title = article.get('title', 'Untitled')
        content_preview = article.get('content', '')[:200]
        article_summaries.append(f"{i}. {title}\n   Preview: {content_preview}...")

    articles_context = "\n\n".join(article_summaries)

    prompt = f"""Based on these Architectural Digest articles, generate {num_questions} diverse test questions for evaluating a RAG system.

Available Articles:
{articles_context}

For each question, provide:
1. A natural user question about interior design, architecture, or home decor
2. A brief ground truth answer based on the article content
3. The title(s) of relevant articles

Format your response as a JSON array with this structure:
[
  {{
    "question": "What are some shower tile ideas?",
    "ground_truth": "Modern shower tiles include geometric patterns, natural stone, and colorful mosaics. Popular choices are subway tiles, hexagonal tiles, and large format porcelain.",
    "relevant_titles": ["52 Refreshing Shower Tile Ideas to Wake Up Your Bathroom"]
  }}
]

Make questions diverse covering: design styles, color schemes, room types, materials, trends, and specific design elements.
Ensure questions are answerable from the article content provided."""

    print("\n[1/3] Generating questions using GPT-4o-mini...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in interior design and architecture. Generate high-quality test questions for RAG evaluation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    # Parse the response
    try:
        result_text = response.choices[0].message.content
        # Handle both direct array and object with array
        if result_text.strip().startswith('['):
            test_cases = json.loads(result_text)
        else:
            parsed = json.loads(result_text)
            test_cases = parsed.get('questions', parsed.get('test_cases', []))

        print(f"[2/3] Generated {len(test_cases)} test questions")

        # Save to file if requested
        if save_to_file:
            output_dir = Path("data")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / "evaluation_questions.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(test_cases, f, indent=2, ensure_ascii=False)

            print(f"[3/3] Saved questions to: {output_file}")

        # Display sample questions
        print("\n" + "=" * 60)
        print("SAMPLE GENERATED QUESTIONS")
        print("=" * 60)
        for i, tc in enumerate(test_cases[:3], 1):
            print(f"\n{i}. {tc['question']}")
            print(f"   Ground Truth: {tc['ground_truth'][:80]}...")

        return test_cases

    except json.JSONDecodeError as e:
        print(f"Error parsing GPT response: {e}")
        print(f"Response: {result_text[:500]}")
        return []


def load_or_generate_questions(articles, force_regenerate=False):
    """
    Load existing test questions or generate new ones.

    Args:
        articles: List of article dictionaries
        force_regenerate: If True, always generate new questions

    Returns:
        List of test case dictionaries
    """
    questions_file = Path("data/evaluation_questions.json")

    if not force_regenerate and questions_file.exists():
        print(f"\n📂 Loading existing questions from: {questions_file}")
        with open(questions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("\n🔄 Generating new test questions...")
        return generate_test_questions(articles, num_questions=12)


def run_evaluation(test_cases, pause_before_scoring=True):
    """
    Run RAGAS evaluation on the RAG system.

    Args:
        test_cases: List of test case dictionaries
        pause_before_scoring: If True, pause before running RAGAS scoring

    Returns:
        RAGAS evaluation results
    """
    print("\n" + "=" * 60)
    print("RAGAS Evaluation - Interior Design RAG")
    print("=" * 60)

    # Initialize RAG
    print("\n[1/5] Loading RAG system...")
    articles = load_articles()
    print(f"   Loaded {len(articles)} articles")
    rag = HybridRAG(articles)
    rag.populate_database()
    print("   ✓ RAG system initialized")

    # Generate responses for each test case
    print(f"\n[2/5] Generating RAG responses for {len(test_cases)} questions...")
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for i, test_case in enumerate(test_cases):
        query = test_case["question"]
        print(f"  Processing {i+1}/{len(test_cases)}: {query[:50]}...")

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

    print("   ✓ All responses generated")

    # Save intermediate results
    print("\n[3/5] Saving intermediate results...")
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    intermediate_file = output_dir / "evaluation_responses.json"
    with open(intermediate_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved to: {intermediate_file}")

    # Pause before scoring if requested
    if pause_before_scoring:
        print("\n" + "=" * 60)
        print("⏸️  PAUSED BEFORE SCORING")
        print("=" * 60)
        print("\nGenerated responses are ready for review.")
        print(f"Review file: {intermediate_file}")
        print("\nPress ENTER to continue with RAGAS scoring, or Ctrl+C to exit...")
        print("=" * 60)
        try:
            input()
        except KeyboardInterrupt:
            print("\n\n❌ Evaluation cancelled by user.")
            return None

    # Create dataset
    print("\n[4/5] Running RAGAS evaluation...")
    print("   This may take a few minutes...")
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

    print("   ✓ RAGAS scoring complete!")

    # Save final results
    print("\n[5/5] Saving final results...")
    results_file = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert RAGAS results - it has dict-like access via []
    metrics_dict = {}
    for key in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        try:
            value = results[key]
            # Handle both scalar and list values
            if isinstance(value, (list, tuple)):
                metrics_dict[key] = float(sum(value) / len(value)) if value else 0.0
            else:
                metrics_dict[key] = float(value) if value is not None else 0.0
        except (KeyError, TypeError, AttributeError) as e:
            print(f"   Warning: Could not extract {key}: {e}")
            metrics_dict[key] = 0.0

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(test_cases),
            "metrics": metrics_dict
        }, f, indent=2)
    print(f"   ✓ Saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation with RAGAS")
    parser.add_argument("--regenerate", action="store_true",
                       help="Force regenerate test questions from current data")
    parser.add_argument("--no-pause", action="store_true",
                       help="Skip pause before scoring (run end-to-end)")
    args = parser.parse_args()

    # Load articles
    print("Loading articles from data/articles.json...")
    articles = load_articles()

    # Load or generate test questions
    test_cases = load_or_generate_questions(articles, force_regenerate=args.regenerate)

    if not test_cases:
        print("\n❌ No test questions available. Exiting.")
        sys.exit(1)

    # Run evaluation
    results = run_evaluation(test_cases, pause_before_scoring=not args.no_pause)

    if results:
        print("\n" + "=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)

        # Extract values safely using [] access
        def get_metric_value(key):
            try:
                value = results[key]
                if isinstance(value, (list, tuple)):
                    return float(sum(value) / len(value)) if value else 0.0
                return float(value) if value is not None else 0.0
            except (KeyError, TypeError, AttributeError):
                return 0.0

        faithfulness = get_metric_value('faithfulness')
        answer_relevancy = get_metric_value('answer_relevancy')
        context_precision = get_metric_value('context_precision')
        context_recall = get_metric_value('context_recall')

        print(f"\nFaithfulness:       {faithfulness:.4f}")
        print(f"Answer Relevancy:   {answer_relevancy:.4f}")
        print(f"Context Precision:  {context_precision:.4f}")
        print(f"Context Recall:     {context_recall:.4f}")
        print("\n" + "=" * 60)
        print("✅ Evaluation complete!")
        print("=" * 60)

