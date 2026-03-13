#!/usr/bin/env python3
"""
Generate evaluation questions based on the current articles dataset.
This script only generates questions - it doesn't run the full evaluation.
"""

import sys
sys.path.append('.')

from src.data_loader import load_articles
from src.evaluation import generate_test_questions

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATE EVALUATION QUESTIONS")
    print("=" * 60)
    
    # Load current articles
    print("\nLoading articles from data/articles.json...")
    articles = load_articles()
    print(f"✓ Loaded {len(articles)} articles")
    
    # Generate questions
    test_cases = generate_test_questions(articles, num_questions=12, save_to_file=True)
    
    if test_cases:
        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print(f"\nGenerated {len(test_cases)} test questions")
        print("Saved to: data/evaluation_questions.json")
        print("\nNext steps:")
        print("1. Review the generated questions in data/evaluation_questions.json")
        print("2. Run the full evaluation with:")
        print("   python src/evaluation.py")
        print("=" * 60)
    else:
        print("\n❌ Failed to generate questions")
        sys.exit(1)

