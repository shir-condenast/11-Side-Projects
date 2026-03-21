#!/usr/bin/env python3
"""Test script to debug search deduplication issues"""

from src.data_loader import load_articles
from infra.bootstrap import get_rag

def test_search(query: str, top_k: int = 10):
    """Test search with a specific query and show detailed results"""
    print(f"\n{'='*80}")
    print(f"Testing query: '{query}'")
    print(f"{'='*80}\n")
    
    # Load articles and initialize RAG
    articles = load_articles()
    print(f"Loaded {len(articles)} articles")
    
    rag = get_rag(articles)
    
    # Perform search
    results = rag.hybrid_search(query=query, top_k=top_k, alpha=0.5)
    
    print(f"\nFound {len(results)} results:\n")
    
    # Track seen titles to detect duplicates
    seen_titles = set()
    duplicates = []
    
    for idx, article in enumerate(results, 1):
        title = article.get('title', 'No title')
        article_id = article.get('id', 'No ID')
        url = article.get('url', 'No URL')
        
        # Check for duplicates
        if title in seen_titles:
            duplicates.append((idx, title))
            print(f"⚠️  DUPLICATE FOUND!")
        
        seen_titles.add(title)
        
        print(f"{idx}. {title}")
        print(f"   ID: {article_id}")
        print(f"   URL: {url[:80]}..." if len(url) > 80 else f"   URL: {url}")
        print(f"   Hybrid Score: {article.get('hybrid_score', 'N/A')}")
        print(f"   Rerank Score: {article.get('rerank_score', 'N/A')}")
        print(f"   Keywords: {article.get('keywords', [])}")
        print()
    
    if duplicates:
        print(f"\n⚠️  WARNING: Found {len(duplicates)} duplicate(s):")
        for idx, title in duplicates:
            print(f"   - Position {idx}: {title}")
    else:
        print("✅ No duplicates found!")
    
    return results

if __name__ == "__main__":
    # Test with "red wall" query
    test_search("red wall", top_k=10)

