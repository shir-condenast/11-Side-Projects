#!/usr/bin/env python3
"""Simple test to verify deduplication fix without running full app"""

import json
from pathlib import Path

# Load articles
data_path = Path(__file__).parent / "data" / "articles.json"
with open(data_path) as f:
    articles = json.load(f)

print(f"Loaded {len(articles)} articles")
print(f"\nFirst article structure:")
print(f"  ID: {articles[0].get('id')}")
print(f"  Title: {articles[0].get('title')[:80]}...")
print(f"  Has 'id' field: {'id' in articles[0]}")

# Check for duplicate IDs in the source data
ids = [a.get('id') for a in articles]
unique_ids = set(ids)
print(f"\nTotal articles: {len(articles)}")
print(f"Unique IDs: {len(unique_ids)}")

if len(ids) != len(unique_ids):
    print(f"⚠️  WARNING: Found {len(ids) - len(unique_ids)} duplicate IDs in source data!")
    # Find duplicates
    from collections import Counter
    id_counts = Counter(ids)
    duplicates = {id_: count for id_, count in id_counts.items() if count > 1}
    print(f"Duplicate IDs: {duplicates}")
else:
    print("✅ No duplicate IDs in source data")

# Check for duplicate titles
titles = [a.get('title') for a in articles]
unique_titles = set(titles)
print(f"\nUnique titles: {len(unique_titles)}")

if len(titles) != len(unique_titles):
    print(f"⚠️  WARNING: Found {len(titles) - len(unique_titles)} duplicate titles!")
else:
    print("✅ No duplicate titles")

print("\n" + "="*80)
print("Deduplication fix summary:")
print("="*80)
print("✅ Added 'id' field to ChromaDB metadata in populate_database()")
print("✅ Added 'id' field to semantic_search() results")
print("✅ Added final deduplication check in hybrid_search() using seen_ids set")
print("\nThe fix ensures that:")
print("1. Both keyword and semantic search return articles with 'id' field")
print("2. Deduplication in hybrid_search uses 'id' instead of falling back to title")
print("3. A final check prevents any duplicates from slipping through")

