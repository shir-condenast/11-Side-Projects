#!/usr/bin/env python3
"""
Fix URLs in articles.json by adding https:// if missing
"""
import json
import os

def fix_url(url):
    """Add https:// to URL if it doesn't have a protocol"""
    if not url:
        return url
    
    url = url.strip()
    if url and not url.startswith(('http://', 'https://')):
        return f"https://{url}"
    return url

def main():
    # Get the path to articles.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    json_file = os.path.join(project_dir, "data", "articles.json")
    
    print("=" * 60)
    print("FIXING URLs IN ARTICLES.JSON")
    print("=" * 60)
    print()
    
    # Read the JSON file
    print(f"1. Reading file: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"   Found {len(articles)} articles")
    print()
    
    # Fix URLs
    print("2. Fixing URLs...")
    fixed_count = 0
    for i, article in enumerate(articles):
        original_url = article.get('url', '')
        fixed_url = fix_url(original_url)

        if i < 3:  # Debug first 3
            print(f"   Article {i}: '{original_url}' -> '{fixed_url}'")

        if original_url != fixed_url:
            article['url'] = fixed_url
            fixed_count += 1

    print(f"   Fixed {fixed_count} URLs")
    print()
    
    # Save back to file
    print("3. Saving updated file...")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Saved to: {json_file}")
    print()
    
    # Show sample
    if articles:
        print("4. Sample article:")
        sample = articles[0]
        print(f"   Title: {sample.get('title', '')[:50]}...")
        print(f"   URL: {sample.get('url', '')}")
    
    print()
    print("=" * 60)
    print("✅ SUCCESS!")
    print("=" * 60)
    print(f"Total articles: {len(articles)}")
    print(f"URLs fixed: {fixed_count}")
    print("=" * 60)

if __name__ == "__main__":
    main()

