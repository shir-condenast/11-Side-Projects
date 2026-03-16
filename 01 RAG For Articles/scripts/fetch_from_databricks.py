"""
Fetch article data directly from Databricks using SQL warehouse.
This script runs the query and saves the results to data/articles.json.
"""

import json
import os
from pathlib import Path
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

def fetch_articles():
    """Fetch articles from Databricks Delta table."""
    
    print("="*60)
    print("FETCHING ARTICLES FROM DATABRICKS")
    print("="*60)
    
    # Initialize Databricks client with WORKSHOP profile
    print("\n1. Connecting to Databricks...")
    config = Config(profile="WORKSHOP")
    w = WorkspaceClient(config=config)
    
    # Get available warehouses
    print("2. Finding SQL warehouse...")
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise Exception("No SQL warehouses found")
    
    # Use the first available warehouse
    warehouse = warehouses[0]
    print(f"   Using warehouse: {warehouse.name} (ID: {warehouse.id})")
    
    # SQL query
    query = """
    SELECT
        id,
        hed,
        body,
        full_url
    FROM gold_us_prod.content.gld_cross_brand_live
    WHERE brand = 'Architectural Digest'
        AND body IS NOT NULL
        AND hed IS NOT NULL
        AND full_url IS NOT NULL
        AND published_date >= '2026-01-01'
    

    """
    
    print("\n3. Executing query...")
    print(f"   Query: SELECT id, hed, body, full_url FROM gold_us_prod.content.gld_cross_brand_live")
    print(f"   Filter: brand = 'Architectural Digest' AND published_date >= '2026-01-01'")
    # print(f"   Limit: 100 articles")
    
    # Execute query
    result = w.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        statement=query,
        wait_timeout="30s"
    )
    
    # Check if query succeeded
    from databricks.sdk.service.sql import StatementState
    if result.status.state != StatementState.SUCCEEDED:
        error_msg = f"Query failed with state: {result.status.state}"
        if result.status.error:
            error_msg += f"\nError: {result.status.error.message}"
        raise Exception(error_msg)

    print(f"   ✅ Query completed successfully")
    
    # Parse results
    print("\n4. Processing results...")
    articles = []
    
    if result.result and result.result.data_array:
        # Get column names
        columns = [col.name for col in result.manifest.schema.columns]
        
        # Process each row
        for row in result.result.data_array:
            article = {}
            for i, col_name in enumerate(columns):
                article[col_name] = row[i]
            
            # Transform to expected schema
            url = article.get("full_url", "").strip()
            # Add https:// if missing
            if url and not url.startswith(("http://", "https://")):
                url = "https://" + url

            transformed = {
                "id": str(article.get("id", "")),
                "title": article.get("hed", "").strip(),
                "content": article.get("body", "").strip(),
                "url": url
            }
            
            # Only add if all fields are present
            if all(transformed.values()):
                articles.append(transformed)
    
    print(f"   Processed {len(articles)} articles")
    
    # Save to JSON
    print("\n5. Saving to file...")
    output_path = Path(__file__).parent.parent / "data" / "articles.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Saved to: {output_path}")
    
    # Show sample
    print("\n6. Sample article:")
    if articles:
        sample = articles[0]
        print(f"   ID: {sample['id']}")
        print(f"   Title: {sample['title'][:80]}...")
        print(f"   Content: {sample['content'][:100]}...")
        print(f"   URL: {sample['url']}")
    
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print(f"Total articles fetched: {len(articles)}")
    print(f"Output file: {output_path}")
    print("\nNext step: Run the Streamlit app")
    print("  streamlit run app.py")
    print("="*60)
    
    return articles

if __name__ == "__main__":
    try:
        articles = fetch_articles()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check Databricks authentication: databricks auth profiles")
        print("2. Verify table access: gold_us_prod.content.gld_cross_brand_live")
        print("3. Check SQL warehouse is running")
        raise

