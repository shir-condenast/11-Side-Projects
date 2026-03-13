"""
Download articles.json from Databricks DBFS to local machine.

This script uses the Databricks SDK to download the articles.json file
that was created by the fetch_delta_data.py notebook.

Prerequisites:
1. Databricks CLI configured with authentication
2. Run fetch_delta_data.py notebook first on Databricks

Usage:
    python scripts/download_articles.py
"""

import os
from pathlib import Path
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

def download_articles():
    """Download articles.json from DBFS to local data directory."""
    
    # Paths
    dbfs_path = "/FileStore/rag_articles/articles.json"
    local_path = Path(__file__).parent.parent / "data" / "articles.json"
    
    print("="*60)
    print("DOWNLOADING ARTICLES FROM DATABRICKS")
    print("="*60)
    print(f"Source (DBFS): {dbfs_path}")
    print(f"Destination:   {local_path}")
    print()
    
    try:
        # Initialize Databricks client
        # This will use your ~/.databrickscfg or environment variables
        print("Connecting to Databricks...")
        w = WorkspaceClient()
        
        # Download file
        print("Downloading file...")
        with w.dbfs.download(dbfs_path).contents as remote_file:
            content = remote_file.read()
        
        # Ensure data directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to local file
        with open(local_path, 'wb') as f:
            f.write(content)
        
        print(f"✅ Successfully downloaded {len(content)} bytes")
        print(f"📁 Saved to: {local_path}")
        print()
        print("Next steps:")
        print("1. Run: streamlit run app.py")
        print("2. Your RAG app will now use the Databricks data!")
        
    except Exception as e:
        print(f"❌ Error downloading file: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure you've run fetch_delta_data.py notebook first")
        print("2. Check your Databricks authentication:")
        print("   - Run: databricks auth login")
        print("   - Or set DATABRICKS_HOST and DATABRICKS_TOKEN env vars")
        print("3. Verify the file exists in DBFS:")
        print(f"   - Check: {dbfs_path}")
        raise

if __name__ == "__main__":
    download_articles()

