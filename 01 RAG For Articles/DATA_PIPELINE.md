# Data Pipeline: Delta Table → RAG Application

This document explains how to fetch article data from Databricks Delta tables and use it in the RAG application.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATABRICKS CLUSTER                        │
│                      (DDA Cluster)                           │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Delta Table:                                       │    │
│  │  gold_us_prod.content.gld_cross_brand_live         │    │
│  │  (Architectural Digest articles)                    │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │                                        │
│                     │ PySpark Query                          │
│                     ▼                                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │  fetch_delta_data.py                               │    │
│  │  - Fetch 100 articles                              │    │
│  │  - Transform: hed→title, body→content              │    │
│  │  - Save to DBFS: /FileStore/rag_articles/          │    │
│  └──────────────────┬──────────────────────────────────┘    │
└────────────────────┼────────────────────────────────────────┘
                     │
                     │ Download
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL MACHINE                             │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  scripts/download_articles.py                      │    │
│  │  - Download from DBFS                              │    │
│  │  - Save to data/articles.json                      │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │                                        │
│                     ▼                                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │  data/articles.json                                │    │
│  │  [{"id": "...", "title": "...", ...}]              │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │                                        │
│                     │ Load                                   │
│                     ▼                                        │
│  ┌────────────────────────────────────────────────────┐    │
│  │  RAG Application (app.py)                          │    │
│  │  - Hybrid search                                   │    │
│  │  - Reranking                                       │    │
│  │  - GPT-4o-mini recommendations                     │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Guide

### 1. Fetch Data from Delta Table (Databricks)

**On Databricks:**

1. Open the notebook: `notebook/fetch_delta_data.py`
2. Attach to cluster: **DDA Cluster**
3. Run all cells

This will:
- Query `gold_us_prod.content.gld_cross_brand_live`
- Filter for Architectural Digest articles
- Transform columns: `hed → title`, `body → content`, `full_url → url`
- Save to DBFS: `/FileStore/rag_articles/articles.json`

### 2. Download to Local Machine

**Option A: Using Python Script (Recommended)**

```bash
cd "11-Side-Projects/01 RAG For Articles"
python scripts/download_articles.py
```

**Option B: Using Databricks CLI**

```bash
databricks fs cp dbfs:/FileStore/rag_articles/articles.json \
  "./11-Side-Projects/01 RAG For Articles/data/articles.json"
```

**Option C: Manual Download**

1. Go to Databricks UI → Data → DBFS
2. Navigate to: `/FileStore/rag_articles/articles.json`
3. Download the file
4. Move to: `data/articles.json`

### 3. Run the RAG Application

```bash
streamlit run app.py
```

The app will automatically load the new data from `data/articles.json`.

## Data Schema

### Delta Table Schema
```sql
SELECT 
    id,           -- Article ID
    hed,          -- Headline/Title
    body,         -- Article content
    full_url      -- Article URL
FROM gold_us_prod.content.gld_cross_brand_live
WHERE brand = 'Architectural Digest'
```

### JSON Schema (for RAG)
```json
[
  {
    "id": "12345",
    "title": "Article headline",
    "content": "Article body text...",
    "url": "https://..."
  }
]
```

## Modifying the Query

To fetch different data, edit `notebook/fetch_delta_data.py`:

```python
# Change the query
query = """
SELECT 
    id,
    hed,
    body,
    full_url
FROM gold_us_prod.content.gld_cross_brand_live
WHERE brand = 'Architectural Digest'
    AND publish_date >= '2024-01-01'  -- Add date filter
LIMIT 500  -- Increase limit
"""
```

## Refreshing Data

To get fresh data from Databricks:

1. Re-run the Databricks notebook
2. Re-run the download script
3. Restart the Streamlit app (it will reload the data)

## Troubleshooting

### "File not found" error
- Make sure you ran the Databricks notebook first
- Check DBFS path: `/FileStore/rag_articles/articles.json`

### Authentication errors
```bash
# Configure Databricks CLI
databricks auth login

# Or set environment variables
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-token"
```

### Empty results
- Check the WHERE clause in the SQL query
- Verify the brand name is correct
- Check for NULL values in required columns

