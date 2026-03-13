# Getting Started - Databricks Integration

This guide will help you set up the RAG application with Databricks Delta table integration.

## 🎯 What Changed?

Previously, the app used static JSON data. Now it fetches real article data from Databricks Delta tables!

**Old Flow:**
```
Static JSON → RAG App
```

**New Flow:**
```
Delta Table → PySpark → DBFS → Local JSON → RAG App
```

## ⚡ Quick Start (5 minutes)

### Step 1: Install Databricks SDK

```bash
cd "11-Side-Projects/01 RAG For Articles"

# Using uv (recommended)
uv pip install databricks-sdk

# Or using pip
pip install databricks-sdk
```

### Step 2: Configure Databricks Authentication

**Option A: Interactive Login (Easiest)**
```bash
databricks auth login
```

**Option B: Environment Variables**
```bash
# Add to your .env file
export DATABRICKS_HOST="https://your-workspace.databricks.com"
export DATABRICKS_TOKEN="your-personal-access-token"
```

### Step 3: Run the Data Fetch Notebook

1. Go to your Databricks workspace
2. Import the notebook: `notebook/fetch_delta_data.py`
3. Attach to cluster: **DDA Cluster**
4. Click "Run All"

This will:
- Query 100 Architectural Digest articles
- Transform the data (hed → title, body → content)
- Save to DBFS: `/FileStore/rag_articles/articles.json`

### Step 4: Download the Data

```bash
python scripts/download_articles.py
```

This downloads the file from DBFS to `data/articles.json`.

### Step 5: Run the App

```bash
streamlit run app.py
```

🎉 Done! Your app now uses real Databricks data!

## 📋 Detailed Workflow

### On Databricks (One-time or when refreshing data)

```python
# notebook/fetch_delta_data.py does this:

# 1. Query Delta table
df = spark.sql("""
    SELECT id, hed, body, full_url
    FROM gold_us_prod.content.gld_cross_brand_live
    WHERE brand = 'Architectural Digest'
    LIMIT 100
""")

# 2. Transform columns
transformed = df.select(
    col("id").cast("string"),
    col("hed").alias("title"),
    col("body").alias("content"),
    col("full_url").alias("url")
)

# 3. Save to DBFS
# Output: /FileStore/rag_articles/articles.json
```

### On Local Machine

```bash
# Download from DBFS
python scripts/download_articles.py

# Run the app
streamlit run app.py
```

## 🔄 Modifying the Data Query

Want different articles? Edit `notebook/fetch_delta_data.py`:

```python
# Change the SQL query
query = """
SELECT id, hed, body, full_url
FROM gold_us_prod.content.gld_cross_brand_live
WHERE brand = 'Architectural Digest'
    AND publish_date >= '2024-01-01'  -- Add date filter
    AND body IS NOT NULL
LIMIT 500  -- Get more articles
"""
```

Then re-run the notebook and download script.

## 🔍 Verifying the Setup

### Check if data exists in DBFS

```bash
databricks fs ls dbfs:/FileStore/rag_articles/
```

### Check local data file

```bash
ls -lh data/articles.json
```

### Test the download script

```bash
python scripts/download_articles.py
```

Expected output:
```
============================================================
DOWNLOADING ARTICLES FROM DATABRICKS
============================================================
Source (DBFS): /FileStore/rag_articles/articles.json
Destination:   /path/to/data/articles.json

Connecting to Databricks...
Downloading file...
✅ Successfully downloaded 123456 bytes
📁 Saved to: /path/to/data/articles.json
```

## 🐛 Common Issues

### Issue: "Authentication failed"

**Solution:**
```bash
# Re-authenticate
databricks auth login

# Or check your token
echo $DATABRICKS_TOKEN
```

### Issue: "File not found in DBFS"

**Solution:**
1. Make sure you ran the Databricks notebook first
2. Check the DBFS path in the notebook output
3. Verify: `databricks fs ls dbfs:/FileStore/rag_articles/`

### Issue: "No module named 'databricks'"

**Solution:**
```bash
uv pip install databricks-sdk
```

### Issue: "OPENAI_API_KEY not found"

**Solution:**
```bash
# Create .env file
cp .env.example .env

# Edit .env and add:
OPENAI_API_KEY=sk-your-key-here
```

## 📊 Data Schema

The notebook transforms Delta table columns to match the RAG schema:

| Delta Table | → | RAG Schema |
|-------------|---|------------|
| `id`        | → | `id`       |
| `hed`       | → | `title`    |
| `body`      | → | `content`  |
| `full_url`  | → | `url`      |

## 🎓 Next Steps

1. **Customize the query**: Edit the SQL in `fetch_delta_data.py`
2. **Schedule refreshes**: Set up a Databricks job to run the notebook daily
3. **Automate downloads**: Create a cron job to run `download_articles.py`
4. **Explore the data**: Use `notebook/EDA.ipynb` for analysis

## 📚 Additional Resources

- [DATA_PIPELINE.md](DATA_PIPELINE.md) - Complete pipeline documentation
- [README.md](README.md) - Full project documentation
- Databricks SDK Docs: https://docs.databricks.com/dev-tools/sdk-python.html

## 💡 Tips

- **Development**: Use LIMIT 10 in the query for faster iteration
- **Production**: Increase LIMIT or remove it for full dataset
- **Performance**: The download script caches data locally, so the app starts fast
- **Freshness**: Re-run the pipeline whenever you need updated articles

