# RAG for Articles - Interior Design Consultant

A hybrid RAG (Retrieval-Augmented Generation) system for interior design article recommendations, powered by Databricks Delta tables.

## 🌟 Features

- **Hybrid Search**: Combines keyword-based (sparse) and semantic (dense) search
- **Reranking**: CrossEncoder reranking for improved relevance
- **Personalized Recommendations**: GPT-4o-mini generates contextual recommendations
- **Databricks Integration**: Fetches real article data from Delta tables
- **Interactive UI**: Streamlit-based chat interface

## 🏗️ Architecture

```
Databricks (DDA Cluster)
    ↓ PySpark
Delta Table → articles.json
    ↓ Python
RAG Pipeline (ChromaDB + OpenAI)
    ↓
Streamlit App
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
uv sync

# Setup Databricks SDK
bash scripts/setup_databricks.sh

# Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Fetch Data from Databricks

**On Databricks:**
1. Open `notebook/fetch_delta_data.py`
2. Attach to cluster: **DDA Cluster**
3. Run all cells

**On Local Machine:**
```bash
# Download the data
python scripts/download_articles.py
```

See [DATA_PIPELINE.md](DATA_PIPELINE.md) for detailed instructions.

### 3. Run the Application

```bash
streamlit run app.py
```

## 📁 Project Structure

```
.
├── app.py                      # Streamlit application
├── src/
│   ├── data_loader.py         # Load articles from JSON
│   ├── rag_loader.py          # Hybrid RAG implementation
│   └── evaluation.py          # Evaluation metrics
├── infra/
│   ├── bootstrap.py           # RAG initialization
│   └── state.py               # State management
├── notebook/
│   ├── fetch_delta_data.py    # Databricks notebook (PySpark)
│   └── EDA.ipynb              # Exploratory data analysis
├── scripts/
│   ├── download_articles.py   # Download from DBFS
│   └── setup_databricks.sh    # Setup script
├── data/
│   └── articles.json          # Article data (generated)
└── DATA_PIPELINE.md           # Data pipeline documentation
```

## 🔧 Configuration

### Search Parameters

Adjust in the Streamlit sidebar:
- **Number of results**: 1-10 articles
- **Search weight (alpha)**:
  - 0.0 = Pure keyword search
  - 0.5 = Balanced hybrid
  - 1.0 = Pure semantic search

### Data Source

Edit `notebook/fetch_delta_data.py` to modify:
- Brand filter
- Date range
- Article limit
- Additional filters

## 📊 How It Works

### 1. Data Fetching (PySpark)
```python
# Query Delta table
SELECT id, hed, body, full_url
FROM gold_us_prod.content.gld_cross_brand_live
WHERE brand = 'Architectural Digest'
LIMIT 100
```

### 2. Hybrid Search
- **Keyword Search**: TF-IDF-like scoring
- **Semantic Search**: Sentence transformers embeddings
- **Fusion**: Weighted combination (alpha parameter)

### 3. Reranking
- CrossEncoder model: `ms-marco-MiniLM-L-6-v2`
- Reranks top candidates for better relevance

### 4. Recommendation Generation
- GPT-4o-mini generates personalized recommendations
- Grounded in article content (no hallucination)

## 🔄 Refreshing Data

To get fresh articles from Databricks:

```bash
# 1. Re-run Databricks notebook
# 2. Download new data
python scripts/download_articles.py

# 3. Restart Streamlit app
streamlit run app.py
```

## 📚 Documentation

- [DATA_PIPELINE.md](DATA_PIPELINE.md) - Data pipeline details
- [EVALUATION.md](EVALUATION.md) - Evaluation metrics
- [RANKING_VS_RERANKING.md](RANKING_VS_RERANKING.md) - Ranking strategies
- [RERANKING_IMPLEMENTATION.md](RERANKING_IMPLEMENTATION.md) - Reranking details
- [SEARCH_STRATEGIES_COMPARISON.md](SEARCH_STRATEGIES_COMPARISON.md) - Search comparison

## 🛠️ Tech Stack

- **Data**: Databricks Delta Lake, PySpark
- **Vector DB**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Reranking**: CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **LLM**: OpenAI GPT-4o-mini
- **UI**: Streamlit
- **Package Manager**: uv

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
```bash
# Add to .env file
OPENAI_API_KEY=sk-...
```

### "File not found: articles.json"
```bash
# Run the data pipeline
python scripts/download_articles.py
```

### Databricks authentication errors
```bash
# Configure Databricks CLI
databricks auth login
```

## 📝 License

MIT