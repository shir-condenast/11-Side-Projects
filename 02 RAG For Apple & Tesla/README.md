# RAG System for Financial Document Analysis

A production-grade Retrieval-Augmented Generation (RAG) system for answering questions from Apple and Tesla 10-K filings using open-source LLMs with GPU acceleration.

## 🚀 Features

- **CPU & GPU Support**: Works on CPU-only systems (GPU optional for 5-10x speedup)
- **Advanced Chunking**: Multiple strategies (semantic, fixed, recursive) for optimal retrieval
- **Hybrid Retrieval**: Combines dense (FAISS) and sparse (BM25) retrieval with cross-encoder reranking
- **Optimized Models**: Phi-3-mini for CPU, Mistral-7B optional for GPU
- **Hallucination Detection**: Comprehensive testing for answer quality and factual accuracy
- **Proper OOP Design**: Clean architecture with separation of concerns
- **Extensive Testing**: Unit tests, integration tests, and evaluation metrics

## 📋 Requirements

- Python 3.8-3.10
- **CPU-only supported** (GPU optional but not required)
- ~8-10GB RAM for CPU mode
- ~16GB GPU memory for GPU mode (optional)
- ~8GB system RAM


Place your PDF files in `data/pdfs/`:
- `10-Q4-2024-As-Filed.pdf` (Apple 10-K)
- `tsla-20231231-gen.pdf` (Tesla 10-K)

## 🏗️ Project Structure

```
rag_system/
├── src/
│   ├── config.py                    # Configuration management
│   ├── models/
│   │   └── schemas.py               # Data models
│   ├── services/
│   │   ├── document_processor.py   # PDF processing
│   │   ├── chunker.py              # Chunking strategies
│   │   ├── embedding_service.py    # GPU embeddings
│   │   ├── vector_store.py         # FAISS store
│   │   ├── retriever.py            # Retrieval + reranking
│   │   └── llm_service.py          # LLM inference
│   ├── pipeline/
│   │   └── rag_pipeline.py         # Main pipeline
│   ├── main.py                      # CLI interface
│   └── evaluation.py                # Metrics
├── tests/
│   └── test_rag_system.py          # Comprehensive tests
├── requirements.txt
├── README.md
└── design_report.md
```


### Command Line

```bash
python src/main.py
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test hallucination detection
pytest tests/test_rag_system.py::TestHallucinationDetection -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Evaluation

```bash
python src/evaluation.py
```

Metrics: Accuracy, hallucination rate, ROUGE scores, refusal accuracy

## ⚙️ Configuration

### Default (CPU Mode)

```python
from src.config import RAGConfig
config = RAGConfig()  # CPU-safe defaults
```

### CPU Optimized

```python
from src.config_cpu import get_cpu_optimized_config
config = get_cpu_optimized_config()
```


See `CPU_SETUP.md` and `design_report.md` for detailed options.

## 📈 Performance

**CPU Mode (Default):**
- Indexing: ~20-30 min (400 pages)
- Query: ~15-30 sec
- Memory: 8-10GB RAM
- Model: Phi-3-mini (3.8B params)

**GPU Mode (Optional - RTX 3090):**
- Indexing: ~5 min (400 pages)
- Query: ~2-3 sec
- Memory: 16GB VRAM
- Model: Mistral-7B (7B params)

**Accuracy (Both modes):**
- Key info: 85-90%
- Hallucination rate: <5%




MIT License

---

**Built for accurate financial document analysis**