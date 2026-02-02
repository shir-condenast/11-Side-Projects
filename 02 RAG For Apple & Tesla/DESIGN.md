# RAG System Design Document

## 1. System Architecture

## 1.1 Overview

This RAG System implements a scable architecturefor answering questions about financial documents using retrieval augmentaed generation(RAG)

## 1.2 Component Design

```
User Query → Retriver → Re-Ranker → LLM → Response
                ↓
            Vectore Store
                ↑
            Document Processor
```

