from chromadb.utils import embedding_functions
from chromadb import Client

import json
import ollama
from typing import List, Dict

import re
from collections import Counter

class HybridRAG:
    def __init__(self, SAMPLE_ARTICLES):
        # Initialize ChromaDB
        self.client = Client()
        
        # Sentence transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create collection
        try:
            self.client.delete_collection("interior_design_articles")
        except:
            pass
        
        self.collection = self.client.create_collection(
            name="interior_design_articles",
            embedding_function=self.embedding_function
        )
        
        # Store articles for keyword search
        self.articles = SAMPLE_ARTICLES
        
    def populate_database(self):
        """Add all articles to ChromaDB"""
        documents = []
        metadatas = []
        ids = []
        
        for article in self.articles:
            # Combine title and content for better semantic search
            doc_text = f"{article['title']}. {article['content']}"
            documents.append(doc_text)
            metadatas.append({
                "title": article["title"],
                "content": article["content"],
                "tags": json.dumps(article["tags"])
            })
            ids.append(article["id"])
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Sparse keyword-based search using TF-IDF-like scoring"""
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        scored_articles = []
        for article in self.articles:
            # Create searchable text
            searchable = f"{article['title']} {article['content']} {' '.join(article['tags'])}".lower()
            searchable_terms = re.findall(r'\w+', searchable)
            
            # Calculate term frequency
            term_freq = Counter(searchable_terms)
            
            # Score based on query term matches
            score = sum(term_freq[term] for term in query_terms if term in term_freq)
            
            if score > 0:
                scored_articles.append({
                    "article": article,
                    "score": score
                })
        
        # Sort by score and return top_k
        scored_articles.sort(key=lambda x: x["score"], reverse=True)
        return [item["article"] for item in scored_articles[:top_k]]
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Dense vector-based semantic search"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        articles = []
        if results['metadatas'] and len(results['metadatas']) > 0:
            for metadata in results['metadatas'][0]:
                articles.append({
                    "title": metadata["title"],
                    "content": metadata["content"],
                    "tags": json.loads(metadata["tags"])
                })
        
        return articles
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """Combine sparse and dense search results
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for semantic search (1-alpha for keyword search)
        """
        # Get results from both methods
        keyword_results = self.keyword_search(query, top_k * 2)
        semantic_results = self.semantic_search(query, top_k * 2)
        
        # Combine and deduplicate
        combined = {}
        
        # Score keyword results
        for idx, article in enumerate(keyword_results):
            article_id = article.get('id', article['title'])
            score = (len(keyword_results) - idx) * (1 - alpha)
            combined[article_id] = {
                'article': article,
                'score': score
            }
        
        # Add/update with semantic results
        for idx, article in enumerate(semantic_results):
            article_id = article.get('id', article['title'])
            score = (len(semantic_results) - idx) * alpha
            
            if article_id in combined:
                combined[article_id]['score'] += score
            else:
                combined[article_id] = {
                    'article': article,
                    'score': score
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['article'] for item in sorted_results[:top_k]]
    
    def generate_summary(self, article: Dict) -> str:
        """Generate 2-3 line summary using Ollama"""
        try:
            prompt = f"""Summarize this interior design article in 2-3 lines for a user searching for design inspiration.

Title: {article['title']}
Content: {article['content']}

Summary:"""
            
            response = ollama.generate(
                model='qwen2.5:0.5b', 
                prompt=prompt
            )
            
            return response['response'].strip()
        except Exception as e:
            # Fallback if Ollama fails
            return article['content'][:150] + "..."