from chromadb.utils import embedding_functions
from chromadb import Client

from openai import OpenAI
from typing import List, Dict
import os
from dotenv import load_dotenv

import re
from collections import Counter

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Load environment variables
load_dotenv()

# NLTK English stopwords + domain-specific words to filter from keywords
STOPWORDS = set(stopwords.words('english')) | {
    'create', 'creates', 'add', 'adds', 'pair', 'pairs', 'perfect',
    'beautiful', 'works', 'feel', 'feels', 'keep', 'mix', 'combine',
    'layer', 'balance', 'style', 'design', 'room', 'rooms', 'space', 'spaces'
}


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

        # Initialize OpenAI client with API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")
        self.openai_client = OpenAI(api_key=api_key)

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
                "url": article.get("url", "")
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
            # Create searchable text (title + content only, no tags)
            searchable = f"{article['title']} {article['content']}".lower()
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

    # Maximum distance for semantic matches (higher = less similar)
    # 0.5 = very similar, 0.7 = somewhat similar, 0.85+ = weak/irrelevant
    MAX_SEMANTIC_DISTANCE = 0.75

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Dense vector-based semantic search with distance filtering"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['metadatas', 'distances']  # Include distances for filtering
        )

        articles = []
        if results['metadatas'] and len(results['metadatas']) > 0:
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                # GUARDRAIL: Filter out weak semantic matches
                if distance > self.MAX_SEMANTIC_DISTANCE:
                    continue

                articles.append({
                    "title": metadata["title"],
                    "content": metadata["content"],
                    "url": metadata.get("url", ""),
                    "semantic_distance": distance  # Include for debugging
                })

        return articles

    def extract_keywords(self, article: Dict, query: str, max_keywords: int = 3) -> List[str]:
        """Extract top keywords from article based on relevance to query.

        Uses a combination of:
        - Term frequency in the article
        - Query term matching (boost words that appear in query)
        """
        # Combine title + content
        text = f"{article['title']} {article['content']}".lower()
        words = re.findall(r'\w+', text)

        # Get word frequencies
        word_freq = Counter(words)

        # Query terms for boosting
        query_terms = set(re.findall(r'\w+', query.lower()))

        # Filter and score words
        scored_words = []
        for word, freq in word_freq.items():
            # Skip stopwords and short words
            if word in STOPWORDS or len(word) < 4:
                continue

            # Base score is frequency
            score = freq

            # Boost if word appears in query
            if word in query_terms:
                score *= 3

            scored_words.append((word, score))

        # Sort by score and return top keywords
        scored_words.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in scored_words[:max_keywords]]

    # Minimum relevance score to include an article (prevents weak matches)
    MIN_RELEVANCE_SCORE = 2.0

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """Combine sparse and dense search results

        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for semantic search (1-alpha for keyword search)

        Returns:
            List of articles with relevance scores. Empty list if no relevant matches.
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

        # Extract keywords for each result, filter by relevance threshold
        final_results = []
        for item in sorted_results[:top_k]:
            # GUARDRAIL: Skip articles below minimum relevance score
            if item['score'] < self.MIN_RELEVANCE_SCORE:
                continue

            article = item['article']
            article['keywords'] = self.extract_keywords(article, query)
            article['relevance_score'] = item['score']  # Include score for transparency
            final_results.append(article)

        return final_results

    def generate_no_results_response(self, query: str) -> str:
        """Generate a helpful response when no relevant articles are found"""
        return (
            f"I don't have specific articles about \"{query}\" in my collection. "
            "My expertise covers interior design topics like dining rooms, bedrooms, "
            "living spaces, color palettes, and decor styles. "
            "Could you try rephrasing your question or ask about a different design topic?"
        )

    def generate_conversation_intro(self, query: str, articles: List[Dict]) -> str:
        """Generate a warm, conversational intro based on the query and results"""
        try:
            # Create a brief overview of found articles
            article_titles = [a['title'] for a in articles[:3]]

            prompt = f"""A client just asked about: "{query}"

You found these relevant design ideas: {', '.join(article_titles)}

Write a warm, 2-3 sentence introduction as an experienced interior design consultant welcoming their interest and briefly previewing what you'll share. Be enthusiastic but professional.

IMPORTANT: Only mention design elements, colors, or styles that appear in the article titles above. Do not invent trends, statistics, or facts not present in the titles."""

            response = self.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": "You are an experienced interior design consultant with 20 years of expertise. You speak warmly and professionally. CRITICAL: You must ONLY reference information from the provided article titles. Never invent design trends, statistics, celebrity endorsements, or facts. If unsure, keep it general."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7  # Lowered from 0.8 for more grounded responses
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"Let me share some wonderful ideas for {query}!"

    def generate_recommendation(self, article: Dict, query: str) -> str:
        """Generate a personalized recommendation for a single article"""
        try:
            prompt = f"""Client asked about: "{query}"

Article Title: {article['title']}
Article Content: {article['content']}

Write ONE sentence recommending this to the client. Speak directly to them ("You might love..." or "Consider..." or "This would be perfect if...").

CRITICAL RULES:
1. ONLY use information explicitly stated in the Article Content above
2. Do NOT invent prices, brands, statistics, or facts not in the content
3. Do NOT mention specific products, stores, or designers unless they appear in the content
4. Keep the recommendation grounded in what the article actually says"""

            response = self.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": "You are an interior design consultant. IMPORTANT: Base your recommendation ONLY on the provided article content. Never hallucinate information, prices, brand names, or specific products not mentioned. If the article content is brief, keep your recommendation brief and general. One sentence only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=80,
                temperature=0.5  # Lowered from 0.7 for more factual responses
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return article['content'][:100] + "..."

    def generate_summary(self, article: Dict) -> str:
        """Generate 2-3 line summary using OpenAI GPT (legacy method)"""
        try:
            prompt = f"""Summarize this interior design article in 2-3 lines for a user searching for design inspiration.

Title: {article['title']}
Content: {article['content']}

Summary:"""

            response = self.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {"role": "system", "content": "You are a helpful interior design assistant that creates concise, inspiring summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback if OpenAI fails
            print(f"OpenAI API error: {e}")
            return article['content'][:150] + "..."