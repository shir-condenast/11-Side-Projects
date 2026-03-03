"""
OpenAI-based LLM Service for Financial RAG.
Uses GPT models via OpenAI API for question answering.
CPU-compatible (no local model loading required).
"""

import os
import re
import time
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

from src.config import LLMConfig
from src.models.schemas import RetrievedContext, PromptTemplate

# Load environment variables
load_dotenv()


class LLMService:
    """OpenAI-based LLM service for financial QA."""

    DEFAULT_PROMPT = PromptTemplate(
        system_prompt="""You are a financial document analyst. Your task is to answer questions based ONLY on the provided context from Apple and Tesla 10-K filings.

CRITICAL RULES:
1. Use ONLY information from the provided context
2. If the answer is not in the context, respond: "Not specified in the document."
3. For out-of-scope questions, respond: "This question cannot be answered based on the provided documents."
4. Always cite sources as: ["Company 10-K", "Item X", "p. Y"]
5. Be precise with numbers, dates, and facts
6. Do not use external knowledge""",

        user_template="""Context from documents:
{context}

Question: {question}

Answer:"""
    )

    def __init__(self, config: LLMConfig):
        self.config = config

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file")

        self.client = OpenAI(api_key=api_key)
        self.model_name = config.model_name  # Use model from config (default: gpt-4o-mini)

        logger.info(f"Initialized OpenAI LLM service with model: {self.model_name}")

    # ----------------------------------------------------
    # GENERATION
    # ----------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """Generate response using OpenAI API."""
        max_tokens = max_new_tokens or self.config.max_new_tokens

        start_total = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1  # Low temperature for factual responses
            )

            output_text = response.choices[0].message.content.strip()

            total_time = time.time() - start_total
            logger.info(f"Generation time: {total_time:.2f} sec")

            return output_text

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Error generating response. Please try again."

    # ----------------------------------------------------
    # QA PIPELINE
    # ----------------------------------------------------
    def answer_question(
        self,
        question: str,
        contexts: List[RetrievedContext],
        prompt_template: Optional[PromptTemplate] = None
    ) -> str:
        """Answer question using OpenAI with provided contexts."""
        template = prompt_template or self.DEFAULT_PROMPT

        # Select contexts (limit to avoid token limits)
        selected_contexts = contexts[:5]  # Take top 5 contexts

        logger.info(f"Using {len(selected_contexts)} contexts for answer generation")

        context_str = self._format_contexts(selected_contexts)

        user_prompt = template.user_template.format(
            context=context_str,
            question=question
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": template.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.max_new_tokens,
                temperature=0.1
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Error generating response. Please try again."

    # ----------------------------------------------------
    # HELPERS
    # ----------------------------------------------------
    def _format_contexts(self, contexts: List[RetrievedContext]) -> str:
        if not contexts:
            return "No relevant context found."

        formatted = []
        for ctx in contexts:
            source_info = (
                f"[{ctx.chunk.metadata.get('company')} 10-K, "
                f"{ctx.chunk.metadata.get('section', 'Unknown')}, "
                f"p. {ctx.chunk.metadata.get('page_number', 'Unknown')}]"
            )
            formatted.append(f"{source_info}\n{ctx.chunk.text}\n")

        return "\n---\n".join(formatted)

    def extract_sources(self, answer: str) -> List[List[str]]:
        pattern = r'\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]'
        matches = re.findall(pattern, answer)
        return [list(match) for match in matches]

    def is_refusal(self, answer: str) -> bool:
        refusal_phrases = [
            "not specified in the document",
            "cannot be answered based on the provided documents",
            "not mentioned in the document"
        ]

        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in refusal_phrases)

    def clear_gpu_cache(self):
        """No-op for OpenAI service (no local GPU usage)."""
        pass