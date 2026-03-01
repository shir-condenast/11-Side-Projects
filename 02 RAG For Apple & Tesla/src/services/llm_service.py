"""
CPU-Optimized LLM Service for Phi-3-mini.
4-bit quantization.
Deterministic generation for financial RAG.
"""

import torch
import re
import time
from typing import List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig
)
from loguru import logger

from src.config import LLMConfig
from src.models.schemas import RetrievedContext, PromptTemplate


class LLMService:
    """CPU-only LLM inference service."""

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

        if config.device != "cpu":
            raise ValueError("This service is CPU-only. Set device='cpu'.")

        torch.set_num_threads(4)  # adjust based on CPU cores

        logger.info(f"Loading model: {config.model_name}")
        self._load_model()

    # ----------------------------------------------------
    # MODEL LOADING
    # ----------------------------------------------------
    def _load_model(self):

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading config...")

        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # 🔥 FORCE SAFE ATTENTION FOR CPU
        model_config.attn_implementation = "eager"

        logger.info("Configuring 4-bit quantization...")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        logger.info("Loading model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=model_config,
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map=None,
            low_cpu_mem_usage=True
        )

        self.model.eval()

        logger.info("Model loaded successfully on CPU.")

    # ----------------------------------------------------
    # GENERATION
    # ----------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None
    ) -> str:

        max_tokens = max_new_tokens or self.config.max_new_tokens

        start_total = time.time()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False
        )

        input_tokens = inputs["input_ids"].shape[1]
        logger.info(f"Input tokens: {input_tokens}")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # deterministic for financial QA
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()

        total_time = time.time() - start_total
        logger.info(f"Generation time: {total_time:.2f} sec")

        return output_text

    # ----------------------------------------------------
    # QA PIPELINE
    # ----------------------------------------------------
    def answer_question(
        self,
        question: str,
        contexts: List[RetrievedContext],
        prompt_template: Optional[PromptTemplate] = None
    ) -> str:

        template = prompt_template or self.DEFAULT_PROMPT

        MAX_INPUT_TOKENS = 1000
        RESERVED_FOR_OUTPUT = 200
        context_budget = MAX_INPUT_TOKENS - RESERVED_FOR_OUTPUT

        selected_contexts = []
        used_tokens = 0

        for ctx in contexts:
            chunk_text = ctx.chunk.text

            token_count = self.tokenizer(
                chunk_text,
                return_tensors="pt"
            )["input_ids"].shape[1]

            if used_tokens + token_count > context_budget:
                break

            selected_contexts.append(ctx)
            used_tokens += token_count

        logger.info(f"Context tokens used: {used_tokens}")

        context_str = self._format_contexts(selected_contexts)

        user_prompt = template.user_template.format(
            context=context_str,
            question=question
        )

        full_prompt = self._format_full_prompt(
            template.system_prompt,
            user_prompt
        )

        total_tokens = self.tokenizer(
            full_prompt,
            return_tensors="pt"
        )["input_ids"].shape[1]

        logger.info(f"Final input tokens: {total_tokens}")

        return self.generate(full_prompt)

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

    def _format_full_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

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