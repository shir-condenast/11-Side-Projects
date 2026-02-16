"""
LLM service with GPU acceleration and 4-bit quantization.
Handles text generation with custom prompting.
"""
import torch
from typing import List, Optional
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from loguru import logger

from src.config import LLMConfig
from src.models.schemas import RetrievedContext, PromptTemplate


class LLMService:
    """Service for LLM inference with GPU support."""
    
    # Default prompt template
    DEFAULT_PROMPT = PromptTemplate(
        system_prompt="""You are a financial document analyst. Your task is to answer questions based ONLY on the provided context from Apple and Tesla 10-K filings.

CRITICAL RULES:
1. Use ONLY information from the provided context
2. If the answer is not in the context, respond: "Not specified in the document."
3. For questions outside the scope of the documents, respond: "This question cannot be answered based on the provided documents."
4. Always cite sources as: ["Company 10-K", "Item X", "p. Y"]
5. Be precise with numbers, dates, and facts
6. Do not make assumptions or use external knowledge""",
        
        user_template="""Context from documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context above
- Cite sources for each fact
- If information is not in the context, say "Not specified in the document."
- For out-of-scope questions, say "This question cannot be answered based on the provided documents."

Answer:"""
    )
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM service.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info(f"Loading LLM: {config.model_name}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load model with quantization if configured."""
        # Configure quantization
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info("Loading model...")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if self.config.device == "cuda" else None,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16


        # config = AutoConfig.from_pretrained(self.config.model_name)
        # Model Loaded with it, % config without it
        config = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )

        # Sanity Check
        logger.info(f"Config class: {type(config)}")


        # # Ensure Rope Scaling is valid
        # if hasattr(config, "rope_scaling"):
        #     if config.rope_scaling is None:
        #         config.rope_scaling = {"type": "none"}
        #     elif isinstance(config.rope_scaling, dict) and "type" not in config.rope_scaling:
        #         config.rope_scaling["type"] = "none"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=config,
            **model_kwargs
        )

        
        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True if self.config.temperature > 0 else False,
        )
        
        logger.info("LLM loaded successfully")
        
        if torch.cuda.is_available():
            logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Override max tokens
            
        Returns:
            Generated text
        """
        max_tokens = max_new_tokens or self.config.max_new_tokens
        
        try:
            # Generate
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated = outputs[0]['generated_text']
            
            # Remove prompt from output
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            
            return generated
        
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def answer_question(
        self,
        question: str,
        contexts: List[RetrievedContext],
        prompt_template: Optional[PromptTemplate] = None
    ) -> str:
        """
        Answer question using retrieved contexts.
        
        Args:
            question: User question
            contexts: Retrieved contexts
            prompt_template: Optional custom prompt template
            
        Returns:
            Generated answer
        """
        template = prompt_template or self.DEFAULT_PROMPT
        
        # Format context
        context_str = self._format_contexts(contexts)
        
        # Create prompt
        user_prompt = template.format(
            context=context_str,
            question=question
        )
        
        # Combine system and user prompts
        full_prompt = self._format_full_prompt(
            template.system_prompt,
            user_prompt
        )
        
        # Generate answer
        answer = self.generate(full_prompt)
        
        return answer
    
    def _format_contexts(self, contexts: List[RetrievedContext]) -> str:
        """Format retrieved contexts for prompt."""
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
        """Format complete prompt based on model type."""
        model_name_lower = self.config.model_name.lower()
        
        # Mistral format
        if 'mistral' in model_name_lower:
            return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        
        # Llama format
        elif 'llama' in model_name_lower:
            return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        
        # Phi format
        elif 'phi' in model_name_lower:
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        # Generic format
        else:
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    
    def extract_sources(self, answer: str) -> List[str]:
        """
        Extract source citations from answer.
        
        Args:
            answer: Generated answer
            
        Returns:
            List of source citations
        """
        import re
        
        # Pattern to match citations like ["Apple 10-K", "Item 8", "p. 28"]
        pattern = r'\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]'
        matches = re.findall(pattern, answer)
        
        sources = []
        for match in matches:
            sources.append(list(match))
        
        return sources
    
    def is_refusal(self, answer: str) -> bool:
        """
        Check if answer is a refusal response.
        
        Args:
            answer: Generated answer
            
        Returns:
            True if answer is a refusal
        """
        refusal_phrases = [
            "not specified in the document",
            "cannot be answered based on the provided documents",
            "not found in the provided",
            "information is not available",
            "not mentioned in the document"
        ]
        
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in refusal_phrases)
    
    def clear_gpu_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")