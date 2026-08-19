"""LLM client implementations supporting open-source models (Qwen) and testing mocks."""

from abc import ABC, abstractmethod
from typing import List, Optional

from vector_rag.utils.errors import GenerationError
from vector_rag.utils.logging import get_logger

logger = get_logger("generation.llm")


class BaseLLM(ABC):
    """Abstract base class for LLM generation engines."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response text for the given user prompt and optional system prompt."""


class MockLLM(BaseLLM):
    """Mock LLM generating deterministic responses citing provided sources."""

    def __init__(self, default_response: Optional[str] = None) -> None:
        self.default_response = default_response

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.default_response:
            return self.default_response
        return (
            "Based on the provided documentation, filesystems use inodes to store metadata "
            "such as permissions, size, and data block pointers [1]. "
            "Filesystem superblocks manage structural geometries [2]."
        )


class HuggingFaceLLM(BaseLLM):
    """
    Open-Source LLM generation runner leveraging HuggingFace Transformers.
    
    Optimized for Qwen models (e.g., Qwen/Qwen2.5-0.5B-Instruct / Qwen 8B series).
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert technical assistant. Answer the user's question accurately and "
        "concisely using ONLY the provided context sources. Whenever you state a fact derived "
        "from a source, cite the source number using brackets, e.g. [1] or [2]. "
        "If the context does not contain the answer, state that the information is unavailable."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 800,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self.load_in_8bit = load_in_8bit
        self._pipe = None

    def _get_pipeline(self):
        """Lazy load the transformers pipeline and tokenizer."""
        if self._pipe is None:
            try:
                import os
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    pipeline,
                    logging as hf_logging,
                )

                # Suppress HuggingFace/tokenizer/tqdm noise on stderr
                hf_logging.set_verbosity_error()
                os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

                logger.info(
                    "Loading open-source LLM '%s' (temperature=%.2f, max_tokens=%d)...",
                    self.model_name,
                    self.temperature,
                    self.max_tokens,
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )

                model_kwargs = {"trust_remote_code": True}
                if self.device:
                    model_kwargs["device_map"] = self.device
                elif torch.cuda.is_available():
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["torch_dtype"] = torch.float16
                else:
                    model_kwargs["device_map"] = "cpu"

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )

                self._pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.max_tokens,
                    do_sample=self.temperature > 0.0,
                    temperature=self.temperature if self.temperature > 0.0 else None,
                )
                logger.info("Successfully loaded LLM model '%s'.", self.model_name)
            except GenerationError:
                raise
            except Exception as e:
                raise GenerationError(
                    f"Failed to load HuggingFace LLM model '{self.model_name}': {e}",
                    hint="Ensure you have adequate RAM/VRAM and correct model repository name.",
                ) from e
        return self._pipe

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pipe = self._get_pipeline()
        sys_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            # Format using tokenizer chat template if supported
            tokenizer = pipe.tokenizer
            if hasattr(tokenizer, "apply_chat_template"):
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                outputs = pipe(formatted_prompt, return_full_text=False)
                return outputs[0]["generated_text"].strip()
            else:
                raw_input = f"{sys_prompt}\n\nUser: {prompt}\n\nAnswer:"
                outputs = pipe(raw_input, return_full_text=False)
                return outputs[0]["generated_text"].strip()
        except Exception as e:
            raise GenerationError(f"LLM generation failed for prompt: {e}") from e


def create_llm(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 800,
    mock: bool = False,
) -> BaseLLM:
    """Factory to instantiate the appropriate LLM generator."""
    if mock:
        return MockLLM()
    return HuggingFaceLLM(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
