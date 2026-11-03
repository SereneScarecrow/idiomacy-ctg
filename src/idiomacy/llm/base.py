"""
Common interface of all language-model backends.

A backend (Ollama, OpenRouter, a local Hugging Face model, ...) is a class with one
method, generate(Request) -> Response. Everything above this level (runner, validation,
logging) works only with these two types.
"""

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    """The backend could not answer (server is down, model is missing, ...)."""


@dataclass(frozen=True)
class Request:
    model: str
    prompt: str
    system: str | None = None
    images: tuple[str, ...] = ()        # paths to image files (for vision-language models)
    schema: dict | None = None          # JSON schema of the expected answer (structured output)
    temperature: float = 0.0
    seed: int | None = None
    max_tokens: int | None = None       # limit of generated tokens
    reasoning: bool = False             # let the model think before answering (models that support it)
    extra: dict = field(default_factory=dict)  # backend-specific options, e.g. {"num_ctx": 8192}


@dataclass
class Response:
    text: str
    finish_reason: str                  # "stop" (finished by itself) or "length" (cut off by the limit)
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    model: str                          # model name as reported by the backend
    raw: dict                           # the whole backend response, JSON-serializable, for the log
    reasoning: str | None = None        # the model's thinking, kept apart from `text` (if reasoning was on)


class LLMClient(ABC):
    provider: str = ""

    @abstractmethod
    def generate(self, request: Request) -> Response:
        """Raises LLMError if the backend could not answer."""

    def model_info(self, model: str) -> dict:
        """Identity of the model for the run manifest (digest, revision, quantization, ...).
        Should raise LLMError if the model is not available, so a run fails before it starts."""
        return {}

    def describe(self) -> dict:
        """Identity of the backend itself for the run manifest (server version, ...)."""
        return {"provider": self.provider}


# provider name -> (module, class); modules are imported only when the provider is used,
# so heavy dependencies (torch, transformers) are not needed for the other backends
_BACKENDS = {
    "ollama": ("idiomacy.llm.ollama_client", "OllamaClient"),
    # "openrouter": ("idiomacy.llm.openrouter_client", "OpenRouterClient"),
    # "hf": ("idiomacy.llm.hf_client", "HuggingFaceClient"),
}


def create_client(provider: str, **kwargs) -> LLMClient:
    try:
        module_name, class_name = _BACKENDS[provider]
    except KeyError:
        raise ValueError(f"Unknown provider {provider!r}; available: {sorted(_BACKENDS)}") from None
    return getattr(importlib.import_module(module_name), class_name)(**kwargs)