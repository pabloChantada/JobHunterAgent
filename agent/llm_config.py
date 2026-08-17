"""Centralized LLM provider and model configuration."""
import os

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


SUPPORTED_PROVIDERS = {"ollama", "groq"}

DEFAULT_MODELS = {
    "ollama": {
        "evaluator": "llama3.1",
        "cover_letter": "llama3.1",
    },
    "groq": {
        "evaluator": "openai/gpt-oss-120b",
        "cover_letter": "openai/gpt-oss-120b",
    },
}


def get_llm_provider(provider: str = None) -> str:
    """Return the normalized LLM provider from args/env."""
    selected = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if selected not in SUPPORTED_PROVIDERS:
        raise ValueError("Unsupported provider. Use 'ollama' or 'groq'.")
    return selected


def get_llm_model(provider: str, task: str) -> str:
    """Resolve model name with task/provider/global env fallbacks."""
    provider = provider.lower()
    task_key = task.upper()

    task_specific_key = f"{provider}_{task_key}_MODEL".upper()
    provider_key = f"{provider}_MODEL".upper()

    return (
        os.getenv(task_specific_key)
        or os.getenv(provider_key)
        or os.getenv("LLM_MODEL")
        or DEFAULT_MODELS[provider][task]
    )


def build_chat_model(
    task: str,
    provider: str = None,
    temperature: float = 0.0,
    num_predict: int = None,
    timeout: int = None,
    ollama_base_url: str = None,
):
    """Create a configured chat model for a task/provider pair."""
    selected_provider = get_llm_provider(provider)
    model_name = get_llm_model(selected_provider, task)

    if selected_provider == "ollama":
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "base_url": ollama_base_url
            or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
        }
        if num_predict is not None:
            kwargs["num_predict"] = num_predict
        if timeout is not None:
            kwargs["timeout"] = timeout
        return ChatOllama(**kwargs)

    return ChatGroq(model=model_name, temperature=temperature)