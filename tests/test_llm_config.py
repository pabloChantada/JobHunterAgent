"""Unit tests for LLM provider and model configuration."""
import pytest

from agent import llm_config


def test_get_llm_provider_uses_explicit_provider():
    assert llm_config.get_llm_provider("GROQ") == "groq"


def test_get_llm_provider_uses_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert llm_config.get_llm_provider() == "groq"


def test_get_llm_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        llm_config.get_llm_provider("openai")


def test_get_llm_model_prefers_task_specific_environment_variable(monkeypatch):
    monkeypatch.setenv("GROQ_EVALUATOR_MODEL", "custom-evaluator")
    monkeypatch.setenv("GROQ_MODEL", "provider-model")
    monkeypatch.setenv("LLM_MODEL", "global-model")

    assert llm_config.get_llm_model("groq", "evaluator") == "custom-evaluator"


def test_get_llm_model_falls_back_to_provider_model(monkeypatch):
    monkeypatch.delenv("GROQ_EVALUATOR_MODEL", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "provider-model")

    assert llm_config.get_llm_model("groq", "evaluator") == "provider-model"


def test_get_llm_model_falls_back_to_global_model(monkeypatch):
    monkeypatch.delenv("GROQ_EVALUATOR_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "global-model")

    assert llm_config.get_llm_model("groq", "evaluator") == "global-model"


def test_get_llm_model_uses_default_model(monkeypatch):
    for key in ("GROQ_EVALUATOR_MODEL", "GROQ_MODEL", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)

    assert llm_config.get_llm_model("groq", "evaluator") == "llama-3.3-70b-versatile"


def test_build_chat_model_uses_ollama_configuration(monkeypatch):
    class FakeOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(llm_config, "ChatOllama", FakeOllama)
    monkeypatch.delenv("OLLAMA_EVALUATOR_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    model = llm_config.build_chat_model(
        task="evaluator",
        provider="ollama",
        temperature=0.2,
        num_predict=100,
        timeout=20,
        ollama_base_url="http://test:11434",
    )

    assert model.kwargs["model"] == "llama3.1"
    assert model.kwargs["temperature"] == 0.2
    assert model.kwargs["num_predict"] == 100
    assert model.kwargs["timeout"] == 20
    assert model.kwargs["base_url"] == "http://test:11434"


def test_build_chat_model_uses_groq_configuration(monkeypatch):
    class FakeGroq:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(llm_config, "ChatGroq", FakeGroq)

    model = llm_config.build_chat_model(
        task="cover_letter",
        provider="groq",
        temperature=0.4,
    )

    assert model.kwargs == {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.4,
    }
