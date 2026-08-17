"""Unit tests for cover letter generation."""

from agent import cover_letter


def test_cover_letter_prompt_contains_required_placeholders():
    prompt = cover_letter.COVER_LETTER_PROMPT

    for placeholder in (
        "{user_name}",
        "{job_title}",
        "{company_name}",
        "{job_description}",
        "{cv_context}",
    ):
        assert placeholder in prompt


def test_generate_cover_letter_passes_all_context(monkeypatch):
    captured = {}

    class FakeFinalChain:
        def invoke(self, payload):
            captured.update(payload)
            return "Dear Hiring Team,\n\nI am a suitable candidate.\n\nBest,\nPablo"

    class FakeLLM:
        def __or__(self, parser):
            return FakeFinalChain()

    class FakePrompt:
        def __or__(self, llm):
            return FakeLLM()

    monkeypatch.setattr(
        cover_letter,
        "build_chat_model",
        lambda **kwargs: object(),
    )

    monkeypatch.setattr(
        cover_letter,
        "cl_prompt",
        FakePrompt(),
    )

    monkeypatch.setattr(
        cover_letter,
        "StrOutputParser",
        lambda: object(),
    )

    result = cover_letter.generate_cover_letter_draft(
        job_description="Build AI systems.",
        cv_context="Python and PyTorch project.",
        company_name="Example Corp",
        job_title="AI Engineer",
        user_name="Pablo",
        provider="ollama",
    )

    assert captured["company_name"] == "Example Corp"
    assert captured["job_title"] == "AI Engineer"
    assert captured["user_name"] == "Pablo"
    assert captured["job_description"] == "Build AI systems."
    assert captured["cv_context"] == "Python and PyTorch project."

    assert result.startswith("Dear Hiring Team")