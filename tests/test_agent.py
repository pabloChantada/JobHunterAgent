"""Unit tests for job evaluation logic."""
from agent.agent import JobEvaluation, evaluate_job_offer


class FakeRetriever:
    def invoke(self, job_description):
        return [
            type("Doc", (), {"page_content": "Python, PyTorch and FastAPI experience."})()
        ]


class FakeVectorStore:
    def as_retriever(self, **kwargs):
        self.kwargs = kwargs
        return FakeRetriever()


class FakeChain:
    def __init__(self, result):
        self.result = result
        self.payload = None

    def invoke(self, payload):
        self.payload = payload
        return self.result


def test_job_evaluation_model_accepts_valid_result():
    result = JobEvaluation(
        score=85,
        verdict=True,
        reasons=["Python matches", "Junior profile matches", "Remote compatible"],
    )

    assert result.score == 85
    assert result.verdict is True
    assert len(result.reasons) == 3


def test_evaluate_job_offer_retrieves_cv_in_job_language(monkeypatch):
    vectorstore = FakeVectorStore()
    chain = FakeChain(
        JobEvaluation(
            score=80,
            verdict=True,
            reasons=["Python", "FastAPI", "Remote"],
        )
    )

    monkeypatch.setattr(
        "agent.agent.detect_language_from_text",
        lambda _: "english",
    )
    monkeypatch.setattr(
        "agent.agent.get_vectorstore",
        lambda: vectorstore,
    )

    result = evaluate_job_offer("Python AI Engineer role", chain=chain)

    assert result.score == 80
    assert result.verdict is True
    assert vectorstore.kwargs["search_kwargs"]["filter"]["language"] == "english"
    assert vectorstore.kwargs["k"] == 7
    assert chain.payload["job_description"] == "Python AI Engineer role"
    assert "Python" in chain.payload["cv_context"]
