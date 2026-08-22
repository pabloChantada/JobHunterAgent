"""Unit tests for CV extraction, sectioning, and language detection."""
from pathlib import Path

import pytest

from agent import vectorstore


def test_detect_language_from_filename():
    assert vectorstore.detect_language("Pablo_ES.pdf") == "spanish"
    assert vectorstore.detect_language("Pablo.pdf") == "english"


def test_chunk_by_sections_extracts_sections():
    text = """Pablo Chantada
CONTACT INFO

PROFILE
AI student and Python developer.

TECHNICAL SKILLS
Python, PyTorch, FastAPI.

EXPERIENCE
Built machine learning applications.
"""

    sections = vectorstore.chunk_by_sections(text)

    assert sections["CONTACTO_HEADER"] == "Pablo Chantada\nCONTACT INFO"
    assert sections["PROFILE"] == "AI student and Python developer."
    assert sections["TECHNICAL SKILLS"] == "Python, PyTorch, FastAPI."
    assert sections["EXPERIENCE"] == "Built machine learning applications."


def test_chunk_by_sections_handles_empty_text():
    assert vectorstore.chunk_by_sections("") == {}


def test_get_splitter_configuration():
    splitter = vectorstore.get_splitter()

    assert splitter._chunk_size == 1500
    assert splitter._chunk_overlap == 100


def test_extract_text_normalizes_unicode(tmp_path, monkeypatch):
    class FakePage:
        def get_text(self):
            return "Cafe\u0301"

    class FakeDocument:
        def __iter__(self):
            return iter([FakePage()])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        vectorstore.pymupdf,
        "open",
        lambda path: FakeDocument(),
    )

    result = vectorstore.extract_text(tmp_path / "cv.pdf")

    assert result == "Café"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python and machine learning engineer with FastAPI experience.", "english"),
        ("Ingeniero de inteligencia artificial con experiencia en Python.", "spanish"),
    ],
)
def test_detect_language_from_text(text, expected):
    assert vectorstore.detect_language_from_text(text) == expected


def test_detect_language_from_text_returns_unknown_on_detection_error(monkeypatch):
    def raise_error(_):
        raise RuntimeError("language detection failed")

    monkeypatch.setattr(vectorstore, "detect", raise_error)

    assert vectorstore.detect_language_from_text("short text") == None


def test_add_cv_to_db_builds_chunks_and_metadata(monkeypatch, tmp_path):
    cv_path = tmp_path / "Pablo_ES.pdf"

    monkeypatch.setattr(
        vectorstore,
        "extract_text",
        lambda _: "PROFILE\nAI student\n\nEXPERIENCE\nPython developer",
    )

    captured = {}

    class FakeStore:
        def add_texts(self, texts, metadatas, ids):
            captured["texts"] = texts
            captured["metadatas"] = metadatas
            captured["ids"] = ids

    monkeypatch.setattr(vectorstore, "get_vectorstore", lambda: FakeStore())

    vectorstore.add_cv_to_db(cv_path)

    assert len(captured["texts"]) == 2
    assert all(item["language"] == "spanish" for item in captured["metadatas"])
    assert captured["metadatas"][0]["section"] == "PROFILE"
    assert captured["metadatas"][1]["section"] == "EXPERIENCE"
    assert captured["ids"][0].startswith("Pablo_ES_spanish_PROFILE_")
