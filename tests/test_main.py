"""Unit tests for the main JobHunter pipeline."""
import json

import pandas as pd

import main


class Evaluation:
    def __init__(self, score, verdict, reasons):
        self.score = score
        self.verdict = verdict
        self.reasons = reasons


def test_get_already_processed_ids_returns_empty_when_tracker_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MASTER_TRACKER_PATH", tmp_path / "tracker.xlsx")

    assert main.get_already_processed_ids() == set()


def test_get_already_processed_ids_reads_job_ids(tmp_path, monkeypatch):
    path = tmp_path / "tracker.xlsx"
    pd.DataFrame({"Job ID": ["1", "2", "3"]}).to_excel(path, index=False)
    monkeypatch.setattr(main, "MASTER_TRACKER_PATH", path)

    assert main.get_already_processed_ids() == {1,2,3}


def test_get_already_processed_ids_handles_missing_column(tmp_path, monkeypatch):
    path = tmp_path / "tracker.xlsx"
    pd.DataFrame({"Title": ["AI Engineer"]}).to_excel(path, index=False)
    monkeypatch.setattr(main, "MASTER_TRACKER_PATH", path)

    assert main.get_already_processed_ids() == set()


def test_process_single_job_rejected_does_not_generate_cover_letter(monkeypatch):
    job = {
        "id": "1",
        "title": "AI Engineer",
        "company": "Example",
        "description": "Python job",
        "url": "https://example.com/1",
    }

    monkeypatch.setattr(main, "evaluate_job_offer", lambda _: Evaluation(
        30, False, ["Too senior", "Wrong location", "Missing experience"]
    ))

    def fail_cover_letter(_):
        raise AssertionError("Cover letter should not be generated")

    monkeypatch.setattr(main, "generate_and_save_cover_letter", fail_cover_letter)

    result = main.process_single_job(job)

    assert result["Job ID"] == "1"
    assert result["Score"] == 30
    assert result["Verdict"] == "❌ REJECTED"
    assert result["Cover Letter Path"] == "N/A"
    assert result["Reasons"] == "- Too senior\n- Wrong location\n- Missing experience"


def test_process_single_job_accepted_generates_cover_letter(monkeypatch):
    job = {
        "id": "2",
        "title": "ML Engineer",
        "company": "Example",
        "description": "Python ML job",
        "url": "https://example.com/2",
    }

    monkeypatch.setattr(main, "evaluate_job_offer", lambda _: Evaluation(
        85, True, ["Python matches", "ML matches", "Junior role"]
    ))
    monkeypatch.setattr(
        main,
        "generate_and_save_cover_letter",
        lambda _: "data/cover_letters/2_Example.txt",
    )

    result = main.process_single_job(job)

    assert result["Score"] == 85
    assert result["Verdict"] == "✅ APPLY"
    assert result["Cover Letter Path"] == "data/cover_letters/2_Example.txt"


def test_generate_and_save_cover_letter_sanitizes_company_name(tmp_path, monkeypatch):
    job = {
        "id": "42",
        "title": "AI Engineer",
        "company": "ACME / AI: Labs!",
        "description": "English AI job",
    }

    class FakeRetriever:
        def invoke(self, _):
            return [
                type("Doc", (), {"page_content": "Python experience"})(),
                type("Doc", (), {"page_content": "FastAPI project"})(),
            ]

    class FakeStore:
        def as_retriever(self, **kwargs):
            return FakeRetriever()

    monkeypatch.setattr(main, "detect_language_from_text", lambda _: "english")
    monkeypatch.setattr(main, "get_vectorstore", lambda: FakeStore())
    monkeypatch.setattr(
        main,
        "generate_cover_letter_draft",
        lambda **kwargs: "Cover letter content",
    )
    monkeypatch.setattr(main, "COVER_LETTERS_DIR", tmp_path)

    result = main.generate_and_save_cover_letter(job)

    expected = tmp_path / "42_ACME__AI_Labs.txt"
    assert expected.exists()
    assert expected.read_text(encoding="utf-8") == "Cover letter content"
    assert result == "data/cover_letters/42_ACME__AI_Labs.txt"


def test_save_results_creates_master_and_daily_reports(tmp_path, monkeypatch):
    master = tmp_path / "master.xlsx"
    daily = tmp_path / "daily.xlsx"

    monkeypatch.setattr(main, "MASTER_TRACKER_PATH", master)
    monkeypatch.setattr(main, "DAILY_REPORT_PATH", daily)

    results = [
        {
            "Date": "2026-08-17 15:00",
            "Job ID": "1",
            "Title": "AI Engineer",
            "Company": "Example",
            "Score": 80,
            "Verdict": "✅ APPLY",
            "Reasons": "- Python",
            "Url": "https://example.com/1",
            "Provider": "ollama",
            "Cover Letter Path": "N/A",
        }
    ]

    main.save_results(results)

    assert master.exists()
    assert daily.exists()
    assert pd.read_excel(master).iloc[0]["Job ID"] == 1
    assert pd.read_excel(daily).iloc[0]["Score"] == 80


def test_process_jobs_skips_already_processed_jobs(tmp_path, monkeypatch):
    jobs_path = tmp_path / "scraped_offers.json"
    jobs_path.write_text(
        json.dumps(
            [
                {
                    "id": "already-seen",
                    "title": "AI Engineer",
                    "company": "Example",
                    "description": "Python",
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(main, "JOBS_PATH", jobs_path)
    monkeypatch.setattr(main, "get_already_processed_ids", lambda: {"already-seen"})

    def fail_process(_):
        raise AssertionError("Already processed jobs should be skipped")

    monkeypatch.setattr(main, "process_single_job", fail_process)

    main.process_jobs()


def test_clean_files_removes_daily_report_and_cover_letters(tmp_path, monkeypatch):
    letters = tmp_path / "cover_letters"
    letters.mkdir()
    (letters / "letter.txt").write_text("test", encoding="utf-8")

    daily = tmp_path / "daily.xlsx"
    daily.write_text("test", encoding="utf-8")

    monkeypatch.setattr(main, "COVER_LETTERS_DIR", letters)
    monkeypatch.setattr(main, "DAILY_REPORT_PATH", daily)

    main.clean_files()

    assert not (letters / "letter.txt").exists()
    assert not daily.exists()
