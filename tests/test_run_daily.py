"""
Unit tests for run_daily.py
"""
from unittest.mock import MagicMock, patch

import pytest

import run_daily


# ---------------------------------------------------------------------------
# run_step
# ---------------------------------------------------------------------------

def test_run_step_returns_stdout_on_success():
    fake_result = MagicMock(returncode=0, stdout="ok output", stderr="")
    with patch("run_daily.subprocess.run", return_value=fake_result) as mock_run:
        output = run_daily.run_step(["python", "fake.py"])

    mock_run.assert_called_once_with(
        ["python", "fake.py"], capture_output=True, text=True
    )
    assert output == "ok output"


def test_run_step_raises_with_stderr_on_failure():
    fake_result = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch("run_daily.subprocess.run", return_value=fake_result):
        with pytest.raises(RuntimeError) as exc_info:
            run_daily.run_step(["python", "fake.py"])

    assert "boom" in str(exc_info.value)
    assert "fake.py" in str(exc_info.value)


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------

@pytest.fixture
def smtp_env(monkeypatch):
    monkeypatch.setenv("SMTP_USER", "bot@example.com")
    monkeypatch.setenv("SMTP_PASS", "app-password")
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    monkeypatch.setenv("SMTP_PORT", "587")


@pytest.fixture
def mock_smtp():
    """Patches run_daily.smtplib.SMTP and returns the mocked instance
    used inside the `with smtplib.SMTP(...) as s:` block."""
    instance = MagicMock()
    context_manager = MagicMock()
    context_manager.__enter__.return_value = instance
    with patch("run_daily.smtplib.SMTP", return_value=context_manager) as smtp_cls:
        yield smtp_cls, instance


def test_send_email_without_attachment(smtp_env, mock_smtp):
    smtp_cls, instance = mock_smtp

    run_daily.send_email("Subject", "<b>body</b>")

    smtp_cls.assert_called_once_with("smtp.gmail.com", 587)
    instance.starttls.assert_called_once()
    instance.login.assert_called_once_with("bot@example.com", "app-password")
    instance.send_message.assert_called_once()

    sent_msg = instance.send_message.call_args[0][0]
    assert sent_msg["Subject"] == "Subject"
    assert sent_msg["From"] == "bot@example.com"
    assert sent_msg["To"] == "bot@example.com"

    attachments = [p for p in sent_msg.walk() if p.get_content_disposition() == "attachment"]
    assert attachments == []


def test_send_email_with_existing_attachment(smtp_env, mock_smtp, tmp_path):
    _, instance = mock_smtp
    attachment = tmp_path / "daily_matches.xlsx"
    attachment.write_bytes(b"fake-excel-bytes")

    run_daily.send_email("Subject", "<b>body</b>", attachment)

    sent_msg = instance.send_message.call_args[0][0]
    attachments = [p for p in sent_msg.walk() if p.get_content_disposition() == "attachment"]

    assert len(attachments) == 1
    assert attachments[0].get_filename() == "daily_matches.xlsx"
    assert attachments[0].get_payload(decode=True) == b"fake-excel-bytes"


def test_send_email_skips_missing_attachment(smtp_env, mock_smtp, tmp_path):
    _, instance = mock_smtp
    missing = tmp_path / "does_not_exist.xlsx"

    run_daily.send_email("Subject", "<b>body</b>", missing)

    sent_msg = instance.send_message.call_args[0][0]
    attachments = [p for p in sent_msg.walk() if p.get_content_disposition() == "attachment"]
    assert attachments == []


def test_send_email_raises_if_smtp_env_vars_missing(monkeypatch):
    monkeypatch.delenv("SMTP_USER", raising=False)
    with pytest.raises(KeyError):
        run_daily.send_email("Subject", "<b>body</b>")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main_happy_path_runs_steps_in_order_and_sends_success_email(smtp_env):
    with patch("run_daily.run_step", side_effect=["scraper ok", "analyzer ok"]) as mock_run_step, \
         patch("run_daily.send_email") as mock_send_email:
        run_daily.main()

    assert mock_run_step.call_count == 2
    mock_run_step.assert_any_call(["python", "scraper/scraper.py"])
    mock_run_step.assert_any_call(["python", "main.py", "--clean"])
    # order matters: scraper must run before analyzer
    first_call_args = mock_run_step.call_args_list[0].args[0]
    assert first_call_args == ["python", "scraper/scraper.py"]

    mock_send_email.assert_called_once()
    subject, body, attachment = mock_send_email.call_args[0]
    assert subject == "Job Scraper - Daily Run"
    assert "analyzer ok" in body
    assert attachment == run_daily.EXCEL_PATH


def test_main_error_path_sends_error_email_and_exits_nonzero(smtp_env):
    with patch("run_daily.run_step", side_effect=RuntimeError("scraper exploded")), \
         patch("run_daily.send_email") as mock_send_email, \
         pytest.raises(SystemExit) as exc_info:
        run_daily.main()

    assert exc_info.value.code == 1
    mock_send_email.assert_called_once()
    call_args = mock_send_email.call_args[0]
    assert call_args[0] == "Job Scraper - ERROR"
    assert "scraper exploded" in call_args[1]
    # error path must not try to attach a (possibly stale/missing) excel file
    assert len(call_args) == 2


def test_main_stops_before_analyzer_if_scraper_fails(smtp_env):
    """If the scraper step fails, the analyzer step must never run —
    otherwise we'd risk emailing stale matches from a previous excel."""
    def fake_run_step(cmd):
        if "scraper" in cmd[1]:
            raise RuntimeError("scraper exploded")
        raise AssertionError("analyzer should not run if scraper failed")

    with patch("run_daily.run_step", side_effect=fake_run_step), \
         patch("run_daily.send_email"), \
         pytest.raises(SystemExit):
        run_daily.main()