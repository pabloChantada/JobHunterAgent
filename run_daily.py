"""
Script to run the daily job hunter agent process, including scraping, analyzing, and sending email notifications. Replaces the n8n workflow since we don't have a server to run it on. This script is intended to be run as a GitHub Action, but can also be run locally for testing purposes.
"""

import subprocess
import smtplib
import os
import sys
import traceback
from email.message import EmailMessage
from pathlib import Path

DATA_DIR = Path("data")  # private data repo, not checked into git
EXCEL_PATH = Path("daily_matches.xlsx") 
MASTER_EXCEL_PATH = DATA_DIR / "master_matches.xlsx"

def run_step(cmd: list[str]):
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)    # New logging for debugging      
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Failed in {cmd}:\n{result.stderr}")
    return result.stdout

def send_email(subject: str, body_html: str, attachments: list[Path] | None = None):
    msg = EmailMessage()
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = os.environ["SMTP_USER"]
    msg["Subject"] = subject
    msg.set_content("View HTML version")
    msg.add_alternative(body_html, subtype="html")

    if attachments:
        for attachment in attachments:
            if attachment.exists():
                msg.add_attachment(
                    attachment.read_bytes(),
                    maintype="application",
                    subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    filename=attachment.name,
                )

    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ["SMTP_PORT"])) as s:
        s.starttls()
        s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        s.send_message(msg)

def main():
    try:
        scraper_out = run_step(["python", "scraper/scraper.py"])
        analyzer_out = run_step(["python", "main.py", "--clean"])
        send_email(
            "Job Scraper - Daily Run",
            f"Daily run completed.<br><br><b>{analyzer_out}</b>. Here's the daily matches report:",
            [EXCEL_PATH, MASTER_EXCEL_PATH]  
        )
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        send_email(
            "Job Scraper - ERROR",
            f"<pre>{traceback.format_exc()}</pre>",
        )
        sys.exit(1)

if __name__ == "__main__":
    main()