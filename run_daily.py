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

from agent.vectorstore import index_all_cvs

DATA_DIR = Path("data")  # private data repo, not checked into git
EXCEL_PATH = Path("daily_matches.xlsx") 
MASTER_EXCEL_PATH = DATA_DIR / "master_matches.xlsx"

PATH_CV = Path("data/cv")
PERSIST_DIR = "data/chroma_cv"

def ensure_vectorstore():
    """Ensure the vector store is populated."""
    chroma_path = Path(PERSIST_DIR)
    
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        print("The vector store is empty. Indexing all CVs...")
        index_all_cvs()
    else:
        print("The vector store is populated.")

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
        #  Run the scraper to find new job offers
        print("Starting scraper...")
        scraper_out = run_step(["python", "scraper/scraper.py"])
        print(scraper_out)

        # Ensure CV indexation (needed for GA)
        print("Checking vector store database...")
        ensure_vectorstore()
        
        # Run the evaluator agent
        print("Starting LLM evaluation...")
        analyzer_out = run_step(["python", "main.py", "--clean"])
        
        # Send the email
        print("Sending email report...")
        send_email(
            "Job Scraper - Daily Run",
            f"Daily run completed.<br><br><b>{analyzer_out}</b><br><br>Here's the daily matches report:",
            [EXCEL_PATH, MASTER_EXCEL_PATH]  
        )
        print("Process completed successfully!")

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        send_email(
            "Job Scraper - ERROR",
            f"<pre>{tb}</pre>",
        )
        sys.exit(1)

if __name__ == "__main__":
    main()