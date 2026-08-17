"""Job Hunter Agent - Automate job evaluation and cover letter generation."""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from agent.agent import evaluate_job_offer
from agent.cover_letter import generate_cover_letter_draft
from agent.vectorstore import detect_language_from_text, get_vectorstore

BASE_DIR = Path(__file__).resolve().parent
# MOCK_JOBS_PATH = BASE_DIR / "data" / "mock_offers.json"
JOBS_PATH = BASE_DIR / "data" / "scraped_offers.json"
COVER_LETTERS_DIR = BASE_DIR / "data" / "cover_letters"

MASTER_TRACKER_PATH = BASE_DIR / "data" / "master_tracker.xlsx" # Long-term memory
DAILY_REPORT_PATH = BASE_DIR / "daily_matches.xlsx" # Daily email attachment


def get_already_processed_ids() -> set:
    """Read the existing master tracker and return processed Job IDs.

    Returns:
        Set of Job IDs already logged in the master tracker.
    """
    if not MASTER_TRACKER_PATH.exists():
        return set()

    existing_df = pd.read_excel(MASTER_TRACKER_PATH)
    if "Job ID" not in existing_df.columns:
        return set()

    return set(existing_df["Job ID"].tolist())


def generate_and_save_cover_letter(job: dict) -> str:
    """Retrieve CV context for a job and generate a cover letter.

    Args:
        job: The job dictionary with description, company, title, etc.

    Returns:
        The path to the saved cover letter file.
    """
    # Add context from the CV to the cover letter generation
    lang = detect_language_from_text(job["description"])
    retriever = get_vectorstore().as_retriever(
        search_kwargs={"k": 7, "filter": {"language": lang}}
    )

    # Search for relevant CV chunks based on the job description
    docs = retriever.invoke(job["description"])

    # Remove metadata and join the content of the retrieved documents
    cv_context = "\n\n".join([doc.page_content for doc in docs])

    cl_content = generate_cover_letter_draft(
        job_description=job["description"],
        cv_context=cv_context,
        company_name=job["company"],
        job_title=job["title"],
        user_name=job.get("user_name", "Pablo Chantada"),
    )

    # Clean the company name to create a safe filename
    safe_company_name = "".join(
        c for c in job["company"] if c.isalnum() or c in (" ", "_")
    ).replace(" ", "_")
    file_name = f"{job['id']}_{safe_company_name}.txt"
    file_path = COVER_LETTERS_DIR / file_name

    # Write the letter to a file
    with open(file_path, "w", encoding="utf-8") as cl_file:
        cl_file.write(cl_content)

    # Path for the n8n sheet
    return f"data/cover_letters/{file_name}"


def process_single_job(job: dict) -> dict:
    """Evaluate one job offer and generate a cover letter if recommended.

    Args:
        job: The job dictionary with description, company, title, etc.

    Returns:
        A dictionary with results to append to the tracker.
    """
    # Set ollama as the default fallback
    current_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    logging.info("Evaluating: %s at %s...", job["title"], job["company"])
    start_eval_time = time.time()

    # Call the LLM to evaluate the offer
    evaluation = evaluate_job_offer(job["description"])
    cover_letter_path = "N/A"  # Default value

    eval_time = time.time() - start_eval_time
    verdict_text = "APPLY" if evaluation.verdict else "REJECTED"
    logging.info(
        "Evaluation completed in %.2f seconds. Verdict: %s",
        eval_time,
        verdict_text,
    )

    # Combine the list of reasons into a single string separated by new lines
    reasons_str = "\n- ".join(evaluation.reasons)
    if evaluation.reasons:
        reasons_str = "- " + reasons_str

    # Only generate a cover letter if the job is recommended to apply
    if evaluation.verdict:
        start_cl_time = time.time()
        cover_letter_path = generate_and_save_cover_letter(job)
        cl_time = time.time() - start_cl_time
        logging.info(
            "Cover letter generated in %.2f seconds. Saved to: %s",
            cl_time,
            cover_letter_path,
        )

    # Format the results
    return {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Job ID": job["id"],
        "Title": job["title"],
        "Company": job["company"],
        "Score": evaluation.score,
        "Verdict": "✅ APPLY" if evaluation.verdict else "❌ REJECTED",
        "Reasons": reasons_str,
        "Url": job.get("url", "N/A"),
        "Provider": current_provider,
        "Cover Letter Path": cover_letter_path,
    }


def save_results(results_list: list):
    """Save results to the Daily Report and append to the Master Tracker.

    Args:
        results_list: List of result dictionaries to save.
    """
    new_data_df = pd.DataFrame(results_list)

    # Update the Master Tracker (Long-term memory)
    if MASTER_TRACKER_PATH.exists():
        existing_df = pd.read_excel(MASTER_TRACKER_PATH)
        updated_df = pd.concat(
            [existing_df, new_data_df], ignore_index=True
        )
        updated_df.to_excel(MASTER_TRACKER_PATH, index=False)
    else:
        new_data_df.to_excel(MASTER_TRACKER_PATH, index=False)

    # Create the Daily Report (Only today's data)
    new_data_df.to_excel(DAILY_REPORT_PATH, index=False)

    logging.info(
        "[SUCCESS] Saved %d new evaluations to %s and updated master database.",
        len(new_data_df),
        DAILY_REPORT_PATH.name,
    )


def process_jobs():
    """Load job offers and process them through the evaluation pipeline."""
    # Switch to real job offers once the scraper is implemented
    if not JOBS_PATH.exists():
        logging.info("Error: File not found: %s", JOBS_PATH)
        sys.exit(1)

    with open(JOBS_PATH, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    # Skip jobs we've already evaluated in a previous run
    already_processed_ids = get_already_processed_ids()
    jobs_to_process = [
        job for job in jobs if job["id"] not in already_processed_ids
    ]
    skipped_count = len(jobs) - len(jobs_to_process)

    if skipped_count:
        logging.info(
            "Skipping %d job(s) already present in master database.",
            skipped_count,
        )

    if not jobs_to_process:
        logging.info("No new jobs to evaluate today.")
        return

    # Make the cover letter directory if it doesn't exist
    COVER_LETTERS_DIR.mkdir(parents=True, exist_ok=True)

    results_list = []
    for idx, job in enumerate(jobs_to_process):
        logging.info(
            "Processing job %d/%d", idx + 1, len(jobs_to_process)
        )
        result = process_single_job(job)
        results_list.append(result)
        time.sleep(3)

    save_results(results_list)

    # Extra for gmail message
    total_offers = len(results_list)
    matches = sum(1 for r in results_list if "APPLY" in r["Verdict"])

    print(
        f"Final Summary: I've evaluated {total_offers} offers. "
        f"You have {matches} new matches today."
    )

def clean_files():
    """Clean up the cover letters directory and the daily report.

    Keep the master tracker intact.
    """
    if COVER_LETTERS_DIR.exists():
        for file in COVER_LETTERS_DIR.iterdir():
            if file.is_file():
                file.unlink()
        logging.info("Cleaned up cover letters in %s", COVER_LETTERS_DIR)
    if DAILY_REPORT_PATH.exists():
        DAILY_REPORT_PATH.unlink()
        logging.info("Deleted daily report file %s", DAILY_REPORT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job Hunter Agent")
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Clean up cover letters and daily report before processing.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Activate verbose logging",
    )

    args = parser.parse_args()

    LOG_LEVEL = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.clean:
        clean_files()

    process_jobs()
