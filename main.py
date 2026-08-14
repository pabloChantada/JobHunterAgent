import argparse
import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

from agent.vectorstore import vectorstore, detect_language_from_text
from agent.agent import evaluate_job_offer
from agent.cover_letter import generate_cover_letter_draft

BASE_DIR = Path(__file__).resolve().parent
MOCK_JOBS_PATH = BASE_DIR / "data" / "mock_offers.json"
COVER_LETTERS_DIR = BASE_DIR / "data" / "cover_letters"

MASTER_TRACKER_PATH = BASE_DIR / "data" / "master_tracker.xlsx" # Long-term memory
DAILY_REPORT_PATH = BASE_DIR / "daily_matches.xlsx" # Daily email attachment


def get_already_processed_ids() -> set:
    """Read the existing master tracker (if any) and return the set of Job IDs already logged,
    so we never re-evaluate, re-generate, or duplicate a row for the same job."""
    if not MASTER_TRACKER_PATH.exists():
        return set()

    existing_df = pd.read_excel(MASTER_TRACKER_PATH)
    if "Job ID" not in existing_df.columns:
        return set()

    return set(existing_df["Job ID"].tolist())


def generate_and_save_cover_letter(job: dict) -> str:
    """Retrieve CV context for this job, generate the cover letter, write it to disk,
    and return the resolved file path."""
    # Add context from the CV to the cover letter generation
    lang = detect_language_from_text(job['description'])
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 7,
            "filter": {"language": lang}
        }
    )

    # Search for relevant CV chunks based on the job description
    docs = retriever.invoke(job['description'])

    # Remove metadata and join the content of the retrieved documents
    cv_context = "\n\n".join([doc.page_content for doc in docs])

    cl_content = generate_cover_letter_draft(
        job_description=job['description'],
        cv_context=cv_context,
        company_name=job['company'],
        job_title=job['title'],
        user_name=job.get('user_name', 'Pablo Chantada')
    )

    # Clean the company name to create a safe filename
    safe_company_name = "".join(c for c in job['company'] if c.isalnum() or c in (' ', '_')).replace(' ', '_')
    file_name = f"{job['id']}_{safe_company_name}.txt"
    file_path = COVER_LETTERS_DIR / file_name

    # Write the letter to a file
    with open(file_path, "w", encoding="utf-8") as cl_file:
        cl_file.write(cl_content)

    # Resolve the path to avoid issues with handwriting or relative paths
    return str(file_path.resolve())


def process_single_job(job: dict) -> dict:
    """Evaluate one job offer, generate a cover letter if it's a recommended match,
    and return the row to append to the tracker."""
    print(f"Evaluating: {job['title']} at {job['company']}...")

    # Call the LLM to evaluate the offer
    evaluation = evaluate_job_offer(job['description'])
    cover_letter_path = "N/A"  # Default value if no cover letter is generated

    # Combine the list of reasons into a single string separated by new lines for Excel
    reasons_str = "\n- ".join(evaluation.reasons)
    if evaluation.reasons:
        reasons_str = "- " + reasons_str

    # Only generate a cover letter if the job is recommended to apply
    if evaluation.verdict:
        cover_letter_path = generate_and_save_cover_letter(job)

    # Format the results
    return {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Job ID": job['id'],
        "Title": job['title'],
        "Company": job['company'],
        "Score": evaluation.score,
        # I don't really want to use emojis but it's a lot easier than using openpyxl to formar with colors
        # We can style it later with openpyxl if we want to make it prettier
        "Verdict": "✅ APPLY" if evaluation.verdict else "❌ REJECTED",
        "Reasons": reasons_str,
        "Url": job.get('url', 'N/A'),
        "Cover Letter Path": cover_letter_path
    }


def save_results(results_list: list):
    """Save the new results to the Daily Report and append them to the Master Tracker."""
    new_data_df = pd.DataFrame(results_list)

    # Update the Master Tracker (Long-term memory)
    if MASTER_TRACKER_PATH.exists():
        existing_df = pd.read_excel(MASTER_TRACKER_PATH)
        updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        updated_df.to_excel(MASTER_TRACKER_PATH, index=False)
    else:
        new_data_df.to_excel(MASTER_TRACKER_PATH, index=False)
        
    # Create the Daily Report (Only today's data)
    new_data_df.to_excel(DAILY_REPORT_PATH, index=False)
    
    print(f"\n[SUCCESS] Saved {len(new_data_df)} new evaluations to {DAILY_REPORT_PATH.name} and updated master database.")


def process_jobs():
    # Switch to real job offers once the scraper is implemented
    if not MOCK_JOBS_PATH.exists():
        print(f"Error: File not found: {MOCK_JOBS_PATH}")
        sys.exit(1)

    with open(MOCK_JOBS_PATH, 'r', encoding='utf-8') as f:
        jobs = json.load(f)

    # Skip jobs we've already evaluated in a previous run, whether they were
    # approved or rejected.
    already_processed_ids = get_already_processed_ids()
    jobs_to_process = [job for job in jobs if job['id'] not in already_processed_ids]
    skipped_count = len(jobs) - len(jobs_to_process)
    
    if skipped_count:
        print(f"Skipping {skipped_count} job(s) already present in master database.")

    if not jobs_to_process:
        print("No new jobs to evaluate today.")
        return

    # Make the cover letter directory if it doesn't exist
    COVER_LETTERS_DIR.mkdir(parents=True, exist_ok=True)

    results_list = [process_single_job(job) for job in jobs_to_process]

    save_results(results_list)

    # Extra for gmail message
    total_offers = len(results_list)
    matches = sum(1 for r in results_list if "APPLY" in r["Verdict"])

    print(f"Final Summary: I've evaluated {total_offers} offers. You have {matches} new matches today.")

def clean_files():
    """Clean up the cover letters directory and the daily report. Keep the master tracker intact."""
    if COVER_LETTERS_DIR.exists():
        for file in COVER_LETTERS_DIR.iterdir():
            if file.is_file():
                file.unlink()
        print(f"Cleaned up cover letters in {COVER_LETTERS_DIR}")

    if DAILY_REPORT_PATH.exists():
        DAILY_REPORT_PATH.unlink()
        print(f"Deleted daily report file {DAILY_REPORT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job Hunter Agent")
    parser.add_argument('--clean', action='store_true', help="Clean up cover letters and daily report before processing jobs.")
    
    args = parser.parse_args()

    # Clean residual files (cover letters and daily report)
    if args.clean:
        clean_files()

    process_jobs()