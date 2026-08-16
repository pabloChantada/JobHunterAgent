"""
Job scraper for JobHunterAgent.

Currently scrapes LinkedIn via Apify (automation-lab/linkedin-jobs-scraper),
which uses LinkedIn's public guest API (no login/cookies needed) and has a
documented input schema: https://apify.com/automation-lab/linkedin-jobs-scraper/input-schema
Built with a pluggable `JobScraper` interface so InfoJobs, Indeed, etc. can
be added later as separate classes without touching this file's core logic.
"""

import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from apify_client import ApifyClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATA_DIR / "scraped_offers.json"

JOB_TITLES = [
    "AI Engineer",
    "Machine Learning Engineer",
    "LLM Engineer",
    "GenAI Engineer",
]

SEARCH_QUERIES = [
    {"location": "A Coruña, Spain", "work_type": "hybrid", "date_posted": "r86400"},
    {"location": "A Coruña, Spain", "work_type": "on-site", "date_posted": "r86400"},
    {"location": "Santiago de Compostela, Spain", "work_type": "hybrid", "date_posted": "r86400"},
    {"location": "Santiago de Compostela, Spain", "work_type": "on-site", "date_posted": "r86400"},
    {"location": "Vigo, Spain", "work_type": "hybrid", "date_posted": "r86400"},
    {"location": "Vigo, Spain", "work_type": "on-site", "date_posted": "r86400"},
    {"location": "Spain", "work_type": "hybrid", "date_posted": "r86400"},
    {"location": "Spain", "work_type": "remote", "date_posted": "r86400"},
    # {"location": "Worldwide", "work_type": "remote", "date_posted": "r86400"},
]


# Titles containing these are filtered out
EXCLUDED_TITLE_KEYWORDS = [
    "senior", "sr.", "sr", "snr", "staff", "principal", "lead", "architect",
    "director", "head", "manager", "vp", "chief", "jefe", "supervisor", "cto",
    "mid", "semi-senior", "semi senior", "experienced", "expert", "specialist", "advanced",
    "founder", "co-founder"
]

# this list only affects offers where workplace_type == "On-site". Extend as needed.
EXCLUDED_ONSITE_LOCATIONS = ["madrid", "barcelona", "valencia", "sevilla", "bilbao"]

# (https://apify.com/automation-lab/linkedin-jobs-scraper/input-schema):
# workplaceType: "1"=On-site, "2"=Remote, "3"=Hybrid.
LINKEDIN_WORK_TYPE_CODES = {"on-site": "1", "remote": "2", "hybrid": "3"}

MAX_JOBS_PER_QUERY = 15  # per location, per day | total = 25x7 = 175 max per run

# Persistent state so daily runs only return new postings
# instead of re-saving the same jobs every day. Stored as
# {job_key: date_first_seen_iso}; entries older than SEEN_ID_RETENTION_DAYS
# are pruned on each run so the file doesn't grow forever.
SEEN_IDS_PATH = DATA_DIR / "seen_job_ids.json"
SEEN_ID_RETENTION_DAYS = 60


class JobOffer(BaseModel):
    """Normalized representation of a scraped job posting."""

    id: str = Field(
        description="Unique identifier for the job posting, usually the job ID."
    )
    title: str = Field(
        description="Job title as listed on the posting."
    )
    company: str = Field(
        description="Hiring company name."
    )
    description: str = Field(
        description=(
            "Plain-text job description (descriptionText from the actor's output)."
        )
    )
    url: str = Field(
        description="URL of the job posting."
    )
    source: str = Field(
        description="Which scraper produced this offer, e.g. 'linkedin'."
    )

    # optional since not every LinkedIn posting includes them.
    # specific for this actor's output
    location: str = Field(
        default="", description="Location string as reported by the posting."
    )
    workplace_type: str | None = Field(
        default=None, description="'On-site', 'Remote', 'Hybrid', or None if unknown."
    )
    salary: str | None = Field(
        default=None, description="Salary range as listed, if disclosed."
    )
    seniority_level: str | None = Field(
        default=None, description="Seniority level as tagged by LinkedIn."
    )
    employment_type: str | None = Field(
        default=None, description="Employment type, e.g. 'Full-time', 'Internship'."
    )
    applicants_count: int | None = Field(
        default=None, description="Number of applicants at scrape time, if shown."
    )

    def is_valid(self) -> bool:
        """Skip postings with no description."""
        return bool(self.description)


class JobScraper(ABC):
    """Base interface so new sources (InfoJobs, Indeed, ...) can plug in
    without changing run_scraper() or the output format."""

    # The interface intentionally has a single abstract method by design.
    # pylint: disable=too-few-public-methods
    source_name: str = "unknown"

    @abstractmethod
    def scrape(self) -> list[JobOffer]:
        """Scrape job offers from the source and return normalized results."""
        raise NotImplementedError


class LinkedInScraper(JobScraper):
    """Scraper implementation for the LinkedIn Apify actor."""

    # This class deliberately exposes a narrow public surface for the adapter.
    # pylint: disable=too-few-public-methods
    source_name = "linkedin"
    actor_id = "automation-lab/linkedin-jobs-scraper"

    def __init__(
        self,
        client: ApifyClient,
        job_titles: list[str],
        search_queries: list[dict[str, str]],
        max_jobs_per_query: int = MAX_JOBS_PER_QUERY,
        scrape_job_details: bool = True,
    ):
        self.client = client
        self.job_titles = job_titles
        self.search_queries = search_queries
        self.max_jobs_per_query = max_jobs_per_query
        # Full description/salary/seniority require scrapeJobDetails=True
        self.scrape_job_details = scrape_job_details

    def scrape(self) -> list[JobOffer]:
        offers = []
        for query in self.search_queries:
            offers.extend(self._run_single_search(query))
        return offers

    def _build_run_input(self, query: dict[str, str]) -> dict[str, Any]:
        # One actor run covers ALL job_titles for this location via
        # searchQuery (first title) + searchQueries (the rest). This keeps
        # us at 1 run per SEARCH_QUERIES entry per day instead of 1 per
        # title x location, which matters because each run has a flat
        # $0.005 start fee on top of the per-listing cost.
        return {
            "searchQuery": self.job_titles[0],
            "searchQueries": self.job_titles[1:],
            "location": query["location"],
            "workplaceType": LINKEDIN_WORK_TYPE_CODES[query["work_type"]],
            "datePosted": query["date_posted"],
            "maxJobs": self.max_jobs_per_query,
            "scrapeJobDetails": self.scrape_job_details,
            "sortBy": "DD",  # most recent first
        }

    def _run_single_search(
        self, query: dict[str, str], max_retries: int = 2
    ) -> list[JobOffer]:
        """Run one LinkedIn search and retry on transient Apify failures."""
        run_input = self._build_run_input(query)
        logger.info(
            "Searching %d titles in %s (%s)",
            len(self.job_titles),
            query["location"],
            query["work_type"],
        )

        # use 2 retries as a simple backoff in case of rate limits, blocks or anything
        for attempt in range(1, max_retries + 1):
            try:
                run = self.client.actor(self.actor_id).call(run_input=run_input)
                return self._parse_dataset(run.default_dataset_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # only log non-costing errors here: rate limit, blocks
                logger.warning(
                    "Attempt %d/%d failed for %s (Apify API error): %s",
                    attempt,
                    max_retries,
                    query["location"],
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(2 * attempt)  # simple backoff before retrying

        logger.error(
            "Giving up on %s after %d attempts", query["location"], max_retries
        )
        return []

    def _parse_dataset(self, dataset_id: str) -> list[JobOffer]:
        """Normalize a dataset returned by the LinkedIn Apify actor."""
        offers = []
        for item in self.client.dataset(dataset_id).iterate_items():
            # descriptionText is the plain-text version of descriptionHtml.
            offer = JobOffer(
                id=str(item.get("id") or item.get("url", "")),
                title=item.get("title", "Unknown"),
                company=item.get("companyName", "Confidential"),
                description=item.get("descriptionText", ""),
                url=item.get("url", ""),
                source=self.source_name,
                location=item.get("location", ""),
                workplace_type=item.get("workplaceType"),
                salary=item.get("salary"),
                seniority_level=item.get("seniorityLevel"),
                employment_type=item.get("employmentType"),
                applicants_count=item.get("applicantsCount"),
            )
            if offer.is_valid():
                offers.append(offer)
        return offers


def matches_profile(offer: JobOffer) -> bool:
    """Filter out senior titles and high-cost on-site relocations."""
    title_lower = offer.title.lower()
    if any(keyword in title_lower for keyword in EXCLUDED_TITLE_KEYWORDS):
        return False

    if offer.workplace_type == "On-site":
        location_lower = offer.location.lower()
        if any(city in location_lower for city in EXCLUDED_ONSITE_LOCATIONS):
            return False

    return True


def deduplicate(offers: list[JobOffer]) -> list[JobOffer]:
    """Drop repeated offers within a single run."""
    # use a set since we won't have duplicates, and it's O(1) lookup time instead
    # of O(n) for a list.
    seen = set()
    unique = []
    for offer in offers:
        key = offer.url or offer.id
        # add if we have a key and haven't seen it yet
        if key and key not in seen:
            seen.add(key)
            unique.append(offer)
    return unique


def save_offers(offers: list[JobOffer], path: Path) -> None:
    """Persist job offers to a JSON file following the project schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump([dict(offer) for offer in offers], file_handle, ensure_ascii=False, indent=4)


def offer_key(offer: JobOffer) -> str:
    """Return a stable deduplication key for an offer."""
    return offer.url or offer.id


def load_seen_ids(path: Path) -> dict[str, str]:
    """Load the {job_key: date_first_seen_iso} state map."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Could not read %s (%s) - starting with empty seen-ids state",
            path.name,
            exc,
        )
        return {}


def prune_seen_ids(
    seen: dict[str, str],
    retention_days: int = SEEN_ID_RETENTION_DAYS,
) -> dict[str, str]:
    """Drop entries older than retention_days so the file does not grow forever."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    pruned: dict[str, str] = {}
    for key, date_str in seen.items():
        try:
            if datetime.fromisoformat(date_str) >= cutoff:
                pruned[key] = date_str
        except ValueError:
            continue  # drop malformed entries rather than crash
    return pruned


def save_seen_ids(seen: dict[str, str], path: Path) -> None:
    """Persist the seen offer IDs to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(seen, file_handle, ensure_ascii=False, indent=2)


def run_scraper() -> None:
    """Run the configured scrapers, filter and deduplicate results, and save new data."""
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise ValueError("No APIFY_API_TOKEN found in environment variables.")

    client = ApifyClient(token)

    scrapers = [
        LinkedInScraper(client, JOB_TITLES, SEARCH_QUERIES),
        # Future sources go here, the classes just need to implement the JobScraper interface:
        # following their respective API schemas
        # InfoJobsScraper(client, JOB_TITLES, SEARCH_QUERIES),
        # IndeedScraper(client, JOB_TITLES, SEARCH_QUERIES),
    ]

    all_offers = []
    fatal_error = None

    for scraper in scrapers:
        logger.info("Running scraper: %s", scraper.source_name)
        try:
            all_offers.extend(scraper.scrape())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Anything that reaches here is NOT an API error.
            # If there's an error here we carry it on every remaining query/scraper
            # and pay for it each time. What was already collected before the
            # failure is kept and still gets saved below.
            logger.critical(
                "Scraper '%s' hit an unexpected error. Aborting the rest of this "
                "run. %d offers already collected will still be saved.",
                scraper.source_name,
                len(all_offers),
                exc_info=True,
            )
            fatal_error = exc
            break

    profile_matched = [offer for offer in all_offers if matches_profile(offer)]
    unique_offers = deduplicate(profile_matched)

    # remove any offers we've already seen in previous runs, and save the new ones
    # to the output file
    seen_ids = prune_seen_ids(load_seen_ids(SEEN_IDS_PATH))
    new_offers = [offer for offer in unique_offers if offer_key(offer) not in seen_ids]

    save_offers(new_offers, OUTPUT_PATH)

    # save the new offers to the seen_ids state file so we don't re-save them in
    # future runs
    today = datetime.now(timezone.utc).date().isoformat()
    for offer in unique_offers:
        seen_ids.setdefault(offer_key(offer), today)
    save_seen_ids(seen_ids, SEEN_IDS_PATH)

    logger.info(
        "Done. %d new offers saved to %s (%d matched profile / %d raw results, "
        "%d total tracked ids)",
        len(new_offers),
        OUTPUT_PATH.name,
        len(unique_offers),
        len(all_offers),
        len(seen_ids),
    )

    if fatal_error is not None:
        # Non-zero exit so GitHub Actions run gets flagged as failed.
        logger.critical("Run finished in a ERROR. Check the error above before the next run.")
        sys.exit(1)


if __name__ == "__main__":
    run_scraper()
