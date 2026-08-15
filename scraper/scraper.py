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
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "scraped_offers.json"

JOB_TITLES = [
    "AI Engineer",
    "Machine Learning Engineer",
    "LLM Engineer",
    "GenAI Engineer",
    "Ingeniero de Inteligencia Artificial",
    "Ingeniero Machine Learning",
]

# Each entry is one "sweep": a location + workplace type + recency filter.
# Profile: remote-first globally, or hybrid anywhere in Spain (including
# Madrid/Barcelona - those are only out of scope if the role is fully
# on-site/in-office, which EXCLUDED_ONSITE_LOCATIONS below handles).
SEARCH_QUERIES: list[dict[str, str]] = [
    {"location": "A Coruña, Spain", "work_type": "hybrid", "date_posted": "r86400"},
    {"location": "Santiago de Compostela, Spain", "work_type": "hybrid", "date_posted": "r86400"},
    {"location": "Spain", "work_type": "hybrid", "date_posted": "r86400"},
    {"location": "Spain", "work_type": "remote", "date_posted": "r86400"},
    {"location": "Worldwide", "work_type": "remote", "date_posted": "r86400"},
]

# Titles containing these (case-insensitive) are filtered out post-scrape -
# matches the "skip Senior/Lead/Architect" application rule.
EXCLUDED_TITLE_KEYWORDS = [
    "senior", "sr.", "staff", "principal", "lead", "architect",
    "director", "head of", "manager", "vp ", "chief",
]

# Fully in-office (on-site) roles only make sense within commuting distance
# of home (Galicia) - relocating full-time to a distant office is out of
# scope. Hybrid roles in these same cities are fine; this list only affects
# offers where workplace_type == "On-site". Extend as needed.
EXCLUDED_ONSITE_LOCATIONS = ["madrid", "barcelona", "valencia", "sevilla", "bilbao"]

# Confirmed against the actor's published input schema
# (https://apify.com/automation-lab/linkedin-jobs-scraper/input-schema):
# workplaceType: "1"=On-site, "2"=Remote, "3"=Hybrid.
LINKEDIN_WORK_TYPE_CODES = {"on-site": "1", "remote": "2", "hybrid": "3"}

MAX_JOBS_PER_QUERY = 25  # per location, per day - shared across all JOB_TITLES in that call

# Persistent "seen" state so daily runs only surface genuinely new postings
# instead of re-downloading the same jobs every day. Stored as
# {job_key: date_first_seen_iso}; entries older than SEEN_ID_RETENTION_DAYS
# are pruned on each run so the file doesn't grow forever.
SEEN_IDS_PATH = DATA_DIR / "seen_job_ids.json"
SEEN_ID_RETENTION_DAYS = 60


@dataclass
class JobOffer:
    id: str
    title: str
    company: str
    description: str
    url: str
    source: str
    # Extra fields this actor provides that the old one didn't - all
    # optional since not every LinkedIn posting includes them.
    location: str = ""
    workplace_type: str | None = None  # "On-site" / "Remote" / "Hybrid" / None
    salary: str | None = None
    seniority_level: str | None = None
    employment_type: str | None = None
    applicants_count: int | None = None

    def is_valid(self) -> bool:
        # Skip postings with no description; they're not useful downstream.
        return bool(self.description)


class JobScraper(ABC):
    """Base interface so new sources (InfoJobs, Indeed, ...) can plug in
    without changing run_scraper() or the output format."""

    source_name: str = "unknown"

    @abstractmethod
    def scrape(self) -> list[JobOffer]:
        ...


class LinkedInScraper(JobScraper):
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
        # (default). Set False for a much faster run when you only need
        # title/company/location/url to eyeball volume.
        self.scrape_job_details = scrape_job_details

    def scrape(self) -> list[JobOffer]:
        offers: list[JobOffer] = []
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
            "sortBy": "DD",  # most recent first - matches the point of date_posted filters
        }

    def _run_single_search(
        self, query: dict[str, str], max_retries: int = 2
    ) -> list[JobOffer]:
        run_input = self._build_run_input(query)
        logger.info(
            "Searching %d titles in %s (%s)",
            len(self.job_titles),
            query["location"],
            query["work_type"],
        )

        for attempt in range(1, max_retries + 1):
            try:
                run = self.client.actor(self.actor_id).call(run_input=run_input)
                return self._parse_dataset(run["defaultDatasetId"])
            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
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
        offers = []
        for item in self.client.dataset(dataset_id).iterate_items():
            # Field names per the actor's documented output schema.
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
    """Post-scrape filter for fit with the target profile: no senior/lead/
    architect-type titles, and no fully on-site postings requiring
    relocation to a distant city. Hybrid and remote postings are kept
    regardless of city."""
    title_lower = offer.title.lower()
    if any(keyword in title_lower for keyword in EXCLUDED_TITLE_KEYWORDS):
        return False

    if offer.workplace_type == "On-site":
        location_lower = offer.location.lower()
        if any(city in location_lower for city in EXCLUDED_ONSITE_LOCATIONS):
            return False

    return True


def deduplicate(offers: list[JobOffer]) -> list[JobOffer]:
    """Drop repeated offers within a single run (e.g. same posting matched
    by two job titles in the searchQueries list). Cross-day deduplication
    is handled separately via the seen_job_ids.json state file."""
    seen: set[str] = set()
    unique: list[JobOffer] = []
    for offer in offers:
        key = offer.url or offer.id
        if key and key not in seen:
            seen.add(key)
            unique.append(offer)
    return unique


def save_offers(offers: list[JobOffer], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(o) for o in offers], f, ensure_ascii=False, indent=4)


def offer_key(offer: JobOffer) -> str:
    return offer.url or offer.id


def load_seen_ids(path: Path) -> dict[str, str]:
    """Load the {job_key: date_first_seen_iso} map. Returns {} if the file
    doesn't exist yet or can't be parsed (e.g. first run, or a corrupted
    file) - a scraping run should never crash because of stale state."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s (%s) - starting with empty seen-ids state", path.name, exc)
        return {}


def prune_seen_ids(seen: dict[str, str], retention_days: int = SEEN_ID_RETENTION_DAYS) -> dict[str, str]:
    """Drop entries older than retention_days so the state file doesn't grow
    forever across months of daily runs."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seen, f, ensure_ascii=False, indent=2)


def run_scraper() -> None:
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise ValueError("No APIFY_API_TOKEN found in environment variables.")

    client = ApifyClient(token)

    scrapers: list[JobScraper] = [
        LinkedInScraper(client, JOB_TITLES, SEARCH_QUERIES),
        # Future sources go here, e.g.:
        # InfoJobsScraper(client, JOB_TITLES, SEARCH_QUERIES),
        # IndeedScraper(client, JOB_TITLES, SEARCH_QUERIES),
    ]

    all_offers: list[JobOffer] = []
    for scraper in scrapers:
        logger.info("Running scraper: %s", scraper.source_name)
        all_offers.extend(scraper.scrape())

    profile_matched = [o for o in all_offers if matches_profile(o)]
    unique_offers = deduplicate(profile_matched)

    # Only surface offers we haven't seen in a previous run - this is what
    # makes a daily cron useful instead of re-downloading the same jobs
    # every day.
    seen_ids = prune_seen_ids(load_seen_ids(SEEN_IDS_PATH))
    new_offers = [o for o in unique_offers if offer_key(o) not in seen_ids]

    save_offers(new_offers, OUTPUT_PATH)

    today = datetime.now(timezone.utc).date().isoformat()
    for offer in unique_offers:
        seen_ids.setdefault(offer_key(offer), today)
    save_seen_ids(seen_ids, SEEN_IDS_PATH)

    logger.info(
        "Done. %d new offers saved to %s (%d matched profile / %d raw results, %d total tracked ids)",
        len(new_offers),
        OUTPUT_PATH.name,
        len(unique_offers),
        len(all_offers),
        len(seen_ids),
    )


if __name__ == "__main__":
    run_scraper()