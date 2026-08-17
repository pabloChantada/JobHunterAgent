"""Unit tests for the job scraper."""
import json
from datetime import datetime, timedelta, timezone
import scraper.scraper as scraper_module

from scraper.scraper import (
    JobOffer,
    LinkedInScraper,
    deduplicate,
    load_seen_ids,
    matches_profile,
    offer_key,
    prune_seen_ids,
    save_offers,
    save_seen_ids,
)


def make_offer(**overrides):
    data = {
        "id": "123",
        "title": "AI Engineer",
        "company": "Example Corp",
        "description": "Build machine learning systems.",
        "url": "https://example.com/jobs/123",
        "source": "linkedin",
    }
    data.update(overrides)
    return JobOffer(**data)


def test_job_offer_is_invalid_without_description():
    offer = make_offer(description="")
    assert offer.is_valid() is False


def test_job_offer_is_valid_with_description():
    assert make_offer().is_valid() is True


def test_matches_profile_rejects_senior_titles():
    assert matches_profile(make_offer(title="Senior AI Engineer")) is False
    assert matches_profile(make_offer(title="AI Engineer")) is True


def test_matches_profile_rejects_excluded_on_site_locations():
    offer = make_offer(
        workplace_type="On-site",
        location="Madrid, Spain",
    )
    assert matches_profile(offer) is False


def test_matches_profile_allows_remote_in_excluded_city():
    offer = make_offer(
        workplace_type="Remote",
        location="Madrid, Spain",
    )
    assert matches_profile(offer) is True


def test_deduplicate_uses_url_or_id():
    first = make_offer(id="1", url="https://example.com/job")
    duplicate = make_offer(id="2", url="https://example.com/job")
    unique = make_offer(id="3", url="https://example.com/other")

    result = deduplicate([first, duplicate, unique])

    assert result == [first, unique]


def test_deduplicate_falls_back_to_id_when_url_is_missing():
    first = make_offer(id="same", url="")
    duplicate = make_offer(id="same", url="")

    assert deduplicate([first, duplicate]) == [first]


def test_offer_key_prefers_url():
    offer = make_offer(id="123", url="https://example.com/job")
    assert offer_key(offer) == "https://example.com/job"


def test_offer_key_falls_back_to_id():
    offer = make_offer(id="123", url="")
    assert offer_key(offer) == "123"


def test_build_run_input():
    scraper = LinkedInScraper(
        client=None,
        job_titles=["AI Engineer", "ML Engineer"],
        search_queries=[],
        max_jobs_per_query=10,
        scrape_job_details=False,
    )

    result = scraper._build_run_input(
        {
            "location": "A Coruña, Spain",
            "work_type": "hybrid",
            "date_posted": "r86400",
        }
    )

    assert result["searchQuery"] == "AI Engineer"
    assert result["searchQueries"] == ["ML Engineer"]
    assert result["location"] == "A Coruña, Spain"
    assert result["workplaceType"] == "3"
    assert result["maxJobs"] == 10
    assert result["scrapeJobDetails"] is False
    assert result["sortBy"] == "DD"


def test_parse_dataset_normalizes_items():
    class FakeDataset:
        def iterate_items(self):
            return [
                {
                    "id": 42,
                    "title": "AI Engineer",
                    "companyName": "Example",
                    "descriptionText": "Python role",
                    "url": "https://example.com/42",
                    "location": "Spain",
                    "workplaceType": "Remote",
                    "salary": "€40k",
                    "seniorityLevel": "Entry level",
                    "employmentType": "Full-time",
                    "applicantsCount": 12,
                },
                {
                    "id": 43,
                    "title": "Missing description",
                    "url": "https://example.com/43",
                },
            ]

    class FakeClient:
        def dataset(self, dataset_id):
            return FakeDataset()

    scraper = LinkedInScraper(FakeClient(), ["AI Engineer"], [])

    offers = scraper._parse_dataset("dataset-id")

    assert len(offers) == 1
    assert offers[0].id == "42"
    assert offers[0].company == "Example"
    assert offers[0].workplace_type == "Remote"
    assert offers[0].applicants_count == 12


def test_run_single_search_retries_then_returns_empty(monkeypatch):
    class FakeActor:
        def call(self, run_input):
            raise RuntimeError("temporary failure")

    class FakeClient:
        def actor(self, actor_id):
            return FakeActor()

    scraper_instance = LinkedInScraper(
        FakeClient(),
        ["AI Engineer"],
        [],
    )

    monkeypatch.setattr(scraper_module.time, "sleep", lambda _: None)

    result = scraper_instance._run_single_search(
        {
            "location": "Spain",
            "work_type": "remote",
            "date_posted": "r86400",
        },
        max_retries=2,
    )

    assert result == []


def test_save_and_load_seen_ids(tmp_path):
    path = tmp_path / "seen.json"
    data = {"job-1": "2026-08-17"}

    save_seen_ids(data, path)

    assert load_seen_ids(path) == data


def test_load_seen_ids_returns_empty_for_invalid_json(tmp_path):
    path = tmp_path / "seen.json"
    path.write_text("{invalid", encoding="utf-8")

    assert load_seen_ids(path) == {}


def test_prune_seen_ids_removes_old_and_invalid_dates():
    recent = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()

    result = prune_seen_ids(
        {
            "recent": recent,
            "old": old,
            "invalid": "not-a-date",
        },
        retention_days=60,
    )

    assert "recent" in result
    assert "old" not in result
    assert "invalid" not in result


def test_save_offers_writes_expected_json(tmp_path):
    path = tmp_path / "offers.json"
    offers = [make_offer()]

    save_offers(offers, path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["id"] == "123"
    assert saved[0]["title"] == "AI Engineer"
