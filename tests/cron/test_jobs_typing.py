"""Typing-regression behavior tests for cron/jobs.py."""

from datetime import datetime, timedelta, timezone

import pytest

import cron.jobs as jobs


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Route the cron store to an isolated temp directory."""
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class _FakeCroniter:
    def __init__(self, _expr, base=None):
        self.base = base if isinstance(base, datetime) else datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.count = 0

    def get_next(self, _kind):
        self.count += 1
        return self.base + timedelta(hours=self.count)


def test_cron_schedule_validation_uses_callable_croniter(monkeypatch):
    monkeypatch.setattr(jobs, "HAS_CRONITER", True)
    monkeypatch.setattr(jobs, "croniter", _FakeCroniter)

    parsed = jobs.parse_schedule("0 9 * * *")

    assert parsed == {"kind": "cron", "expr": "0 9 * * *", "display": "0 9 * * *"}


def test_compute_next_run_uses_callable_croniter(monkeypatch):
    base = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "HAS_CRONITER", True)
    monkeypatch.setattr(jobs, "croniter", _FakeCroniter)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: base)
    monkeypatch.setattr(jobs, "get_timezone", lambda: timezone.utc)

    next_run = jobs.compute_next_run({"kind": "cron", "expr": "0 9 * * *"})

    assert next_run == datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc).isoformat()


def test_schedule_cadence_uses_callable_croniter(monkeypatch):
    base = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "HAS_CRONITER", True)
    monkeypatch.setattr(jobs, "croniter", _FakeCroniter)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: base)
    jobs._cron_cadence_cache.clear()

    cadence = jobs._schedule_cadence_seconds({"kind": "cron", "expr": "0 * * * *"})

    assert cadence == 3600.0


def test_create_job_without_name_uses_string_label(cron_store):
    job = jobs.create_job(prompt="", schedule="every 1h", skills=["daily-brief"])

    assert job["name"] == "daily-brief"
    assert isinstance(job["next_run_at"], str)
