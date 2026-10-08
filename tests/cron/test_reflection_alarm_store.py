"""RED tracer for persisted reflection failure alarm job metadata."""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest

from cron import jobs


def _loaded_job(job_id: str) -> dict[str, Any]:
    return next(job for job in jobs.load_jobs() if job["id"] == job_id)


def test_create_job_accepts_reflection_failure_alarm_and_defaults_absent(tmp_path) -> None:
    """The public create API must expose opt-in persistence, not rely on raw dict edits."""
    with jobs.use_cron_store(tmp_path):
        signature = inspect.signature(jobs.create_job)
        assert "failure_alarm" in signature.parameters
        assert signature.parameters["failure_alarm"].default is None

        plain = jobs.create_job(prompt="ordinary job", schedule="every 1h")
        assert "failure_alarm" not in plain
        assert "failure_alarm" not in _loaded_job(plain["id"])

        created = cast(Any, jobs.create_job)(
            prompt="reflect on the session",
            schedule="every 1h",
            name="reflection alarm",
            skills=["session-reflection"],
            model="pinned-model",
            provider="openrouter",
            script="/tmp/collect-reflection-context.sh",
            workdir=str(tmp_path),
            deliver="local",
            reasoning_effort="low",
            failure_alarm="Reflection",
        )

        assert created["failure_alarm"] == "reflection"
        stored = _loaded_job(created["id"])
        assert stored["failure_alarm"] == "reflection"
        assert stored["skills"] == ["session-reflection"]
        assert stored["model"] == "pinned-model"
        assert stored["provider"] == "openrouter"
        assert stored["script"] == "/tmp/collect-reflection-context.sh"
        assert stored["workdir"] == str(tmp_path)
        assert stored["deliver"] == "local"
        assert stored["reasoning_effort"] == "low"


def test_update_job_canonicalizes_reflection_failure_alarm(tmp_path) -> None:
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="reflect", schedule="every 1h")

        updated = jobs.update_job(job["id"], {"failure_alarm": " Reflection "})

        assert updated is not None
        assert updated["failure_alarm"] == "reflection"
        assert _loaded_job(job["id"])["failure_alarm"] == "reflection"


def test_update_job_rejects_unknown_failure_alarm_values(tmp_path) -> None:
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="reflect", schedule="every 1h")

        with pytest.raises(ValueError, match="failure_alarm"):
            jobs.update_job(job["id"], {"failure_alarm": "email"})

        assert "failure_alarm" not in _loaded_job(job["id"])


def test_update_job_blank_failure_alarm_clears_persisted_field(tmp_path) -> None:
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="reflect", schedule="every 1h")
        seeded = jobs.update_job(job["id"], {"failure_alarm": "reflection"})
        assert seeded is not None
        assert seeded["failure_alarm"] == "reflection"

        cleared = jobs.update_job(job["id"], {"failure_alarm": "  "})

        assert cleared is not None
        assert "failure_alarm" not in cleared
        assert "failure_alarm" not in _loaded_job(job["id"])


def test_unrelated_update_preserves_alarm_and_existing_job_fields(tmp_path) -> None:
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(
            prompt="reflect on the session",
            schedule="every 2h",
            name="reflection alarm",
            skills=["session-reflection"],
            model="pinned-model",
            provider="openrouter",
            script="/tmp/collect-reflection-context.sh",
            workdir=str(tmp_path),
            deliver="telegram",
            reasoning_effort="medium",
        )
        with_alarm = jobs.update_job(job["id"], {"failure_alarm": "reflection"})
        assert with_alarm is not None
        before = _loaded_job(job["id"])

        renamed = jobs.update_job(job["id"], {"name": "renamed reflection alarm"})

        assert renamed is not None
        stored = _loaded_job(job["id"])
        assert stored["name"] == "renamed reflection alarm"
        for field in (
            "failure_alarm",
            "prompt",
            "schedule",
            "schedule_display",
            "next_run_at",
            "skills",
            "skill",
            "model",
            "provider",
            "script",
            "workdir",
            "deliver",
            "reasoning_effort",
        ):
            assert stored.get(field) == before.get(field)
