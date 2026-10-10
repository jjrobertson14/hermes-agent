"""Typing-contract regression coverage for tools.cronjob_tools."""

import json

from tools.cronjob_tools import (
    _hand_off_primary_routed_run,
    _run_heartbeat,
    cronjob,
)


def _job(job_id: str = "job-typing-1") -> dict:
    return {
        "id": job_id,
        "name": "typing job",
        "prompt": "hi",
        "enabled": True,
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
    }


def test_update_reports_missing_job_when_store_loses_record_after_resolution(monkeypatch):
    monkeypatch.setattr("tools.cronjob_tools.resolve_job_ref", lambda _job_id: _job())
    monkeypatch.setattr("tools.cronjob_tools.update_job", lambda _job_id, _updates: None)

    out = json.loads(cronjob(action="update", job_id="job-typing-1", name="renamed"))

    assert out["success"] is False
    assert "not found" in out["error"]


def test_pause_reports_missing_job_when_store_loses_record_after_resolution(monkeypatch):
    monkeypatch.setattr("tools.cronjob_tools.resolve_job_ref", lambda _job_id: _job())
    monkeypatch.setattr("tools.cronjob_tools.pause_job", lambda _job_id, reason=None: None)

    out = json.loads(cronjob(action="pause", job_id="job-typing-1"))

    assert out["success"] is False
    assert "not found" in out["error"]


def test_resume_reports_missing_job_when_store_loses_record_after_resolution(monkeypatch):
    monkeypatch.setattr("tools.cronjob_tools.resolve_job_ref", lambda _job_id: _job())
    monkeypatch.setattr("tools.cronjob_tools.resume_job", lambda _job_id: None)

    out = json.loads(cronjob(action="resume", job_id="job-typing-1"))

    assert out["success"] is False
    assert "not found" in out["error"]


def test_primary_routed_handoff_reports_missing_job_when_trigger_loses_record(monkeypatch):
    monkeypatch.setattr("tools.cronjob_tools.is_job_runnable", lambda _job: True)
    monkeypatch.setattr("tools.cronjob_tools._primary_routed_delivery_platforms", lambda _job: {"telegram"})
    monkeypatch.setattr("hermes_cli.cron._builtin_gateway_liveness", lambda: True)
    monkeypatch.setattr("tools.cronjob_tools.trigger_job", lambda _job_id, extra_prompt=None: None)

    result = _hand_off_primary_routed_run(_job())
    assert result is not None
    out = json.loads(result)

    assert out["success"] is False
    assert "no longer exists" in out["error"]


def test_run_heartbeat_accepts_absent_activity_callback(monkeypatch):
    monkeypatch.setattr("tools.environments.base.get_activity_callback", lambda: None)

    with _run_heartbeat("typing job"):
        pass
