"""Behavior-preservation tests for cron CLI typing narrowings."""

import argparse


def test_dispatch_display_ignores_missing_dispatch_metadata():
    from hermes_cli.cron import _dispatch_display

    assert _dispatch_display(None) is None
    assert _dispatch_display({"scheduled_at": "2026-01-01T00:00:00Z"}) is None


def test_cron_run_restores_async_delivery_context(monkeypatch):
    from gateway.session_context import _SESSION_ASYNC_DELIVERY
    from hermes_cli import cron

    observed = []

    def fake_cron_api(**kwargs):
        observed.append((kwargs, _SESSION_ASYNC_DELIVERY.get()))
        return {"success": True, "job": {"name": "daily", "next_run_at": "soon"}}

    token = _SESSION_ASYNC_DELIVERY.set(True)
    try:
        monkeypatch.setattr(cron, "_cron_api", fake_cron_api)

        assert cron._job_action("run", "job-1", "Triggered") == 0
        assert observed == [({"action": "run", "job_id": "job-1"}, False)]
        assert _SESSION_ASYNC_DELIVERY.get() is True
    finally:
        _SESSION_ASYNC_DELIVERY.reset(token)


def test_cron_notepad_set_validates_before_storage(monkeypatch):
    from cron import notepad
    from hermes_cli.cron import cron_notepad

    stored = []

    def fake_set_note(job_id: str, key: str, value: str):
        stored.append((job_id, key, value))
        return {"job_id": job_id, "key": key, "value": value}

    monkeypatch.setattr(notepad, "set_note", fake_set_note)

    missing_value = argparse.Namespace(job_id="job-1", notepad_action="set", key="cursor", value=None)
    assert cron_notepad(missing_value) == 1
    assert stored == []

    valid = argparse.Namespace(job_id="job-1", notepad_action="set", key="cursor", value="42")
    assert cron_notepad(valid) == 0
    assert stored == [("job-1", "cursor", "42")]
