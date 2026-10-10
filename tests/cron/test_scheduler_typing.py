"""Behavioral contracts for scheduler typing-only corrections."""

from __future__ import annotations

import errno
from types import SimpleNamespace


def test_finalize_cron_session_releases_original_session_through_bounded_proxy(monkeypatch):
    import cron.scheduler as scheduler

    calls: list[tuple[str, object]] = []

    class Session:
        def get_compression_tip(self, _session_id):
            calls.append(("tip", self))
            return None

        def set_session_title(self, _session_id, _title):
            calls.append(("title", self))

        def session_lifecycle_statuses(self, session_ids):
            calls.append(("statuses", self))
            return {session_ids[0]: "complete"}

        def end_session(self, _session_id, _reason):
            calls.append(("end", self))

    session = Session()
    released = []

    monkeypatch.setattr(
        scheduler,
        "_run_cron_cleanup_with_timeout",
        lambda fn, **_kwargs: fn() is None or True,
    )
    monkeypatch.setattr("hermes_state_registry.release_or_close", released.append)

    agent = SimpleNamespace(session_id="cron_session", _end_session_on_close=True)
    scheduler._finalize_cron_session(session, agent, "job-1", "Job One", "cron_session")

    assert released == [session]
    assert [name for name, obj in calls] == ["tip", "title", "statuses", "end"]
    assert all(obj is session for _name, obj in calls)
    assert agent._end_session_on_close is False


def test_external_dispatch_failure_marks_delivery_error_before_fire_owner(monkeypatch):
    import cron.scheduler as scheduler

    def handoff_boom(_job):
        raise RuntimeError("worker exited before ownership acknowledgement")

    marked = []

    def mark_job_run(job_id, success, error=None, delivery_error=None, **kwargs):
        marked.append((job_id, success, error, delivery_error, kwargs))
        return True

    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", handoff_boom)
    monkeypatch.setattr(scheduler, "_deliver_crash_failure", lambda *_a, **_k: ("delivery failed", "failed"))
    monkeypatch.setattr(scheduler, "mark_job_run", mark_job_run)
    monkeypatch.setattr(scheduler, "finish_execution", lambda *_a, **_k: None)

    job = {
        "id": "job-dispatch",
        "execution_id": "exec-1",
        "fire_claim": {"by": "owner-1"},
    }

    assert scheduler.run_one_job(job, adapters=None) is True

    assert len(marked) == 1
    job_id, success, error, delivery_error, kwargs = marked[0]
    assert job_id == "job-dispatch"
    assert success is False
    assert error.startswith("Restart-safe cron worker dispatch failed: ")
    assert delivery_error == "delivery failed"
    assert kwargs == {"expected_fire_owner": "owner-1"}


def test_release_tick_lock_suppresses_unlock_error_and_still_closes(monkeypatch):
    import cron.scheduler as scheduler

    calls = []

    class LockFile:
        def fileno(self):
            return 7

        def close(self):
            calls.append("close")

    def locking(fd, mode, size):
        calls.append((fd, mode, size))
        raise OSError(errno.EACCES, "not locked")

    monkeypatch.setattr(scheduler, "fcntl", None)
    monkeypatch.setattr(
        scheduler,
        "msvcrt",
        SimpleNamespace(locking=locking, LK_UNLCK=123),
    )

    scheduler._release_tick_lock(LockFile())

    assert calls == [(7, 123, 1), "close"]
