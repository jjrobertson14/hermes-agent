"""Boundary tests for reflection recovery-alarm wiring through cron callsites."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest


RECOVERY_FOOTER = "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- cron runtime interrupted 🚨"


@pytest.fixture()
def wiring_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import cron.delivery_queue as delivery_queue
    import cron.executions as executions

    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", home / "cron" / "deliveries.db")
    return executions, delivery_queue, home


def _alarm_key(execution_id: str) -> str:
    return f"{execution_id}:reflection-alarm"


def _create_reflection_job(jobs: Any, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "prompt": "reflect",
        "schedule": "every 1h",
        "name": "Reflection Alarm",
        "deliver": "origin",
        "failure_deliver": "telegram",
        "attach_to_session": True,
        "failure_alarm": "reflection",
        "provider": "openai",
        "model": "openai/gpt-5.5",
        "reasoning_effort": "medium",
    }
    fields.update(overrides)
    return jobs.create_job(**fields)


def _recover_dead_owner(executions: Any, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
    return int(executions.recover_interrupted_executions())


def _pending_recovery_outbox(wiring_env: tuple[Any, Any, Any], monkeypatch: pytest.MonkeyPatch) -> str:
    import cron.jobs as jobs

    executions, _delivery_queue, home = wiring_env
    with jobs.use_cron_store(home):
        job = _create_reflection_job(jobs)
        record = executions.create_execution(job["id"], source="direct")
        assert executions.snapshot_execution_recovery_alarm(
            record["id"], job, RECOVERY_FOOTER, for_failure=True
        ) is True
        assert _recover_dead_owner(executions, monkeypatch) == 1
        return str(record["id"])


def test_claim_fire_snapshots_trusted_alarm_route_before_later_job_mutation(
    wiring_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cron.jobs as jobs
    from cron.scheduler_provider import InProcessCronScheduler

    executions, delivery_queue, home = wiring_env
    with jobs.use_cron_store(home):
        job = _create_reflection_job(jobs)
        claimed = InProcessCronScheduler().claim_fire(job["id"], force=True)
        assert claimed is not None
        execution_id = str(claimed["execution_id"])

        assert jobs.update_job(
            job["id"],
            {
                "failure_alarm": None,
                "failure_deliver": "discord",
                "deliver": "local",
                "provider": "openrouter",
                "model": "changed-after-claim",
            },
        ) is not None
        assert jobs.remove_job(job["id"]) is True

        assert delivery_queue.get_status(_alarm_key(execution_id)) is None
        assert _recover_dead_owner(executions, monkeypatch) == 1
        assert executions.drain_execution_recovery_outbox(delivery_queue.enqueue) == 1

    claimed_alarm = delivery_queue.claim_next()
    assert claimed_alarm is not None
    assert claimed_alarm["execution_id"] == _alarm_key(execution_id)
    assert claimed_alarm["content"] == RECOVERY_FOOTER
    assert claimed_alarm["job"] == {
        "id": job["id"],
        "name": "Reflection Alarm",
        "deliver": "origin",
        "failure_deliver": "telegram",
        "attach_to_session": True,
    }


def test_claim_fire_does_not_retroactively_opt_in_legacy_claim(
    wiring_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cron.jobs as jobs
    from cron.scheduler_provider import InProcessCronScheduler

    executions, delivery_queue, home = wiring_env
    with jobs.use_cron_store(home):
        legacy = _create_reflection_job(jobs, failure_alarm=None, failure_deliver="local")
        claimed = InProcessCronScheduler().claim_fire(legacy["id"], force=True)
        assert claimed is not None
        execution_id = str(claimed["execution_id"])

        assert jobs.update_job(
            legacy["id"], {"failure_alarm": "reflection", "failure_deliver": "telegram"}
        ) is not None

        assert _recover_dead_owner(executions, monkeypatch) == 1
        assert executions.drain_execution_recovery_outbox(delivery_queue.enqueue) == 0

    assert delivery_queue.get_status(_alarm_key(execution_id)) is None
    assert delivery_queue.claim_next() is None


@pytest.mark.parametrize(
    ("drain_action", "expected"),
    [
        pytest.param("startup", 1, id="provider-startup"),
        pytest.param("idle-tick", 0, id="idle-tick"),
    ],
)
def test_provider_startup_and_idle_tick_drain_pending_recovery_outbox_without_new_recovery(
    wiring_env,
    monkeypatch: pytest.MonkeyPatch,
    drain_action: str,
    expected: int,
) -> None:
    import cron.jobs as jobs
    import cron.scheduler as scheduler
    from cron.scheduler_provider import InProcessCronScheduler

    executions, delivery_queue, home = wiring_env
    execution_id = _pending_recovery_outbox(wiring_env, monkeypatch)
    assert executions.recover_interrupted_executions() == 0
    assert delivery_queue.get_status(_alarm_key(execution_id)) is None

    with jobs.use_cron_store(home):
        if drain_action == "startup":
            stop = threading.Event()
            stop.set()
            InProcessCronScheduler().start(stop, interval=0)
        else:
            assert scheduler.tick(verbose=False, sync=True) == expected

    queued = delivery_queue.get_status(_alarm_key(execution_id))
    assert queued is not None
    assert queued["status"] == "pending"
    assert queued["for_failure"] == 1
    claimed = delivery_queue.claim_next()
    assert claimed is not None
    assert claimed["execution_id"] == _alarm_key(execution_id)
    assert claimed["content"] == RECOVERY_FOOTER


def test_recovery_alarm_outbox_failure_rolls_back_unknown_transition_and_pending_delivery(
    wiring_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cron.jobs as jobs

    executions, delivery_queue, home = wiring_env
    with jobs.use_cron_store(home):
        job = _create_reflection_job(jobs)
        record = executions.create_execution(job["id"], source="direct")

        def fail_pending(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("outbox insert failed")

        monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
        monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
        monkeypatch.setattr(executions, "_mark_recovery_alarm_pending_unlocked", fail_pending)

        with pytest.raises(RuntimeError, match="outbox insert failed"):
            executions.recover_interrupted_executions()

        assert executions.get_execution(record["id"])["status"] == "claimed"
        assert executions.drain_execution_recovery_outbox(delivery_queue.enqueue) == 0

    assert delivery_queue.get_status(_alarm_key(record["id"])) is None
    assert delivery_queue.claim_next() is None
