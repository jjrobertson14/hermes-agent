"""RED tests for restart-durable reflection recovery alarms."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any
from unittest.mock import Mock

import pytest


def _require_recovery_api(executions: Any) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    snapshot = getattr(executions, "snapshot_execution_recovery_alarm", None)
    drain = getattr(executions, "drain_execution_recovery_outbox", None)
    acknowledge = getattr(executions, "acknowledge_execution_recovery_outbox", None)
    assert callable(snapshot), "cron.executions.snapshot_execution_recovery_alarm is public contract"
    assert callable(drain), "cron.executions.drain_execution_recovery_outbox is public contract"
    assert callable(acknowledge), "cron.executions.acknowledge_execution_recovery_outbox is public contract"
    return snapshot, drain, acknowledge


@pytest.fixture()
def recovery_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import cron.delivery_queue as delivery_queue
    import cron.executions as executions

    home = tmp_path / "profile"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", home / "cron" / "deliveries.db")
    return executions, delivery_queue, home


def _trusted_reflection_job(job_id: str = "reflection") -> dict[str, Any]:
    return {
        "id": job_id,
        "name": "Reflection Alarm",
        "failure_alarm": "reflection",
        "no_agent": False,
        "deliver": "origin",
        "failure_deliver": "telegram",
        "attach_to_session": True,
        "prompt": "unsafe prompt must not enter recovery alarm snapshot",
        "raw_exception": "secret raw exception must not enter snapshot",
        "config": {"api_key": "SECRET"},
    }


def _alarm_key(execution_id: str) -> str:
    return f"{execution_id}:reflection-alarm"


RECOVERY_FOOTER = (
    "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- cron runtime interrupted 🚨"
)


def test_public_recovery_outbox_contract_is_callable_and_create_execution_signature_is_actual(
    recovery_env,
) -> None:
    executions, _delivery_queue, _home = recovery_env

    _require_recovery_api(executions)
    signature = inspect.signature(executions.create_execution)
    assert list(signature.parameters)[:1] == ["job_id"]
    assert "source" in signature.parameters
    assert "scheduled_instant" in signature.parameters


def test_snapshot_keeps_live_owner_out_of_queue_then_dead_owner_recovery_enqueues_safe_alarm(
    recovery_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions, delivery_queue, _home = recovery_env
    snapshot, drain, _acknowledge = _require_recovery_api(executions)
    record = executions.create_execution("reflection", source="direct")

    # A normal execution delivery already terminalized under the raw execution id must not tombstone
    # the separate recovery alarm lane.
    delivery_queue.enqueue(record["id"], {"id": "old-normal"}, "old normal result")
    assert delivery_queue.claim_next()["execution_id"] == record["id"]
    assert delivery_queue._finish(record["id"], error=None)

    assert snapshot(
        record["id"],
        _trusted_reflection_job(),
        RECOVERY_FOOTER,
        for_failure=True,
    ) is True
    assert executions.get_execution(record["id"])["status"] == "claimed"
    assert drain(delivery_queue.enqueue) == 0
    assert delivery_queue.get_status(_alarm_key(record["id"])) is None

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
    assert executions.recover_interrupted_executions() == 1
    assert executions.get_execution(record["id"])["status"] == "unknown"
    assert executions.recover_interrupted_executions() == 0
    assert drain(delivery_queue.enqueue) == 1

    queued = delivery_queue.get_status(_alarm_key(record["id"]))
    assert queued is not None
    assert queued["status"] == "pending"
    assert queued["for_failure"] == 1
    assert delivery_queue.get_status(record["id"])["status"] == "delivered"
    claimed = delivery_queue.claim_next()
    assert claimed["execution_id"] == _alarm_key(record["id"])
    assert claimed["content"] == RECOVERY_FOOTER
    assert claimed["job"] == {
        "id": "reflection",
        "name": "Reflection Alarm",
        "deliver": "origin",
        "failure_deliver": "telegram",
        "attach_to_session": True,
    }
    serialized = str(claimed["job"]) + claimed["content"]
    assert "unsafe prompt" not in serialized
    assert "secret raw exception" not in serialized
    assert "SECRET" not in serialized


@pytest.mark.parametrize(
    ("job_update", "expected"),
    [
        pytest.param({"manual_run_at": "2026-10-07T00:00:00+00:00"}, True, id="manual-opt-in"),
        pytest.param({"no_agent": True}, False, id="explicit-no-agent"),
        pytest.param({"failure_alarm": None}, False, id="not-opted-in"),
    ],
)
def test_snapshot_eligibility_keeps_manual_opt_in_but_excludes_explicit_non_agent_cases(
    recovery_env,
    job_update: dict[str, Any],
    expected: bool,
) -> None:
    executions, delivery_queue, _home = recovery_env
    snapshot, drain, _acknowledge = _require_recovery_api(executions)
    job = _trusted_reflection_job("eligibility") | job_update
    source = "manual" if "manual_run_at" in job_update else "direct"
    record = executions.create_execution(job["id"], source=source)

    assert snapshot(record["id"], job, RECOVERY_FOOTER, for_failure=True) is expected
    assert drain(delivery_queue.enqueue) == 0
    assert delivery_queue.get_status(_alarm_key(record["id"])) is None


def test_snapshot_ignores_execution_finished_through_terminal_completion_api(
    recovery_env,
) -> None:
    executions, delivery_queue, _home = recovery_env
    snapshot, drain, _acknowledge = _require_recovery_api(executions)
    record = executions.create_execution("completed-reflection", source="direct")
    assert executions.finish_execution(record["id"], success=True) is not None

    assert snapshot(record["id"], _trusted_reflection_job(), RECOVERY_FOOTER, for_failure=True) is False
    assert drain(delivery_queue.enqueue) == 0
    assert delivery_queue.get_status(_alarm_key(record["id"])) is None


def test_dead_owner_recovery_snapshots_unknown_alarm_once_and_later_drain_survives_zero_recovery(
    recovery_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cron.jobs as jobs

    executions, delivery_queue, home = recovery_env
    _snapshot, drain, _acknowledge = _require_recovery_api(executions)
    with jobs.use_cron_store(home):
        claimed_job = jobs.create_job(
            prompt="reflect",
            schedule="every 1h",
            name="claimed reflection",
            deliver="origin",
            failure_alarm="reflection",
        )
        running_job = jobs.create_job(
            prompt="reflect",
            schedule="every 1h",
            name="running reflection",
            deliver="origin",
            failure_alarm="reflection",
        )
        claimed = executions.create_execution(claimed_job["id"], source="direct")
        running = executions.create_execution(running_job["id"], source="direct")
        assert executions.mark_execution_running(running["id"]) is not None

        monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
        monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)

        assert executions.recover_interrupted_executions() == 2
        assert executions.get_execution(claimed["id"])["status"] == "unknown"
        assert executions.get_execution(running["id"])["status"] == "unknown"
        assert executions.recover_interrupted_executions() == 0
        assert delivery_queue.get_status(_alarm_key(claimed["id"])) is None
        assert delivery_queue.get_status(_alarm_key(running["id"])) is None
        assert drain(delivery_queue.enqueue, limit=20) == 2
        assert drain(delivery_queue.enqueue, limit=20) == 0

    claimed_alarm = delivery_queue.get_status(_alarm_key(claimed["id"]))
    running_alarm = delivery_queue.get_status(_alarm_key(running["id"]))
    assert claimed_alarm is not None and claimed_alarm["status"] == "pending"
    assert running_alarm is not None and running_alarm["status"] == "pending"


def test_callback_crash_after_queue_insert_keeps_outbox_pending_for_idempotent_retry(
    recovery_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions, delivery_queue, _home = recovery_env
    snapshot, drain, _acknowledge = _require_recovery_api(executions)
    record = executions.create_execution("reflection", source="direct")
    assert snapshot(record["id"], _trusted_reflection_job(), RECOVERY_FOOTER) is True
    assert drain(delivery_queue.enqueue) == 0

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
    assert executions.recover_interrupted_executions() == 1
    assert executions.recover_interrupted_executions() == 0

    def enqueue_then_crash(execution_id: str, job: dict[str, Any], content: str, *, for_failure: bool = False):
        delivery_queue.enqueue(execution_id, job, content, for_failure=for_failure)
        raise RuntimeError("crash before outbox acknowledgment")

    with pytest.raises(RuntimeError, match="before outbox acknowledgment"):
        drain(enqueue_then_crash)
    assert delivery_queue.get_status(_alarm_key(record["id"]))["status"] == "pending"

    assert drain(delivery_queue.enqueue) == 1
    assert drain(delivery_queue.enqueue) == 0
    claimed = delivery_queue.claim_next()
    assert claimed["execution_id"] == _alarm_key(record["id"])
    assert claimed["content"] == RECOVERY_FOOTER
    assert delivery_queue.claim_next() is None


def test_transport_unknown_alarm_is_acknowledged_without_duplicate_retry(
    recovery_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions, delivery_queue, _home = recovery_env
    snapshot, drain, _acknowledge = _require_recovery_api(executions)
    record = executions.create_execution("reflection", source="direct")
    alarm_key = _alarm_key(record["id"])
    assert snapshot(record["id"], _trusted_reflection_job(), RECOVERY_FOOTER) is True
    assert drain(delivery_queue.enqueue) == 0

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
    assert executions.recover_interrupted_executions() == 1
    assert executions.recover_interrupted_executions() == 0

    delivery_queue.enqueue(alarm_key, {"id": "reflection"}, RECOVERY_FOOTER, for_failure=True)
    assert delivery_queue.claim_next()["execution_id"] == alarm_key
    monkeypatch.setattr(delivery_queue, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(delivery_queue, "_owner_is_live", lambda _pid, _started: False)
    assert delivery_queue.recover_abandoned() == 1
    assert delivery_queue.get_status(alarm_key)["status"] == "unknown"

    callback = Mock(side_effect=delivery_queue.enqueue)
    assert drain(callback) == 1
    assert drain(callback) == 0
    assert callback.call_count == 1
    assert delivery_queue.get_status(alarm_key)["status"] == "unknown"
    assert delivery_queue.claim_next() is None
