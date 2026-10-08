"""Behavioral RED tests for recovery alarm metadata at the durable claim boundary."""

from __future__ import annotations

from typing import Any

import pytest


RECOVERY_FOOTER = (
    "\U0001f6a8 REFLECTION JOB FAILED TO BE COMPLETED -- cron runtime interrupted \U0001f6a8"
)


@pytest.fixture()
def claim_race_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
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


def _claim_with_actual_job_mutation(
    jobs: Any,
    monkeypatch: pytest.MonkeyPatch,
    job_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    from cron.scheduler_provider import InProcessCronScheduler

    original_claim = jobs.claim_job_for_fire

    def claim_after_mutation(claimed_job_id: str, *args: Any, **kwargs: Any) -> Any:
        assert claimed_job_id == job_id
        assert jobs.update_job(job_id, updates) is not None
        return original_claim(claimed_job_id, *args, **kwargs)

    monkeypatch.setattr(jobs, "claim_job_for_fire", claim_after_mutation)
    claimed = InProcessCronScheduler().claim_fire(job_id, force=True)
    assert claimed is not None
    return claimed


def test_claim_fire_recovery_alarm_uses_actual_claimed_route_when_job_changes_before_claim(
    claim_race_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cron.jobs as jobs

    executions, delivery_queue, home = claim_race_env
    with jobs.use_cron_store(home):
        job = _create_reflection_job(jobs)
        claimed = _claim_with_actual_job_mutation(
            jobs,
            monkeypatch,
            job["id"],
            {
                "name": "Changed Claimed Reflection Alarm",
                "deliver": "local",
                "failure_deliver": "discord",
                "attach_to_session": False,
                "provider": "openrouter",
                "model": "changed-before-claim",
            },
        )
        execution_id = str(claimed["execution_id"])

        assert delivery_queue.get_status(_alarm_key(execution_id)) is None
        assert _recover_dead_owner(executions, monkeypatch) == 1
        assert executions.drain_execution_recovery_outbox(delivery_queue.enqueue) == 1

    claimed_alarm = delivery_queue.claim_next()
    assert claimed_alarm is not None
    assert claimed_alarm["execution_id"] == _alarm_key(execution_id)
    assert claimed_alarm["content"] == RECOVERY_FOOTER
    assert claimed_alarm["job"] == {
        "id": job["id"],
        "name": "Changed Claimed Reflection Alarm",
        "deliver": "local",
        "failure_deliver": "discord",
        "attach_to_session": False,
    }


@pytest.mark.parametrize(
    "actual_claim_updates",
    [
        pytest.param({"failure_alarm": None}, id="actual-claim-opted-out"),
        pytest.param(
            {"no_agent": True, "script": "/bin/true"},
            id="actual-claim-no-agent",
        ),
    ],
)
def test_claim_fire_recovery_alarm_stays_quiet_when_actual_claim_is_not_eligible(
    claim_race_env,
    monkeypatch: pytest.MonkeyPatch,
    actual_claim_updates: dict[str, Any],
) -> None:
    import cron.jobs as jobs

    executions, delivery_queue, home = claim_race_env
    with jobs.use_cron_store(home):
        job = _create_reflection_job(jobs)
        claimed = _claim_with_actual_job_mutation(
            jobs, monkeypatch, job["id"], actual_claim_updates
        )
        execution_id = str(claimed["execution_id"])

        assert _recover_dead_owner(executions, monkeypatch) == 1
        assert executions.drain_execution_recovery_outbox(delivery_queue.enqueue) == 0

    assert delivery_queue.get_status(_alarm_key(execution_id)) is None
    assert delivery_queue.claim_next() is None
