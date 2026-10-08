"""Built-in cron claim-boundary recovery alarm regressions."""

from __future__ import annotations

import concurrent.futures
from typing import Any, Callable

import pytest


RECOVERY_FOOTER = (
    "\U0001f6a8 REFLECTION JOB FAILED TO BE COMPLETED -- cron runtime interrupted \U0001f6a8"
)


class _InlinePool:
    def submit(self, callback: Callable[[], Any]) -> concurrent.futures.Future[Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            future.set_result(callback())
        except BaseException as exc:
            future.set_exception(exc)
        return future


@pytest.fixture()
def builtin_claim_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import cron.delivery_queue as delivery_queue
    import cron.executions as executions
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", home / "cron" / "deliveries.db")
    monkeypatch.setattr(scheduler, "run_one_job", lambda *_args, **_kwargs: True)
    yield executions, delivery_queue, scheduler, home
    scheduler._running_job_ids.clear()
    scheduler._running_since.clear()
    scheduler._running_allowance_s.clear()
    scheduler._running_futures.clear()
    scheduler._running_worker_pids.clear()


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


def _run_builtin_fire_with_actual_claim_mutation(
    jobs: Any,
    scheduler: Any,
    monkeypatch: pytest.MonkeyPatch,
    job: dict[str, Any],
    updates: dict[str, Any],
) -> str:
    original_claim = jobs.claim_job_for_fire

    def claim_after_mutation(job_id: str, *args: Any, **kwargs: Any) -> Any:
        assert job_id == job["id"]
        assert jobs.update_job(job_id, updates) is not None
        return original_claim(job_id, *args, **kwargs)

    monkeypatch.setattr(scheduler, "claim_job_for_fire", claim_after_mutation)
    future = scheduler._submit_with_guard(
        job,
        _InlinePool(),
        lambda dispatched: scheduler._process_due_job(dispatched, None, None, False),
    )
    assert future is not None
    assert future.result() is True
    from cron.executions import latest_execution

    execution = latest_execution(job["id"])
    assert execution is not None
    return str(execution["id"])


def _recover_dead_owner(executions: Any, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
    return int(executions.recover_interrupted_executions())


@pytest.mark.parametrize(
    ("actual_claim_updates", "expected_alarm_job"),
    [
        pytest.param(
            {
                "name": "Changed Builtin Reflection Alarm",
                "deliver": "local",
                "failure_deliver": "discord",
                "attach_to_session": False,
            },
            {
                "name": "Changed Builtin Reflection Alarm",
                "deliver": "local",
                "failure_deliver": "discord",
                "attach_to_session": False,
            },
            id="actual-claim-route",
        ),
        pytest.param({"failure_alarm": None}, None, id="actual-claim-opted-out"),
        pytest.param(
            {"no_agent": True, "script": "/bin/true"},
            None,
            id="actual-claim-no-agent",
        ),
    ],
)
def test_builtin_recovery_alarm_uses_actual_claim_authority(
    builtin_claim_env,
    monkeypatch: pytest.MonkeyPatch,
    actual_claim_updates: dict[str, Any],
    expected_alarm_job: dict[str, Any] | None,
) -> None:
    import cron.jobs as jobs

    executions, delivery_queue, scheduler, home = builtin_claim_env
    with jobs.use_cron_store(home):
        job = _create_reflection_job(jobs)
        execution_id = _run_builtin_fire_with_actual_claim_mutation(
            jobs, scheduler, monkeypatch, job, actual_claim_updates
        )
        assert delivery_queue.get_status(_alarm_key(execution_id)) is None
        assert _recover_dead_owner(executions, monkeypatch) == 1
        expected_drains = 1 if expected_alarm_job is not None else 0
        assert executions.drain_execution_recovery_outbox(delivery_queue.enqueue) == expected_drains

    claimed_alarm = delivery_queue.claim_next()
    if expected_alarm_job is None:
        assert claimed_alarm is None
        assert delivery_queue.get_status(_alarm_key(execution_id)) is None
    else:
        assert claimed_alarm is not None
        assert claimed_alarm["execution_id"] == _alarm_key(execution_id)
        assert claimed_alarm["content"] == RECOVERY_FOOTER
        assert claimed_alarm["job"] == {"id": job["id"], **expected_alarm_job}
