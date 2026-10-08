"""Post-CAS interruption recovery alarm authority tests."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import Any

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
def cas_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import cron.delivery_queue as delivery_queue
    import cron.executions as executions
    import cron.scheduler as scheduler

    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", home / "cron" / "deliveries.db")
    monkeypatch.setattr(scheduler, "run_one_job", lambda *_args, **_kwargs: True)
    yield executions, scheduler, home
    scheduler._running_job_ids.clear()
    scheduler._running_since.clear()
    scheduler._running_allowance_s.clear()
    scheduler._running_futures.clear()
    scheduler._running_worker_pids.clear()


def _create_reflection_job(jobs: Any, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "prompt": "reflect",
        "schedule": "every 1h",
        "name": "Preliminary Reflection Alarm",
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


def _recover_and_drain(executions: Any, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any], str, bool]]:
    captured: list[tuple[str, dict[str, Any], str, bool]] = []

    def capture(alarm_key: str, job: dict[str, Any], content: str, *, for_failure: bool = False) -> None:
        captured.append((alarm_key, job, content, for_failure))

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
    assert executions.recover_interrupted_executions() == 1
    assert executions.drain_execution_recovery_outbox(capture) == len(captured)
    return captured


def _raise_after_actual_claim(
    jobs: Any,
    monkeypatch: pytest.MonkeyPatch,
    target: Any,
    job_id: str,
    updates: dict[str, Any],
) -> None:
    original_claim = jobs.claim_job_for_fire

    def claim_after_mutation(claimed_job_id: str, *args: Any, **kwargs: Any) -> Any:
        assert claimed_job_id == job_id
        assert jobs.update_job(job_id, updates) is not None
        claimed = original_claim(claimed_job_id, *args, **kwargs)
        assert isinstance(claimed, dict), "actual durable winning claim must succeed before interruption"
        raise SystemExit("death after durable claim before recovery snapshot")

    monkeypatch.setattr(target, "claim_job_for_fire", claim_after_mutation)


def _interrupt_provider_claim(
    jobs: Any,
    monkeypatch: pytest.MonkeyPatch,
    job_id: str,
    updates: dict[str, Any],
) -> str:
    import cron.scheduler_provider as scheduler_provider
    from cron.executions import latest_execution

    _raise_after_actual_claim(jobs, monkeypatch, jobs, job_id, updates)
    with pytest.raises(SystemExit, match="death after durable claim"):
        scheduler_provider.InProcessCronScheduler().claim_fire(job_id, force=True)
    execution = latest_execution(job_id)
    assert execution is not None
    return str(execution["id"])


def _interrupt_builtin_claim(
    jobs: Any,
    scheduler: Any,
    monkeypatch: pytest.MonkeyPatch,
    job: dict[str, Any],
    updates: dict[str, Any],
) -> str:
    from cron.executions import latest_execution

    _raise_after_actual_claim(jobs, monkeypatch, scheduler, job["id"], updates)
    future = scheduler._submit_with_guard(
        job,
        _InlinePool(),
        lambda dispatched: scheduler._process_due_job(dispatched, None, None, False),
    )
    assert future is not None
    with pytest.raises(SystemExit, match="death after durable claim"):
        future.result()
    execution = latest_execution(job["id"])
    assert execution is not None
    return str(execution["id"])


@pytest.mark.parametrize(
    ("case", "initial_overrides", "actual_updates", "expected_job"),
    [
        pytest.param(
            "route-change",
            {},
            {
                "name": "Actual Claimed Reflection Alarm",
                "deliver": "local",
                "failure_deliver": "discord",
                "attach_to_session": False,
            },
            {
                "name": "Actual Claimed Reflection Alarm",
                "deliver": "local",
                "failure_deliver": "discord",
                "attach_to_session": False,
            },
            id="route-change",
        ),
        pytest.param(
            "actual-opted-out",
            {},
            {"failure_alarm": None},
            None,
            id="actual-opted-out",
        ),
        pytest.param(
            "actual-no-agent",
            {},
            {"no_agent": True, "script": "/bin/true"},
            None,
            id="actual-no-agent",
        ),
        pytest.param(
            "previous-unopted-to-actual-opted-in",
            {"failure_alarm": None},
            {
                "failure_alarm": "reflection",
                "name": "Actual Opted In Reflection Alarm",
                "failure_deliver": "slack",
            },
            {
                "name": "Actual Opted In Reflection Alarm",
                "deliver": "origin",
                "failure_deliver": "slack",
                "attach_to_session": True,
            },
            id="previous-unopted-to-actual-opted-in",
        ),
    ],
)
@pytest.mark.parametrize("claim_route", ["provider", "builtin"])
def test_post_cas_interruption_uses_actual_claim_authority(
    cas_env,
    monkeypatch: pytest.MonkeyPatch,
    claim_route: str,
    case: str,
    initial_overrides: dict[str, Any],
    actual_updates: dict[str, Any],
    expected_job: dict[str, Any] | None,
) -> None:
    import cron.jobs as jobs

    executions, scheduler, home = cas_env
    with jobs.use_cron_store(home):
        job = _create_reflection_job(jobs, **initial_overrides)
        if claim_route == "provider":
            execution_id = _interrupt_provider_claim(jobs, monkeypatch, job["id"], actual_updates)
        else:
            execution_id = _interrupt_builtin_claim(jobs, scheduler, monkeypatch, job, actual_updates)

        recovered = executions.get_execution(execution_id)
        assert recovered is not None
        captured = _recover_and_drain(executions, monkeypatch)

    assert executions.get_execution(execution_id)["status"] == "unknown"
    if expected_job is None:
        assert captured == [], f"{case} should stay quiet after actual ineligible claim"
    else:
        assert captured == [
            (
                f"{execution_id}:reflection-alarm",
                {"id": job["id"], **expected_job},
                RECOVERY_FOOTER,
                True,
            )
        ]
