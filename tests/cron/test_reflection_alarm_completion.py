"""Completion/no-work boundaries for reflection runtime alarms."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

import cron.scheduler as scheduler


class _RunState(TypedDict):
    delivered: list[tuple[dict[str, Any], str, bool]]
    marked: list[tuple[tuple[Any, ...], dict[str, Any]]]
    finished: list[tuple[tuple[Any, ...], dict[str, Any]]]


def _patch_run_one_job_side_effects(monkeypatch: pytest.MonkeyPatch) -> _RunState:
    state: _RunState = {"delivered": [], "marked": [], "finished": []}

    def fake_deliver_result(
        job: dict[str, Any],
        content: str,
        **kwargs: Any,
    ) -> None:
        state["delivered"].append((job, content, bool(kwargs.get("for_failure"))))
        return None

    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda _job: False)
    monkeypatch.setattr(
        scheduler,
        "create_execution",
        lambda job_id, **_kwargs: {"id": f"exec-{job_id}"},
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _execution_id: {})
    monkeypatch.setattr(
        scheduler,
        "save_job_output",
        lambda job_id, _output: f"/tmp/{job_id}.txt",
    )
    monkeypatch.setattr(scheduler, "_deliver_result", fake_deliver_result)
    monkeypatch.setattr(
        scheduler,
        "mark_job_run",
        lambda *args, **kwargs: state["marked"].append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda *args, **kwargs: state["finished"].append((args, kwargs)),
    )
    monkeypatch.setattr(scheduler, "_upsert_incident_for_failure", lambda *_a, **_kw: (False, None))
    monkeypatch.setattr(scheduler, "load_config", lambda: {})
    return state


def _last_nonblank_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines, "delivery content must contain a nonblank alarm footer"
    return lines[-1]


@pytest.mark.parametrize(
    ("script_output", "expected_status"),
    [
        pytest.param("", "silent (empty output)", id="collect-only-noop-empty-stdout"),
        pytest.param('{"wakeAgent": false}\n', "silent (wakeAgent=false)", id="explicit-wake-false"),
    ],
)
def test_reflection_alarm_no_agent_completed_no_work_stays_quiet(
    monkeypatch: pytest.MonkeyPatch,
    script_output: str,
    expected_status: str,
) -> None:
    """A successful script-only collector no-op is complete work, not an agent-empty failure."""
    state = _patch_run_one_job_side_effects(monkeypatch)
    monkeypatch.setattr(
        scheduler,
        "_run_job_script_with_claim_heartbeat",
        lambda *_args, **_kwargs: (True, script_output),
    )
    job = {
        "id": "reflection-collector-no-work",
        "name": "Reflection Collector No Work",
        "script": "/repos/Meta/hermes-reflection-cron-foxtop-collect-fix-wt/scripts/reflection_cron_launcher.py",
        "no_agent": True,
        "deliver": "telegram",
        "failure_alarm": "reflection",
    }

    scheduler.run_one_job(job)

    assert state["delivered"] == []
    assert state["marked"]
    mark_args, mark_kwargs = state["marked"][-1]
    assert mark_args == ("reflection-collector-no-work", True, None)
    assert mark_kwargs.get("delivery_error") is None
    assert state["finished"]
    finish_args, finish_kwargs = state["finished"][-1]
    assert finish_args == ("exec-reflection-collector-no-work",)
    assert finish_kwargs["success"] is True
    assert finish_kwargs["delivery_outcome"] == "suppressed"

    success, output, final_response, error = scheduler.run_job(job)
    assert success is True
    assert expected_status in output
    assert final_response == scheduler.SILENT_MARKER
    assert error is None


def test_reflection_alarm_plain_collector_signal_failure_still_alarms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _patch_run_one_job_side_effects(monkeypatch)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (
            False,
            "collector produced a runtime signal",
            "",
            "REFLECTION-CRON-SIGNAL: collector did not complete required stages",
        ),
    )
    job = {
        "id": "reflection-collector-signal",
        "name": "Reflection Collector Signal",
        "deliver": "telegram",
        "failure_alarm": "reflection",
    }

    scheduler.run_one_job(job)

    assert len(state["delivered"]) == 1
    _delivered_job, content, for_failure = state["delivered"][0]
    assert for_failure is True
    assert _last_nonblank_line(content) == (
        "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- cron runtime reported failure 🚨"
    )
    assert "REFLECTION-CRON-SIGNAL" not in content
    mark_args, _mark_kwargs = state["marked"][-1]
    assert mark_args[0] == "reflection-collector-signal"
    assert mark_args[1] is False


def test_reflection_alarm_agent_empty_response_still_alarms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _patch_run_one_job_side_effects(monkeypatch)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (True, "raw output", "", None),
    )
    job = {
        "id": "reflection-agent-empty",
        "name": "Reflection Agent Empty",
        "deliver": "telegram",
        "failure_alarm": "reflection",
    }

    scheduler.run_one_job(job)

    assert len(state["delivered"]) == 1
    _delivered_job, content, for_failure = state["delivered"][0]
    assert for_failure is True
    assert _last_nonblank_line(content) == (
        "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- agent returned empty final response 🚨"
    )
    mark_args, _mark_kwargs = state["marked"][-1]
    assert mark_args[0] == "reflection-agent-empty"
    assert mark_args[1] is False
