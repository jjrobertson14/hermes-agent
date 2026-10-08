"""Behavioral RED tests for reflection runtime empty/sentinel alarms."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

import cron.scheduler as scheduler


class _RunState(TypedDict):
    delivered: list[tuple[dict[str, Any], str, bool]]
    marked: list[tuple[tuple[Any, ...], dict[str, Any]]]
    finished: list[tuple[tuple[Any, ...], dict[str, Any]]]


def _patch_run_one_job_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_response: str | None,
) -> _RunState:
    state: _RunState = {"delivered": [], "marked": [], "finished": []}

    def fake_run_job(job: dict[str, Any], **_kwargs: Any) -> tuple[bool, str, str | None, None]:
        return True, "raw reflection output", final_response, None

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
    monkeypatch.setattr(scheduler, "run_job", fake_run_job)
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


def _patch_run_one_job_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_job_impl: Any,
    delivery_raises: bool = False,
) -> _RunState:
    state: _RunState = {"delivered": [], "marked": [], "finished": []}

    def fake_deliver_result(
        job: dict[str, Any],
        content: str,
        **kwargs: Any,
    ) -> None:
        state["delivered"].append((job, content, bool(kwargs.get("for_failure"))))
        if delivery_raises:
            raise RuntimeError("post-delivery transport failure")
        return None

    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda _job: False)
    monkeypatch.setattr(
        scheduler,
        "create_execution",
        lambda job_id, **_kwargs: {"id": f"exec-{job_id}"},
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda _execution_id: {})
    monkeypatch.setattr(scheduler, "run_job", run_job_impl)
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


def _raise_runtime_error(*_args: Any, **_kwargs: Any) -> tuple[bool, str, str, str | None]:
    raise RuntimeError("provider exploded\nRAW_INTERNAL_DETAIL\nMALICIOUS_SECOND_LINE")


@pytest.mark.parametrize(
    ("run_job_impl", "expected_footer", "delivery_raises"),
    [
        pytest.param(
            lambda *_args, **_kwargs: (
                False,
                "raw failure output",
                "",
                "RuntimeError: provider failed\nRAW_INTERNAL_DETAIL\nMALICIOUS_SECOND_LINE",
            ),
            "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- cron runtime reported failure 🚨",
            False,
            id="ordinary-run-job-false-result",
        ),
        pytest.param(
            lambda *_args, **_kwargs: (
                False,
                "raw timeout output",
                "",
                "RuntimeError: Job 'reflection' idle for 601s (limit 600s)\nRAW_INTERNAL_DETAIL",
            ),
            "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- cron runtime timed out 🚨",
            False,
            id="inactivity-timeout-failure",
        ),
        pytest.param(
            _raise_runtime_error,
            "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- cron runtime crashed 🚨",
            True,
            id="runtime-error-crash-post-delivery-error",
        ),
        pytest.param(
            lambda *_args, **_kwargs: (
                True,
                "raw agent-declared output",
                "[CRON_FAILURE]\nChild reflection worker timed out.\nRAW_INTERNAL_DETAIL",
                None,
            ),
            "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- agent declared failure 🚨",
            False,
            id="agent-declared-cron-failure",
        ),
    ],
)
def test_reflection_alarm_opt_in_delivers_runtime_failure_footer(
    monkeypatch: pytest.MonkeyPatch,
    run_job_impl: Any,
    expected_footer: str,
    delivery_raises: bool,
) -> None:
    state = _patch_run_one_job_runtime_side_effects(
        monkeypatch,
        run_job_impl=run_job_impl,
        delivery_raises=delivery_raises,
    )
    job = {
        "id": "reflection-runtime-failure",
        "name": "Reflection Runtime Alarm",
        "deliver": "origin",
        "failure_alarm": "reflection",
    }

    scheduler.run_one_job(job)

    assert len(state["delivered"]) == 1
    delivered_job, content, for_failure = state["delivered"][0]
    assert delivered_job is job
    assert delivered_job["deliver"] == "origin"
    assert for_failure is True
    assert _last_nonblank_line(content) == expected_footer
    assert "RAW_INTERNAL_DETAIL" not in content
    assert "MALICIOUS_SECOND_LINE" not in content
    assert content.count("🚨 REFLECTION JOB FAILED TO BE COMPLETED -- ") == 1
    assert state["marked"]
    mark_args, _mark_kwargs = state["marked"][-1]
    assert mark_args[0] == "reflection-runtime-failure"
    assert mark_args[1] is False


def test_internal_reflection_prefix_without_alarm_opt_in_keeps_legacy_failure_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _patch_run_one_job_runtime_side_effects(
        monkeypatch,
        run_job_impl=lambda *_args, **_kwargs: (
            False,
            "raw failure output",
            "",
            "[reflection_runtime_alarm] legacy internal-looking error",
        ),
    )
    job = {"id": "legacy-runtime-prefix", "name": "Legacy", "deliver": "telegram"}

    scheduler.run_one_job(job)

    assert len(state["delivered"]) == 1
    _delivered_job, content, for_failure = state["delivered"][0]
    assert for_failure is True
    assert _last_nonblank_line(content) != (
        "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- legacy internal-looking error 🚨"
    )
    assert content.startswith("⚠️ Cron 'Legacy' failed:")


def test_reflection_alarm_successful_nonempty_all_rejected_report_is_not_failure_alarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_response = "All candidate changes were rejected explicitly; no code changes are needed."
    state = _patch_run_one_job_side_effects(monkeypatch, final_response=final_response)
    job = {
        "id": "reflection-all-rejected",
        "name": "Reflection Runtime Alarm",
        "deliver": "telegram",
        "failure_alarm": "reflection",
    }

    scheduler.run_one_job(job)

    assert len(state["delivered"]) == 1
    _delivered_job, content, for_failure = state["delivered"][0]
    assert for_failure is False
    assert _last_nonblank_line(content) == final_response
    assert "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- " not in content
    assert state["marked"]
    mark_args, mark_kwargs = state["marked"][-1]
    assert mark_args == ("reflection-all-rejected", True, None)
    assert mark_kwargs.get("delivery_error") is None


@pytest.mark.parametrize(
    "final_response",
    [None, "", "   \n\t  ", "[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"],
)
def test_reflection_alarm_opt_in_delivers_failure_for_empty_and_sentinel_outputs(
    monkeypatch: pytest.MonkeyPatch,
    final_response: str | None,
) -> None:
    """Opt-in reflection jobs must not book empty/sentinel output as healthy silence."""
    state = _patch_run_one_job_side_effects(monkeypatch, final_response=final_response)
    job = {
        "id": "reflection-empty",
        "name": "Reflection Runtime Alarm",
        "deliver": "telegram",
        "failure_alarm": "reflection",
    }

    scheduler.run_one_job(job)

    assert len(state["delivered"]) == 1
    delivered_job, content, for_failure = state["delivered"][0]
    assert delivered_job is job
    assert for_failure is True
    footer = _last_nonblank_line(content)
    assert footer.startswith("🚨 REFLECTION JOB FAILED TO BE COMPLETED -- ")
    assert footer.endswith(" 🚨")
    reason = footer.removeprefix(
        "🚨 REFLECTION JOB FAILED TO BE COMPLETED -- "
    ).removesuffix(" 🚨")
    assert reason.strip()
    assert reason == reason.strip()
    assert "\n" not in reason
    assert state["marked"]
    mark_args, mark_kwargs = state["marked"][-1]
    assert mark_args[0] == "reflection-empty"
    assert mark_args[1] is False
    assert mark_args[2]
    assert mark_kwargs.get("delivery_error") is None


def test_legacy_silent_response_without_reflection_alarm_remains_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent opt-in, the existing [SILENT] suppression behavior stays unchanged."""
    state = _patch_run_one_job_side_effects(monkeypatch, final_response="[SILENT]")

    scheduler.run_one_job({"id": "legacy-silent", "name": "Legacy", "deliver": "telegram"})

    assert state["delivered"] == []
    assert state["marked"]
    mark_args, mark_kwargs = state["marked"][-1]
    assert mark_args == ("legacy-silent", True, None)
    assert mark_kwargs.get("delivery_error") is None
