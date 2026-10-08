"""Behavioral boundaries for additive reflection failure-alarm job metadata."""

from __future__ import annotations

import argparse
import json
from typing import Any, cast

import pytest
from fastapi import HTTPException

from cron import jobs
from cron.job_definition import import_job_definitions
from hermes_cli.subcommands.cron import build_cron_parser
from hermes_cli.web_models import CronJobCreate, CronJobUpdate


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    cron_dir = hermes_home / "cron"
    output_dir = cron_dir / "output"
    output_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(jobs, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", output_dir)
    return hermes_home


def _build_cron_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_cron_parser(subparsers, cmd_cron=lambda _args: None)
    return parser


def test_import_job_definitions_preserves_reflection_failure_alarm(cron_env) -> None:
    import_job_definitions(
        {
            "ref-alarm": {
                "name": "reflection alarm",
                "prompt": "reflect",
                "schedule": "every 1h",
                "deliver": "local",
                "failure_alarm": "reflection",
            }
        },
        paused_reason="distribution import",
    )

    imported = jobs.get_job("ref-alarm")
    assert imported is not None
    assert imported["failure_alarm"] == "reflection"


def test_cronjob_tool_create_update_accepts_reflection_failure_alarm(cron_env) -> None:
    from tools.cronjob_tools import cronjob

    created = json.loads(cast(Any, cronjob)(
        action="create",
        prompt="reflect",
        schedule="every 1h",
        failure_alarm="reflection",
    ))

    assert created["success"] is True
    assert jobs.get_job(created["job_id"])["failure_alarm"] == "reflection"  # type: ignore[index]

    updated = json.loads(cast(Any, cronjob)(
        action="update",
        job_id=created["job_id"],
        failure_alarm="",
    ))

    assert updated["success"] is True
    assert "failure_alarm" not in jobs.get_job(created["job_id"])  # type: ignore[operator]


def test_cli_cron_create_and_edit_accept_failure_alarm(cron_env) -> None:
    from hermes_cli.cron import cron_command

    parser = _build_cron_parser()
    created_args = parser.parse_args([
        "cron",
        "create",
        "every 1h",
        "reflect",
        "--failure-alarm",
        "reflection",
    ])

    cron_command(created_args)

    created = jobs.list_jobs(include_disabled=True)[0]
    assert created["failure_alarm"] == "reflection"

    edited_args = parser.parse_args([
        "cron",
        "edit",
        created["id"],
        "--failure-alarm",
        "",
    ])
    cron_command(edited_args)

    assert "failure_alarm" not in jobs.get_job(created["id"])  # type: ignore[operator]


def test_web_create_dto_exposes_and_forwards_failure_alarm(tmp_path, monkeypatch) -> None:
    from hermes_cli import web_server_cron

    body = cast(Any, CronJobCreate)(
        prompt="reflect",
        schedule="every 1h",
        deliver="local",
        failure_alarm="reflection",
    )

    assert cast(Any, body).failure_alarm == "reflection"

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(web_server_cron, "_cron_profile_home", lambda _profile: ("default", tmp_path))
    monkeypatch.setattr(web_server_cron, "_validate_dashboard_cron_context_from", lambda *_args: None)
    monkeypatch.setattr(web_server_cron, "_validate_dashboard_cron_effective_job", lambda *_args: None)

    def fake_mutate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"id": "job-web-create", **kwargs}

    monkeypatch.setattr(web_server_cron, "_mutate_cron_for_profile", fake_mutate)

    out = cast(dict[str, Any], web_server_cron._create_cron_job_sync(body))

    assert out["failure_alarm"] == "reflection"
    assert calls[0][1]["failure_alarm"] == "reflection"


def test_web_update_forwards_valid_and_rejects_invalid_failure_alarm(tmp_path, monkeypatch) -> None:
    from hermes_cli.web_routers import cron as cron_router

    updates_seen: list[dict[str, Any]] = []

    monkeypatch.setattr(cron_router, "_job_profile", lambda _job_id, _profile: "default")
    monkeypatch.setattr(cron_router, "_cron_profile_home", lambda _profile: ("default", tmp_path))
    monkeypatch.setattr(
        cron_router,
        "_call_cron_for_profile",
        lambda _profile, func, *_args: {"id": "job-web-update", "prompt": "reflect"}
        if func == "get_job" else None,
    )

    def fake_mutate(_profile: str, _func: str, _job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        if updates.get("failure_alarm") not in {None, "reflection"}:
            raise ValueError("Invalid failure_alarm value: use 'reflection' or leave it blank to clear.")
        updates_seen.append(dict(updates))
        return {"id": "job-web-update", **updates}

    monkeypatch.setattr(cron_router, "_mutate_cron_for_profile", fake_mutate)

    valid = cast(dict[str, Any], cron_router._update_cron_job_sync(
        "job-web-update", CronJobUpdate(updates={"failure_alarm": "reflection"})
    ))

    assert valid["failure_alarm"] == "reflection"
    assert updates_seen == [{"failure_alarm": "reflection"}]

    with pytest.raises(HTTPException) as exc_info:
        cron_router._update_cron_job_sync(
            "job-web-update", CronJobUpdate(updates={"failure_alarm": "email"})
        )

    assert exc_info.value.status_code == 400
    assert updates_seen == [{"failure_alarm": "reflection"}]
