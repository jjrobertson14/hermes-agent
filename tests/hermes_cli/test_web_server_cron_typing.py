"""Typing-focused behavior tests for dashboard cron helpers."""

import hermes_cli.web_server_cron as _web_server_cron


def test_find_cron_job_profile_ignores_non_list_profile_jobs(monkeypatch):
    calls = []

    monkeypatch.setattr(
        _web_server_cron,
        "_cron_profile_dicts",
        lambda: [{"name": "broken"}, {"name": "owner"}],
    )

    def fake_call_cron_for_profile(profile, func_name, include_disabled):
        calls.append((profile, func_name, include_disabled))
        if profile == "broken":
            return "not-a-job-list"
        return [{"id": "target-job", "name": "Target"}]

    monkeypatch.setattr(
        _web_server_cron,
        "_call_cron_for_profile",
        fake_call_cron_for_profile,
    )

    assert _web_server_cron._find_cron_job_profile("target-job") == "owner"
    assert calls == [
        ("broken", "list_jobs", True),
        ("owner", "list_jobs", True),
    ]


def test_find_cron_job_profile_ignores_non_dict_job_entries(monkeypatch):
    monkeypatch.setattr(
        _web_server_cron,
        "_cron_profile_dicts",
        lambda: [{"name": "owner"}],
    )
    monkeypatch.setattr(
        _web_server_cron,
        "_call_cron_for_profile",
        lambda profile, func_name, include_disabled: ["bad-entry", {"name": "target-name"}],
    )

    assert _web_server_cron._find_cron_job_profile("target-name") == "owner"
