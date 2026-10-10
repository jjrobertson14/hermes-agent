from contextlib import nullcontext

from cron import job_definition


def test_import_job_definitions_merges_string_local_ids(monkeypatch):
    saved_jobs = []
    schedule = {"kind": "interval", "minutes": 30, "display": "every 30m"}
    local_jobs = [
        {
            "id": "daily",
            "name": "old name",
            "prompt": "old prompt",
            "schedule": schedule,
            "enabled": False,
            "state": "paused",
            "next_run_at": None,
            "repeat": {"completed": 2, "times": 5},
        }
    ]
    shipped = {
        "daily": {
            "name": "new name",
            "prompt": "new prompt",
            "schedule": schedule,
            "repeat": {"times": 7},
        }
    }

    monkeypatch.setattr(job_definition, "_jobs_lock", nullcontext)
    monkeypatch.setattr(job_definition, "load_jobs", lambda: list(local_jobs))
    monkeypatch.setattr(job_definition, "save_jobs", saved_jobs.append)

    job_definition.import_job_definitions(shipped, paused_reason="installed from profile")

    assert len(saved_jobs) == 1
    [merged] = saved_jobs[0]
    assert merged["id"] == "daily"
    assert merged["name"] == "new name"
    assert merged["prompt"] == "new prompt"
    assert merged["next_run_at"] is None
    assert merged["repeat"] == {"completed": 2, "times": 7}


def test_import_job_definitions_keeps_malformed_local_ids_separate(monkeypatch):
    saved_jobs = []
    schedule = {"kind": "interval", "minutes": 15, "display": "every 15m"}
    local_jobs = [{"id": None, "name": "local", "schedule": schedule}]
    shipped = {"incoming": {"name": "incoming", "prompt": "run", "schedule": schedule}}

    monkeypatch.setattr(job_definition, "_jobs_lock", nullcontext)
    monkeypatch.setattr(job_definition, "load_jobs", lambda: list(local_jobs))
    monkeypatch.setattr(job_definition, "save_jobs", saved_jobs.append)

    job_definition.import_job_definitions(shipped, paused_reason="installed from profile")

    assert len(saved_jobs) == 1
    assert saved_jobs[0][0] == local_jobs[0]
    assert saved_jobs[0][1]["id"] == "incoming"
    assert saved_jobs[0][1]["name"] == "incoming"


def test_import_job_definitions_keeps_unhashable_malformed_local_ids_separate(monkeypatch):
    saved_jobs = []
    schedule = {"kind": "interval", "minutes": 15, "display": "every 15m"}
    local_jobs = [{"id": ["incoming"], "name": "local", "schedule": schedule}]
    shipped = {"incoming": {"name": "incoming", "prompt": "run", "schedule": schedule}}

    monkeypatch.setattr(job_definition, "_jobs_lock", nullcontext)
    monkeypatch.setattr(job_definition, "load_jobs", lambda: list(local_jobs))
    monkeypatch.setattr(job_definition, "save_jobs", saved_jobs.append)

    job_definition.import_job_definitions(shipped, paused_reason="installed from profile")

    assert len(saved_jobs) == 1
    assert saved_jobs[0][0] == local_jobs[0]
    assert saved_jobs[0][1]["id"] == "incoming"
    assert saved_jobs[0][1]["name"] == "incoming"
