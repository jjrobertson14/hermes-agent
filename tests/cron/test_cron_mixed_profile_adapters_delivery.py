"""Regression coverage for mixed own/profile-routed cron delivery adapters."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import hermes_yaml as yaml
from pytest import MonkeyPatch

from cron.scheduler_delivery import _resolve_target_transport
from gateway.config import Platform, PlatformConfig


def _fake_config(*platforms: Platform) -> object:
    return type(
        "FakeGatewayConfig",
        (),
        {"platforms": {platform: PlatformConfig(enabled=True) for platform in platforms}},
    )()


def _capture_secondary_tick_adapters(
    tmp_path: Any,
    monkeypatch: MonkeyPatch,
    *,
    secondary_adapters: dict[Platform, object],
    primary_adapters: dict[Platform, object],
) -> object:
    from cron.scheduler_provider import InProcessCronScheduler

    root = tmp_path / "root"
    default_home = root
    fennec_home = root / "profiles" / "fennec"
    for home in (default_home, fennec_home):
        (home / "cron").mkdir(parents=True)
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "gateway": {
                    "multiplex_profiles": True,
                    "profile_routes": [
                        {
                            "name": "fennec-telegram-topic",
                            "platform": "telegram",
                            "chat_id": "123456789",
                            "thread_id": "222333",
                            "profile": "fennec",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)

    captured: list[object] = []
    stop = threading.Event()

    def _tick(*args: object, **kwargs: object) -> int:
        captured.append(kwargs.get("adapters"))
        if len(captured) >= 2:
            stop.set()
        return 0

    with (
        patch("cron.scheduler.tick", side_effect=_tick),
        patch("cron.jobs.record_ticker_heartbeat", lambda **kwargs: None),
    ):
        thread = threading.Thread(
            target=InProcessCronScheduler().start,
            args=(stop,),
            kwargs={
                "interval": 0,
                "profile_homes": [("default", default_home), ("fennec", fennec_home)],
                "adapters": primary_adapters,
                "profile_adapters": {"fennec": secondary_adapters},
                "default_profile": "default",
            },
            daemon=True,
        )
        thread.start()
        deadline = time.monotonic() + 10
        while len(captured) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(captured) >= 2
    return captured[1]


def test_secondary_with_unrelated_own_adapter_still_uses_exact_routed_primary_target(
    tmp_path: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    primary_telegram = object()
    own_discord = object()
    adapters = _capture_secondary_tick_adapters(
        tmp_path,
        monkeypatch,
        secondary_adapters={Platform.DISCORD: own_discord},
        primary_adapters={Platform.TELEGRAM: primary_telegram},
    )
    config = _fake_config(Platform.DISCORD)

    own_resolved, own_error = _resolve_target_transport(
        {"id": "own"},
        Platform.DISCORD,
        "discord",
        {"platform": "discord", "chat_id": "secondary-owned"},
        adapters,
        config,
    )
    assert own_error is None
    assert own_resolved is not None
    assert own_resolved[2] is own_discord

    routed_resolved, routed_error = _resolve_target_transport(
        {"id": "routed"},
        Platform.TELEGRAM,
        "telegram",
        {"platform": "telegram", "chat_id": "123456789", "thread_id": "222333"},
        adapters,
        config,
    )
    assert routed_error is None
    assert routed_resolved is not None
    assert routed_resolved[2] is primary_telegram

    mismatched_resolved, mismatched_error = _resolve_target_transport(
        {"id": "mismatch"},
        Platform.TELEGRAM,
        "telegram",
        {"platform": "telegram", "chat_id": "123456789", "thread_id": "999999"},
        adapters,
        config,
    )
    assert mismatched_resolved is None
    assert mismatched_error is not None
    assert getattr(adapters, "get")(Platform.TELEGRAM) is None


def test_restart_drain_secondary_with_unrelated_own_adapter_still_uses_exact_routed_primary_target(
    tmp_path: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    from cron import scheduler, scheduler_preflight
    import gateway.run as gateway_run
    from gateway.profile_routing import parse_profile_routes

    secondary_home = tmp_path / "root" / "profiles" / "fennec"
    primary_telegram = object()
    own_a2a = object()
    a2a_platform = Platform("a2a")
    runner = SimpleNamespace(_profile_adapters={"fennec": {a2a_platform: own_a2a}})
    captured: list[object] = []

    @contextmanager
    def fake_scope(_home: object):
        yield

    monkeypatch.setattr(gateway_run, "_handoff_watch_scopes", lambda _runner: [("fennec", secondary_home)])
    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    monkeypatch.setattr(
        scheduler_preflight,
        "_primary_profile_routes_for_current_home",
        lambda: parse_profile_routes([
            {
                "name": "fennec-telegram-topic",
                "platform": "telegram",
                "chat_id": "123456789",
                "thread_id": "222333",
                "profile": "fennec",
            }
        ]),
    )
    monkeypatch.setattr(
        scheduler,
        "drain_delivery_queue",
        lambda adapters, _loop: captured.append(adapters),
    )

    gateway_run._drain_restart_safe_cron_deliveries(
        {Platform.TELEGRAM: primary_telegram}, object(), runner
    )

    assert len(captured) == 1
    adapters = captured[0]
    config = _fake_config(a2a_platform)

    own_resolved, own_error = _resolve_target_transport(
        {"id": "own"},
        a2a_platform,
        "a2a",
        {"platform": "a2a", "chat_id": "secondary-owned"},
        adapters,
        config,
    )
    assert own_error is None
    assert own_resolved is not None
    assert own_resolved[2] is own_a2a

    routed_resolved, routed_error = _resolve_target_transport(
        {"id": "routed"},
        Platform.TELEGRAM,
        "telegram",
        {"platform": "telegram", "chat_id": "123456789", "thread_id": "222333"},
        adapters,
        config,
    )
    assert routed_error is None
    assert routed_resolved is not None
    assert routed_resolved[2] is primary_telegram

    mismatched_resolved, mismatched_error = _resolve_target_transport(
        {"id": "mismatch"},
        Platform.TELEGRAM,
        "telegram",
        {"platform": "telegram", "chat_id": "123456789", "thread_id": "999999"},
        adapters,
        config,
    )
    assert mismatched_resolved is None
    assert mismatched_error is not None
    assert getattr(adapters, "get")(Platform.TELEGRAM) is None
