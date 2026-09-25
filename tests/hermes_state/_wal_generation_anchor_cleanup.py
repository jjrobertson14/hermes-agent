"""Test-only cleanup for process-lifetime WAL generation anchors."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

import hermes_state_generation_anchor as anchor


@pytest.fixture(autouse=True)
def cleanup_test_created_generation_anchors(tmp_path: Path) -> Iterator[None]:
    """Remove this test's anchors without closing descriptors recycled for another inode."""
    with anchor._LOCK:
        before = dict(anchor._FALLBACKS)

    yield

    removed: list[tuple[anchor.Identity, anchor.Anchor]] = []
    with anchor._LOCK:
        for identity, record in list(anchor._FALLBACKS.items()):
            if before.get(identity) != record:
                removed.append((identity, record))
                del anchor._FALLBACKS[identity]

    expected_identities_by_fd: dict[int, set[anchor.Identity]] = {}
    for identity, record in removed:
        for fd in record:
            expected_identities_by_fd.setdefault(fd, set()).add(identity)
    if os.path.isdir("/proc/self/fd"):
        for name in os.listdir("/proc/self/fd"):
            if not name.isdigit():
                continue
            fd = int(name)
            try:
                target = os.readlink(f"/proc/self/fd/{fd}")
                stat = os.fstat(fd)
            except OSError:
                continue
            path = target.removesuffix(" (deleted)")
            if (target.endswith(" (deleted)") and path.startswith(f"{tmp_path}{os.sep}")
                    and path.endswith(("-wal", "-shm"))):
                expected_identities_by_fd.setdefault(fd, set()).add((stat.st_dev, stat.st_ino))
    for fd, identities in expected_identities_by_fd.items():
        try:
            stat = os.fstat(fd)
        except OSError:
            continue
        if (stat.st_dev, stat.st_ino) in identities:
            try:
                os.close(fd)
            except OSError:
                pass
