"""Hold a state.db writer's WAL-mode locks on private descriptors.

SQLite's WAL locks are process-owned POSIX locks, so an unrelated in-process
``open()``/``close()`` of state.db or -shm can cancel them.  This module keeps
equivalent OFD locks on private descriptors.  They survive that stray close
and conflict with the EXCLUSIVE locks required to reset a live WAL generation.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("hermes_state")

_PENDING_BYTE = 0x40000000
_SHARED_FIRST = _PENDING_BYTE + 2
_SHARED_SIZE = 510
_SHM_DMS_BYTE = 128

try:
    import fcntl
    _F_OFD_SETLK: Optional[int] = getattr(fcntl, "F_OFD_SETLK", 37) if sys.platform.startswith("linux") else None
    _F_RDLCK, _F_UNLCK, _SEEK_SET = fcntl.F_RDLCK, fcntl.F_UNLCK, os.SEEK_SET
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
    _F_OFD_SETLK = None
    _F_RDLCK = _F_UNLCK = _SEEK_SET = 0

def _flock(lock_type: int, start: int, length: int) -> bytes:
    return struct.pack("@hhqqi", lock_type, _SEEK_SET, start, length, 0)


def _ofd_lock(fd: int, lock_type: int, start: int, length: int) -> bool:
    assert fcntl is not None and _F_OFD_SETLK is not None
    try:
        fcntl.fcntl(fd, _F_OFD_SETLK, _flock(lock_type, start, length))
    except BlockingIOError:
        return False
    return True


class _PathGuard:
    __slots__ = ("main_fd", "main_ident", "shm_fd", "shm_ident", "refs", "main_locked", "shm_locked", "retired_fds")

    def __init__(self) -> None:
        self.main_fd = self.shm_fd = -1
        self.main_ident = self.shm_ident = None
        self.refs = 0
        self.main_locked = self.shm_locked = False
        self.retired_fds: List[int] = []


_LOCK = threading.Lock()
_GUARDS: Dict[str, _PathGuard] = {}


def _key(db_path: Path) -> str:
    """Canonical path key, matching ``hermes_cli.sqlite_safe_read`` tracking."""
    try:
        return str(Path(db_path).resolve())
    except OSError:
        return str(db_path)


def supported() -> bool:
    return _F_OFD_SETLK is not None


def _bind_fd(path: str, fd: int, ident, retired_fds: List[int]) -> tuple:
    """Return a private fd for the inode currently named by *path*.

    A replaced sidecar must not inherit a lock on an old generation.  Retired
    descriptors stay open until all tracked connections on this path are gone:
    that is the only point at which closing one cannot perturb a live SQLite
    connection's process locks.
    """
    try:
        st = os.stat(path)
    except OSError:
        return fd, ident
    current = (st.st_dev, st.st_ino)
    if fd >= 0 and ident == current:
        return fd, ident
    if fd >= 0:
        retired_fds.append(fd)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError:
        return -1, None
    return fd, current


def _apply_locked(guard: _PathGuard, db_path: str) -> None:
    guard.main_fd, guard.main_ident = _bind_fd(
        db_path, guard.main_fd, guard.main_ident, guard.retired_fds)
    if guard.main_fd >= 0:
        guard.main_locked = _ofd_lock(guard.main_fd, _F_RDLCK, _SHARED_FIRST, _SHARED_SIZE)
    guard.shm_fd, guard.shm_ident = _bind_fd(
        db_path + "-shm", guard.shm_fd, guard.shm_ident, guard.retired_fds)
    if guard.shm_fd >= 0:
        guard.shm_locked = _ofd_lock(guard.shm_fd, _F_RDLCK, _SHM_DMS_BYTE, 1)


def hold(db_path: Path) -> None:
    """Acquire one per-handle reference after a writer enters WAL mode."""
    if not supported():
        return
    key = _key(db_path)
    with _LOCK:
        guard = _GUARDS.setdefault(key, _PathGuard())
        guard.refs += 1
        try:
            _apply_locked(guard, key)
        except OSError:
            logger.debug("WAL lock guard unavailable for %s", key, exc_info=True)


def refresh(db_path: Path) -> None:
    """Refresh a held guard after SQLite creates/replaces its -shm sidecar."""
    if not supported():
        return
    key = _key(db_path)
    with _LOCK:
        guard = _GUARDS.get(key)
        if guard is None or guard.refs <= 0:
            return
        try:
            _apply_locked(guard, key)
        except OSError:
            logger.debug("WAL lock guard refresh failed for %s", key, exc_info=True)


def release(db_path: Path) -> None:
    """Release one handle reference before that handle closes its SQLite connection."""
    if not supported():
        return
    key = _key(db_path)
    with _LOCK:
        guard = _GUARDS.get(key)
        if guard is None or guard.refs <= 0:
            return
        guard.refs -= 1
        if guard.refs:
            return
        for fd, start, length in ((guard.main_fd, _SHARED_FIRST, _SHARED_SIZE), (guard.shm_fd, _SHM_DMS_BYTE, 1)):
            if fd >= 0:
                try:
                    _ofd_lock(fd, _F_UNLCK, start, length)
                except OSError:
                    logger.debug("WAL lock guard unlock failed for %s", key, exc_info=True)
        guard.main_locked = guard.shm_locked = False


def retire_idle(db_path: Path) -> None:
    """Close active and retired private fds only after tracked connections are gone."""
    if not supported():
        return
    key = _key(db_path)
    with _LOCK:
        guard = _GUARDS.get(key)
        if guard is None or guard.refs or _path_has_live_connection(key):
            return
        del _GUARDS[key]
        for fd in (guard.main_fd, guard.shm_fd, *guard.retired_fds):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass


def _path_has_live_connection(key: str) -> bool:
    try:
        from hermes_cli.sqlite_safe_read import has_live_connection
    except ImportError:
        return True
    return has_live_connection(key)


def held(db_path: Path) -> bool:
    with _LOCK:
        guard = _GUARDS.get(_key(db_path))
        return bool(guard and guard.refs and guard.main_locked and guard.shm_locked)


def owned_fds(db_path: Optional[Path] = None) -> frozenset:
    """Private descriptors, optionally limited to one canonical database path."""
    with _LOCK:
        fds = set()
        guards = (_GUARDS.get(_key(db_path)),) if db_path is not None else _GUARDS.values()
        for guard in guards:
            if guard is None:
                continue
            fds.update(fd for fd in (guard.main_fd, guard.shm_fd, *guard.retired_fds) if fd >= 0)
        return frozenset(fds)
