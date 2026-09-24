"""Keep SQLite WAL locks safe from an unrelated in-process ``close()``.

SQLite's POSIX locks are process scoped, so an unrelated open/close of state.db
or state.db-shm can discard its WAL protection.  We mirror the two lock ranges
with OFD locks.  Usually SQLite's descriptors carry those locks.  On Linux
only, when a *currently named* DB/SHM inode has no such descriptor (for example
after a renamed SHM is recreated), we pin that anomalous generation with a
private anchor open-file-description instead.

An anchor is intentionally process-lifetime state: it is never closed by
Python, including normal SessionDB teardown.  Process exit is its cleanup.  At
most one anchor OFD is retained for each abnormal inode generation; replacement
generations can therefore grow this map, but ordinary opens never enter it.
The guard range itself is still refcounted: the last guarded SessionDB unlocks
the anchor before it closes, so a foreign final close or journal-mode change is
not blocked by an idle anchor.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
import threading
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger("hermes_state")

_PENDING_BYTE = 0x40000000
_SHARED_FIRST = _PENDING_BYTE + 2
_SHARED_SIZE = 510
_SHM_DMS_BYTE = 128
# Outside every SQLite lock range; solely an OFD ownership canary.
_FALLBACK_CANARY_BYTE = _SHARED_FIRST + _SHARED_SIZE

try:
    import fcntl
    _F_OFD_SETLK: Optional[int] = getattr(
        fcntl, "F_OFD_SETLK", {"linux": 37, "darwin": 90}.get(sys.platform.rstrip("0123456789")))
    _F_RDLCK, _F_UNLCK, _SEEK_SET = fcntl.F_RDLCK, fcntl.F_UNLCK, os.SEEK_SET
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
    _F_OFD_SETLK = None
    _F_RDLCK = _F_UNLCK = _SEEK_SET = 0

_FLOCK_FORMAT = "@qqihh" if sys.platform == "darwin" or "bsd" in sys.platform else "@hhqqi"

Identity = Tuple[int, int]
Held = Dict[Identity, Tuple[int, int]]
Anchor = Tuple[int, int, int]  # hidden anchor, hidden canary witness, working duplicate

_LOCK = threading.RLock()
_HANDLES: Dict[Identity, int] = {}
# Private Linux anomaly anchors.  All three fds initially share one OFD.  The
# first two are never returned; the witness lets us fail closed if an external
# close/recycle ever makes the hidden anchor itself unverifiable.
_FALLBACKS: Dict[Identity, Anchor] = {}


def supported() -> bool:
    return _F_OFD_SETLK is not None


def _flock(lock_type: int, start: int, length: int) -> bytes:
    if _FLOCK_FORMAT == "@qqihh":
        return struct.pack(_FLOCK_FORMAT, start, length, 0, lock_type, _SEEK_SET)
    return struct.pack(_FLOCK_FORMAT, lock_type, _SEEK_SET, start, length, 0)


def _ofd_lock(fd: int, lock_type: int, start: int, length: int) -> bool:
    assert fcntl is not None and _F_OFD_SETLK is not None
    try:
        fcntl.fcntl(fd, _F_OFD_SETLK, _flock(lock_type, start, length))
    except BlockingIOError:
        return False
    return True


def _identity(path: str) -> Optional[Identity]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _own_fds_for(identities: Set[Identity]):
    """Yield our descriptors for the requested inodes."""
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            names = os.listdir(fd_dir)
        except OSError:
            continue
        for name in names:
            if name.isdigit():
                fd = int(name)
                try:
                    st = os.fstat(fd)
                except OSError:
                    continue
                ident = st.st_dev, st.st_ino
                if ident in identities:
                    yield fd, ident
        return


def _same_fallback_ofd(candidate: int, anchor: int) -> bool:
    """Authenticate a duplicate against an anchor OFD without trusting fd numbers."""
    if candidate < 0 or anchor < 0 or not sys.platform.startswith("linux"):
        return False
    try:
        if not _ofd_lock(anchor, getattr(fcntl, "F_WRLCK"), _FALLBACK_CANARY_BYTE, 1):
            return False
        same = _ofd_lock(candidate, _F_RDLCK, _FALLBACK_CANARY_BYTE, 1)
        _ofd_lock(anchor, getattr(fcntl, "F_WRLCK"), _FALLBACK_CANARY_BYTE, 1)
        return same
    except OSError:
        return False


def _verified_anchor(record: Anchor) -> Set[int]:
    """Return private fds only after hidden anchor and witness prove each other."""
    anchor, witness, working = record
    if not _same_fallback_ofd(anchor, witness):
        return set()
    verified = {anchor, witness}
    if _same_fallback_ofd(working, anchor):
        verified.add(working)
    return verified


def _fallback_fd(ident: Identity, path: str) -> Optional[int]:
    """Return a verified working duplicate of a Linux process-lifetime anchor.

    A recycled working number is never closed.  If the two hidden fds still
    prove the anchor OFD, a fresh duplicate is made from it.  If they cannot,
    ownership is unverifiable and this fails closed without touching any fd.
    """
    if not sys.platform.startswith("linux"):
        return None
    record = _FALLBACKS.get(ident)
    if record is not None:
        anchor, witness, working = record
        verified = _verified_anchor(record)
        if not verified:
            return None
        if working in verified:
            return working
        try:
            fresh = os.dup(anchor)
        except OSError:
            return None
        _FALLBACKS[ident] = anchor, witness, fresh
        return fresh
    anchor = witness = working = -1
    try:
        anchor = os.open(path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
        st = os.fstat(anchor)
        if (st.st_dev, st.st_ino) != ident:
            os.close(anchor)
            return None
        witness, working = os.dup(anchor), os.dup(anchor)
    except OSError:
        for fd in (working, witness, anchor):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        return None
    # Establish the canary before publishing the anchor.  Failure here is setup
    # cleanup, not normal lifecycle retirement.
    if not _ofd_lock(anchor, getattr(fcntl, "F_WRLCK"), _FALLBACK_CANARY_BYTE, 1):
        for fd in (working, witness, anchor):
            try:
                os.close(fd)
            except OSError:
                pass
        return None
    _FALLBACKS[ident] = anchor, witness, working
    return working


def _is_fallback_fd(fd: int, identity: Optional[Identity] = None) -> bool:
    """Whether *fd* is an active, proven private anchor descriptor."""
    if not sys.platform.startswith("linux"):
        return False
    with _LOCK:
        for ident, record in _FALLBACKS.items():
            if (identity is None or identity == ident) and fd in _verified_anchor(record):
                return True
    return False


def _is_fallback_fd_path(fd_path: str) -> bool:
    try:
        return _is_fallback_fd(int(os.path.basename(fd_path)))
    except ValueError:
        return False


def _guard_ranges(db_path) -> Held:
    base = os.fspath(db_path)
    ranges: Held = {}
    for path, rng in ((base, (_SHARED_FIRST, _SHARED_SIZE)), (base + "-shm", (_SHM_DMS_BYTE, 1))):
        ident = _identity(path)
        if ident is not None:
            ranges[ident] = rng
    return ranges


def hold(db_path, held: Optional[Held] = None) -> Held:
    """Install this handle's OFD WAL guard, pinning only anomalous Linux inodes."""
    held = {} if held is None else held
    if not supported():
        return held
    base = os.fspath(db_path)
    ranges = _guard_ranges(base)
    try:
        with _LOCK:
            own_fds: Dict[Identity, list[int]] = {ident: [] for ident in ranges}
            for fd, ident in _own_fds_for(set(ranges)):
                if not _is_fallback_fd(fd, ident):
                    own_fds[ident].append(fd)
            shm_ident = _identity(base + "-shm")
            for ident, rng in ranges.items():
                fds = own_fds[ident]
                if not fds:
                    fallback = _fallback_fd(ident, base + ("-shm" if ident == shm_ident else ""))
                    fds = [fallback] if fallback is not None else []
                for fd in fds:
                    if _ofd_lock(fd, _F_RDLCK, *rng) and ident not in held:
                        held[ident] = rng
                        _HANDLES[ident] = _HANDLES.get(ident, 0) + 1
    except OSError:
        logger.debug("WAL lock guard unavailable for %s", base, exc_info=True)
    return held


def release(held: Held) -> None:
    """Release final guard ranges but deliberately retain any anomaly anchor OFD."""
    if not supported() or not held:
        return
    with _LOCK:
        to_unlock: Held = {}
        for ident, rng in held.items():
            remaining = _HANDLES.get(ident, 1) - 1
            if remaining > 0:
                _HANDLES[ident] = remaining
            else:
                _HANDLES.pop(ident, None)
                to_unlock[ident] = rng
        held.clear()
        try:
            for ident, rng in to_unlock.items():
                record = _FALLBACKS.get(ident)
                verified = _verified_anchor(record) if record else set()
                if record and verified:
                    _ofd_lock(record[0], _F_UNLCK, *rng)
                # The anchor is only the exceptional first guard.  A later
                # SQLite connection may also have carried this range.
                for fd, found in _own_fds_for({ident}):
                    if found == ident and fd not in verified:
                        _ofd_lock(fd, _F_UNLCK, *rng)
        except OSError:
            pass
