"""Private process-lifetime OFD anchors for anomalous SQLite sidecar generations."""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional, Set, Tuple

Identity = Tuple[int, int]
Anchor = Tuple[int, int, int]  # hidden anchor, canary witness, working duplicate

# Outside every SQLite lock range; solely an OFD ownership canary.
_FALLBACK_CANARY_BYTE = 0x40000000 + 2 + 510
_LOCK = threading.RLock()
_FALLBACKS: dict[Identity, Anchor] = {}


def _guard():
    # A late import avoids the lockguard -> anchor import cycle.
    import hermes_state_lockguard

    return hermes_state_lockguard


def _same_fallback_ofd(candidate: int, anchor: int, ident: Identity) -> bool:
    """Authenticate a duplicate against an anchor OFD without trusting fd numbers."""
    guard = _guard()
    if candidate < 0 or anchor < 0 or not sys.platform.startswith("linux"):
        return False
    try:
        if any((st.st_dev, st.st_ino) != ident for st in (os.fstat(anchor), os.fstat(candidate))):
            return False
        if not guard._ofd_lock(anchor, getattr(guard.fcntl, "F_WRLCK"), _FALLBACK_CANARY_BYTE, 1):
            return False
        same = guard._ofd_lock(candidate, guard._F_RDLCK, _FALLBACK_CANARY_BYTE, 1)
        guard._ofd_lock(anchor, getattr(guard.fcntl, "F_WRLCK"), _FALLBACK_CANARY_BYTE, 1)
        return same
    except OSError:
        return False


def _verified_anchor(ident: Identity, record: Anchor) -> Set[int]:
    """Return private fds only after hidden anchor and witness prove each other."""
    anchor, witness, working = record
    try:
        if any((st.st_dev, st.st_ino) != ident for st in map(os.fstat, record)):
            return set()
    except OSError:
        return set()
    if not _same_fallback_ofd(anchor, witness, ident):
        return set()
    verified = {anchor, witness}
    if _same_fallback_ofd(working, anchor, ident):
        verified.add(working)
    return verified


def fallback_fd(ident: Identity, path: str) -> Optional[int]:
    """Return a verified working duplicate of a Linux process-lifetime anchor."""
    if not sys.platform.startswith("linux"):
        return None
    with _LOCK:
        record = _FALLBACKS.get(ident)
        if record is not None:
            anchor, witness, working = record
            verified = _verified_anchor(ident, record)
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
        guard = _guard()
        if not guard._ofd_lock(anchor, getattr(guard.fcntl, "F_WRLCK"), _FALLBACK_CANARY_BYTE, 1):
            for fd in (working, witness, anchor):
                try:
                    os.close(fd)
                except OSError:
                    pass
            return None
        _FALLBACKS[ident] = anchor, witness, working
        return working


def is_fallback_fd(fd: int, identity: Optional[Identity] = None) -> bool:
    """Whether *fd* is an active, proven private anchor descriptor."""
    if not sys.platform.startswith("linux"):
        return False
    with _LOCK:
        return any((identity is None or identity == ident) and fd in _verified_anchor(ident, record)
                   for ident, record in _FALLBACKS.items())


def is_fallback_fd_path(fd_path: str) -> bool:
    try:
        return is_fallback_fd(int(os.path.basename(fd_path)))
    except ValueError:
        return False


def release_final(ident: Identity, rng: tuple[int, int], own_fds_for) -> None:
    """Release exactly the final range while retaining its process-lifetime anchor."""
    guard = _guard()
    with _LOCK:
        record = _FALLBACKS.get(ident)
        verified = _verified_anchor(ident, record) if record else set()
        if record:
            if verified:
                guard._ofd_lock(record[0], guard._F_UNLCK, *rng)
            return
        for fd, found in own_fds_for({ident}):
            if found == ident:
                guard._ofd_lock(fd, guard._F_UNLCK, *rng)
