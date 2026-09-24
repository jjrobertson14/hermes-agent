"""A live writer's WAL generation survives lock cancellation and sibling closes.

SQLite guards a WAL generation with per-PROCESS POSIX locks, so any raw ``open()``/``close()`` of
``state.db`` or ``-shm`` inside the holder (howtocorrupt.html §2.2) silently drops them; the next
last-connection close anywhere then checkpoints and unlinks ``-wal``/``-shm`` and the holder
sticky-halts with ``DeletedWalGenerationError`` (#109727, #110042, #110276, Desktop "chat fails
after update"). ``hermes_state_lockguard`` re-holds the same ranges as OFD locks that a stray
close cannot cancel. Linux-only: the scenario needs ``/proc``-visible POSIX lock semantics.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tests.hermes_state._wal_generation_harness import make_db, pin_wal, require_wal

pytestmark = pytest.mark.linux_only


def _foreign_open_close(db_path: Path) -> None:
    """Another process opens state.db, reads, closes: SQLite's last-connection WAL reset runs there."""
    subprocess.run([sys.executable, "-c",
                    f"import sqlite3; c = sqlite3.connect({str(db_path)!r}); "
                    "c.execute('select count(*) from messages').fetchone(); c.close()"], check=True)


def test_holder_survives_stray_close_then_sibling_closes(tmp_path, monkeypatch):
    pin_wal(monkeypatch)
    db = make_db(tmp_path / "state.db", "s", "seed")
    wal = require_wal(db)
    wal_inode = wal.stat().st_ino
    try:
        for name in ("state.db", "state.db-shm"):  # the §2.2 bug, e.g. a raw header probe
            os.close(os.open(tmp_path / name, os.O_RDONLY))
        # Same-process second handle (Desktop backend / Herder tab shape) closing...
        sibling = SessionDB(db_path=db.db_path)
        sibling.append_message("s", role="user", content="tab")
        sibling.close()
        # ...and a foreign process (cron worker / one-shot CLI shape) closing.
        _foreign_open_close(db.db_path)
        assert wal.exists() and wal.stat().st_ino == wal_inode, "a sibling close unlinked the live WAL"
        db.append_message("s", role="user", content="after")  # would raise DeletedWalGenerationError
    finally:
        db.close()
    assert sqlite3.connect(db.db_path).execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3


def test_holder_survives_replaced_named_shm_then_close_cascade(tmp_path, monkeypatch):
    """The current sidecars survive closes after a live writer's named SHM is replaced."""
    pin_wal(monkeypatch)
    path = tmp_path / "state.db"
    writer = make_db(path, "s", "seed")
    wal = require_wal(writer)
    shm = Path(f"{path}-shm")
    if not shm.exists():
        writer.close()
        pytest.skip("WAL SHM sidecar missing after first write")

    retired_shm = tmp_path / "retired-state.db-shm"
    os.replace(shm, retired_shm)
    try:
        # A foreign SQLite opener recreates the named SHM generation for the live WAL.
        _foreign_open_close(path)
        current_shm = Path(f"{path}-shm")
        assert current_shm.exists(), "foreign SQLite open did not recreate the named SHM"

        # Public writes reach SessionDB's ordinary periodic checkpoint cadence.
        for index in range(50):
            writer.append_message("s", role="user", content=f"cadence-{index}")

        current_wal_inode = wal.stat().st_ino
        current_shm_inode = current_shm.stat().st_ino
        for sidecar in (path, current_shm):
            os.close(os.open(sidecar, os.O_RDONLY))
        sibling = SessionDB(db_path=path)
        try:
            sibling.append_message("s", role="assistant", content="sibling")
        finally:
            sibling.close()
        _foreign_open_close(path)

        assert wal.exists() and wal.stat().st_ino == current_wal_inode, (
            "sibling/foreign close removed the current WAL generation"
        )
        assert current_shm.exists() and current_shm.stat().st_ino == current_shm_inode, (
            "sibling/foreign close removed the current SHM generation"
        )
        writer.append_message("s", role="user", content="writer survives")
        assert writer.get_messages("s")[-1]["content"] == "writer survives"
    finally:
        writer.close()
