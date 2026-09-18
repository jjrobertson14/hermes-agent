"""A live writer's WAL generation survives lock cancellation.

SQLite guards a WAL generation with per-PROCESS POSIX locks, so a raw
``open()``/``close()`` of ``state.db`` or ``-shm`` in the writer process can
silently drop them. A foreign SQLite connection may then reset the WAL while
the original writer is still live. Linux-only: the scenario requires POSIX
lock semantics.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_state import SessionDB
import hermes_state_lockguard
from hermes_state_dbfile import iter_deleted_sqlite_sidecar_holders
from tests.hermes_state._wal_generation_harness import make_db, pin_wal, require_wal

pytestmark = pytest.mark.linux_only


def _foreign_open_close(db_path: Path) -> None:
    """Another process opens state.db, reads, then closes."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sqlite3; c = sqlite3.connect({str(db_path)!r}); "
            "c.execute('select count(*) from messages').fetchone(); c.close()",
        ],
        check=True,
    )


def _foreign_shm_exclusive_ok(shm_path: Path) -> bool:
    """Whether another process can acquire the WAL-index DMS byte exclusively."""
    code = (
        "import fcntl, os, struct\n"
        f"fd = os.open({str(shm_path)!r}, os.O_RDWR)\n"
        "lk = struct.pack('@hhqqi', fcntl.F_WRLCK, os.SEEK_SET, 128, 1, 0)\n"
        "try:\n    fcntl.fcntl(fd, 37, lk); print('EXCLUSIVE_ACQUIRED')\n"
        "except BlockingIOError:\n    print('REFUSED')\n"
    )
    return "EXCLUSIVE_ACQUIRED" in subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout


def _foreign_main_exclusive_ok(db_path: Path) -> bool:
    """Whether a foreign process can take SQLite's main-file exclusive range."""
    code = (
        "import fcntl, os, struct\n"
        f"fd = os.open({str(db_path)!r}, os.O_RDWR)\n"
        "lk = struct.pack('@hhqqi', fcntl.F_WRLCK, os.SEEK_SET, 0x40000002, 510, 0)\n"
        "try:\n    fcntl.fcntl(fd, 37, lk); print('EXCLUSIVE_ACQUIRED')\n"
        "except BlockingIOError:\n    print('REFUSED')\n"
    )
    return "EXCLUSIVE_ACQUIRED" in subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout


def _foreign_delete_mode_ok(db_path: Path) -> bool:
    """Whether a foreign SQLite connection can take the exclusive DELETE-mode lock."""
    code = (
        "import sqlite3\n"
        f"c = sqlite3.connect({str(db_path)!r})\n"
        "print(c.execute('PRAGMA journal_mode=DELETE').fetchone()[0])\n"
        "c.close()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    return result.returncode == 0 and result.stdout.strip() == "delete"


def test_writer_survives_raw_close_then_foreign_sqlite_close(tmp_path, monkeypatch):
    pin_wal(monkeypatch)
    db = make_db(tmp_path / "state.db", "s", "seed")
    wal = require_wal(db)
    wal_inode = wal.stat().st_ino
    try:
        for name in ("state.db", "state.db-shm"):
            os.close(os.open(tmp_path / name, os.O_RDONLY))
        _foreign_open_close(db.db_path)
        assert wal.exists() and wal.stat().st_ino == wal_inode, (
            "foreign close unlinked the live WAL"
        )
        db.append_message("s", role="user", content="after")
    finally:
        db.close()

    reopened = sqlite3.connect(db.db_path)
    try:
        assert reopened.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    finally:
        reopened.close()


def test_first_writer_survives_second_writer_close_then_raw_close(tmp_path, monkeypatch):
    pin_wal(monkeypatch)
    db = make_db(tmp_path / "state.db", "s", "seed")
    wal = require_wal(db)
    wal_inode = wal.stat().st_ino
    try:
        sibling = SessionDB(db_path=db.db_path)
        sibling.append_message("s", role="user", content="from-second-writer")
        sibling.close()
        for name in ("state.db", "state.db-shm"):
            os.close(os.open(tmp_path / name, os.O_RDONLY))
        _foreign_open_close(db.db_path)
        assert wal.exists() and wal.stat().st_ino == wal_inode, (
            "second writer release let a foreign close unlink the live WAL"
        )
        db.append_message("s", role="user", content="after")
    finally:
        db.close()

    reopened = sqlite3.connect(db.db_path)
    try:
        assert reopened.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
    finally:
        reopened.close()


def test_second_writer_survives_first_writer_close_then_raw_close(tmp_path, monkeypatch):
    pin_wal(monkeypatch)
    first = make_db(tmp_path / "state.db", "s", "seed")
    require_wal(first)
    second = SessionDB(db_path=first.db_path)
    second.append_message("s", role="user", content="from-second-writer")
    wal = require_wal(second)
    wal_inode = wal.stat().st_ino
    try:
        first.close()
        for name in ("state.db", "state.db-shm"):
            os.close(os.open(tmp_path / name, os.O_RDONLY))
        _foreign_open_close(second.db_path)
        assert wal.exists() and wal.stat().st_ino == wal_inode, (
            "first writer release let a foreign close unlink the second writer's live WAL"
        )
        second.append_message("s", role="user", content="after")
    finally:
        second.close()

    reopened = sqlite3.connect(second.db_path)
    try:
        assert reopened.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
    finally:
        reopened.close()


def test_guard_rebinds_to_replaced_shm_after_checkpoint_refresh(tmp_path, monkeypatch):
    pin_wal(monkeypatch)
    db = make_db(tmp_path / "state.db", "s", "seed")
    require_wal(db)
    shm = tmp_path / "state.db-shm"
    retired_shm = tmp_path / "retired-shm"
    try:
        shutil.copyfile(shm, retired_shm)
        os.replace(retired_shm, shm)
        db._try_wal_checkpoint()
        for name in ("state.db", "state.db-shm"):
            os.close(os.open(tmp_path / name, os.O_RDONLY))
        assert not _foreign_shm_exclusive_ok(shm), (
            "checkpoint refresh left the replacement -shm generation unguarded"
        )
    finally:
        db.close()


def test_last_writer_close_releases_private_guard_for_foreign_delete_mode(tmp_path, monkeypatch):
    pin_wal(monkeypatch)
    db = make_db(tmp_path / "state.db", "s", "seed")
    require_wal(db)
    db.close()

    assert _foreign_delete_mode_ok(db.db_path), (
        "the last writer close left a private WAL guard that blocks foreign DELETE mode"
    )


def test_reopened_writer_is_guarded_again(tmp_path, monkeypatch):
    """A writer reopened after close keeps the private WAL lock alive (#94736)."""
    pin_wal(monkeypatch)
    db = make_db(tmp_path / "state.db", "s", "seed")
    require_wal(db)
    db.close()
    db.append_message("s", role="user", content="after close")
    try:
        assert db._conn is not None, "reopen returned no live writer"
        assert not _foreign_main_exclusive_ok(db.db_path), (
            "reopen returned an unguarded writer"
        )
    finally:
        db.close()

    assert _foreign_main_exclusive_ok(db.db_path), (
        "a true last close did not lift the guard"
    )


def test_guard_never_outlives_handle_under_fd_reuse(tmp_path, monkeypatch):
    """A+B live -> close A -> open C -> close B leaves C guarded until C closes."""
    pin_wal(monkeypatch)
    path = tmp_path / "state.db"
    a = make_db(path, "s", "seed")
    require_wal(a)
    b = SessionDB(db_path=path)
    a.close()
    c = SessionDB(db_path=path)
    wal = require_wal(c)
    wal_inode = wal.stat().st_ino
    b.close()
    try:
        assert not _foreign_main_exclusive_ok(path), (
            "C was recorded as guarded while no private lock remained"
        )
        for name in ("state.db", "state.db-shm"):
            os.close(os.open(tmp_path / name, os.O_RDONLY))
        _foreign_open_close(path)
        assert wal.exists() and wal.stat().st_ino == wal_inode, (
            "foreign close unlinked C's live WAL"
        )
        c.append_message("s", role="user", content="still writes")
    finally:
        c.close()

    assert _foreign_main_exclusive_ok(path), "a lock survived the last handle's close"
    reopened = sqlite3.connect(path)
    try:
        assert reopened.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    finally:
        reopened.close()


def test_relative_and_absolute_handles_release_one_guard_after_final_close(tmp_path, monkeypatch):
    """Aliases share the guard lifecycle tracked by the SQLite connection registry."""
    pin_wal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    relative = Path("state.db")
    absolute = tmp_path / relative
    first = make_db(relative, "s", "seed")
    require_wal(first)
    second = SessionDB(db_path=absolute)
    try:
        # Closing the absolute spelling first left its zero-ref guard descriptors
        # stranded: the connection tracker had a live relative-path entry.
        second.close()
        first.close()

        assert not hermes_state_lockguard.held(relative)
        assert hermes_state_lockguard.owned_fds(relative) == frozenset()
        assert iter_deleted_sqlite_sidecar_holders(absolute) == []
        assert _foreign_delete_mode_ok(absolute)
    finally:
        first.close()
        second.close()
