"""The WAL lock guard tracks the lifecycle of the handle it protects."""

import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import hermes_state_generation_anchor as anchor
import hermes_state_lockguard as lg
from hermes_state import SessionDB
from tests.hermes_state._wal_generation_harness import make_db, pin_wal, require_wal

pytest_plugins = ("tests.hermes_state._wal_generation_anchor_cleanup",)
pytestmark = pytest.mark.platforms("linux")


def _foreign_exclusive_ok(path: str) -> bool:
    """Another process tries the EXCLUSIVE a close-time WAL reset needs; True = nothing guards."""
    code = (
        "import fcntl, os, struct, sys\n"
        f"fd = os.open({path!r}, os.O_RDWR)\n"
        "lk = struct.pack('@hhqqi', fcntl.F_WRLCK, 0, 0x40000002, 510, 0)\n"
        "try:\n    fcntl.fcntl(fd, 37, lk); print('EXCLUSIVE_ACQUIRED')\n"
        "except BlockingIOError:\n    print('REFUSED')\n"
    )
    return "EXCLUSIVE_ACQUIRED" in subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout


def _foreign_canary_write_ok(path: str) -> bool:
    code = (
        "import fcntl, os, struct, sys\n"
        "fd = os.open(sys.argv[1], os.O_RDWR)\n"
        "lk = struct.pack('@hhqqi', fcntl.F_WRLCK, 0, 0x40000000 + 2 + 510, 1, 0)\n"
        "try:\n    fcntl.fcntl(fd, 37, lk); print('WRITE_ACQUIRED')\n"
        "except BlockingIOError:\n    print('REFUSED')\n"
    )
    return "WRITE_ACQUIRED" in subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True).stdout


def test_recycled_working_fd_for_another_inode_is_not_an_anchor(tmp_path):
    """A recycled working descriptor must not authenticate or lock its replacement inode."""
    protected = tmp_path / "protected"
    unrelated = tmp_path / "unrelated"
    protected.touch()
    unrelated.touch()
    protected_stat = protected.stat()
    protected_identity = (protected_stat.st_dev, protected_stat.st_ino)

    working_fd = anchor.fallback_fd(protected_identity, str(protected))
    assert working_fd is not None
    try:
        os.close(working_fd)
        replacement_fd = os.open(unrelated, os.O_RDWR)
        try:
            os.dup2(replacement_fd, working_fd)
        finally:
            if replacement_fd != working_fd:
                os.close(replacement_fd)

        unrelated_stat = unrelated.stat()
        assert os.fstat(working_fd).st_ino == unrelated_stat.st_ino
        assert os.fstat(working_fd).st_ino != protected_identity[1]
        assert not anchor.is_fallback_fd(working_fd, protected_identity)
        assert os.fstat(working_fd).st_ino == unrelated_stat.st_ino, "verification closed the unrelated fd"
        assert _foreign_canary_write_ok(unrelated), "verification locked the unrelated fd"
    finally:
        try:
            os.close(working_fd)
        except OSError:
            pass


def test_final_release_does_not_unlock_recycled_foreign_ofd(tmp_path):
    """The real descriptor scan must not release a recycled foreign OFD's lock."""
    path = tmp_path / "state.db"
    path.touch()
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino)
    rng = (0x40000002, 510)

    working_fd = anchor.fallback_fd(identity, str(path))
    assert working_fd is not None
    foreign_fd = -1
    try:
        os.close(working_fd)
        foreign_fd = os.open(path, os.O_RDWR)
        if foreign_fd != working_fd:
            os.dup2(foreign_fd, working_fd)
            os.close(foreign_fd)
            foreign_fd = working_fd

        assert lg._ofd_lock(foreign_fd, lg._F_RDLCK, *rng)
        assert not _foreign_exclusive_ok(str(path))

        anchor.release_final(identity, rng, lg._own_fds_for)

        assert not _foreign_exclusive_ok(str(path)), "final release unlocked the foreign OFD"
    finally:
        if foreign_fd >= 0:
            os.close(foreign_fd)


def test_reopened_writer_is_guarded_again(tmp_path, monkeypatch):
    pin_wal(monkeypatch)
    db = make_db(tmp_path / "state.db", "s", "seed")
    require_wal(db)
    db.close()
    db.append_message("s", role="user", content="after close")  # #94736 teardown/worker reopen
    try:
        assert db._conn is not None and db._wal_lock_guard, "reopen returned an unguarded writer"
        assert not _foreign_exclusive_ok(str(db.db_path))
    finally:
        db.close()
    assert _foreign_exclusive_ok(str(db.db_path))  # a true last close lifts the guard


def test_guard_never_outlives_the_handle_under_fd_reuse(tmp_path, monkeypatch):
    """A+B live -> close A (its fd number is recycled by C) -> close B: C must still be guarded,
    and once C closes nothing may be left locked."""
    pin_wal(monkeypatch)
    path = tmp_path / "state.db"
    a = make_db(path, "s", "seed")
    require_wal(a)
    b = SessionDB(db_path=path)
    a.close()
    c = SessionDB(db_path=path)
    b.close()
    try:
        assert not _foreign_exclusive_ok(str(path)), "C recorded as guarded while nothing locks"
        for name in ("state.db", "state.db-shm"):  # the stray close the guard exists for
            os.close(os.open(tmp_path / name, os.O_RDONLY))
        subprocess.run([sys.executable, "-c",
                        f"import sqlite3; c = sqlite3.connect({str(path)!r}); "
                        "c.execute('select count(*) from messages').fetchone(); c.close()"], check=True)
        c.append_message("s", role="user", content="still writes")
    finally:
        c.close()
    assert _foreign_exclusive_ok(str(path)), "a lock survived the last handle's close"
    assert not lg._HANDLES
    assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2


def test_hidden_anchor_refresh_does_not_close_recycled_sqlite_fd(tmp_path, monkeypatch):
    """Refreshing a recycled working fd leaves SQLite's replacement descriptor alone."""
    path = tmp_path / "state.db"
    fallback_fd = getattr(lg, "_fallback_fd")
    is_fallback_fd = getattr(lg, "_is_fallback_fd")
    sqlite3.connect(path).execute("CREATE TABLE entries (value TEXT)").connection.close()
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino)

    working_fd = fallback_fd(identity, str(path))
    assert working_fd is not None
    os.close(working_fd)  # Simulate an external close of the disposable duplicate only.

    conn = sqlite3.connect(path)
    held = {}
    try:
        recycled_stat = os.fstat(working_fd)
        assert (recycled_stat.st_dev, recycled_stat.st_ino) == identity, (
            "SQLite did not recycle the deliberately freed fallback fd"
        )

        # Model the anomalous no-visible-owner generation so the public refresh
        # must obtain a new working duplicate from the verified hidden anchor.
        monkeypatch.setattr(lg, "_own_fds_for", lambda identities: iter(()))
        held = lg.hold(path)
        refreshed_fd = fallback_fd(identity, str(path))
        assert refreshed_fd is not None and refreshed_fd != working_fd
        assert not is_fallback_fd(working_fd, identity), "SQLite fd was mistaken for an anchor"
        # The anchor guard mirrors a reader range, so lift this synthetic hold
        # before asking the unrelated SQLite connection to become a writer.
        lg.release(held)
        held = {}

        conn.execute("INSERT INTO entries VALUES ('still live')")
        conn.commit()
        assert conn.execute("SELECT value FROM entries").fetchone() == ("still live",)
    finally:
        lg.release(held)
        conn.close()


def test_process_lifetime_anchors_survive_last_close_during_new_guard_registration(tmp_path, monkeypatch):
    """A last close never retires anchors while a sibling is opening its guard."""
    path = tmp_path / "state.db"
    # Force the anomaly path without observing its private descriptor records.
    monkeypatch.setattr(lg, "_own_fds_for", lambda identities: iter(()))
    retiring = make_db(path, "s", "seed")
    require_wal(retiring)

    release_started = threading.Event()
    release_finished = threading.Event()
    new_connection_open = threading.Event()
    allow_last_release = threading.Event()
    allow_new_guard = threading.Event()
    original_release = lg.release
    original_hold = lg.hold
    original_close = lg.os.close
    closes_from_guard = []
    new_holder = []
    new_errors = []

    def pause_last_release(held):
        release_started.set()
        assert new_connection_open.wait(timeout=2)
        assert allow_last_release.wait(timeout=2)
        original_release(held)
        release_finished.set()

    def pause_new_guard(db_path, held=None):
        if threading.current_thread().name == "new-session":
            new_connection_open.set()
            assert allow_new_guard.wait(timeout=2)
        return original_hold(db_path, held)

    def observe_guard_close(fd):
        closes_from_guard.append(fd)
        return original_close(fd)

    monkeypatch.setattr(lg, "release", pause_last_release)
    monkeypatch.setattr(lg, "hold", pause_new_guard)
    monkeypatch.setattr(lg.os, "close", observe_guard_close)

    close_thread = threading.Thread(target=retiring.close, name="retiring-session")

    def open_new_session():
        try:
            new_holder.append(SessionDB(db_path=path))
        except BaseException as exc:  # Surface thread failures at the assertion site.
            new_errors.append(exc)

    new_thread = threading.Thread(target=open_new_session, name="new-session")
    close_thread.start()
    assert release_started.wait(timeout=2)
    new_thread.start()
    assert new_connection_open.wait(timeout=2)
    try:
        allow_last_release.set()
        assert release_finished.wait(timeout=2)
        close_thread.join(timeout=2)
        assert not close_thread.is_alive()
        assert not closes_from_guard, "last-close lifecycle retired a private anchor descriptor"

        allow_new_guard.set()
        new_thread.join(timeout=2)
        assert not new_thread.is_alive()
        assert not new_errors
        assert new_holder
        assert not _foreign_exclusive_ok(str(path)), (
            "newly registered guard did not survive the concurrent last close"
        )
    finally:
        allow_new_guard.set()
        allow_last_release.set()
        close_thread.join(timeout=2)
        new_thread.join(timeout=2)
        if new_holder:
            new_holder[0].close()

    assert _foreign_exclusive_ok(str(path)), "new handle's final close did not release the guard range"


def _foreign_switches_to_delete(path: Path) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sqlite3, sys; "
            "conn = sqlite3.connect(sys.argv[1], timeout=2); "
            "print(conn.execute('PRAGMA journal_mode=DELETE').fetchone()[0]); "
            "conn.close()",
            path,
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip().lower()


def test_relative_absolute_aliases_release_locks_on_final_close(tmp_path, monkeypatch):
    """Closing both spellings leaves SQLite free to change journal mode in another process."""
    pin_wal(monkeypatch)
    monkeypatch.chdir(tmp_path)
    relative_path = Path("state.db")
    absolute_path = tmp_path / relative_path
    relative = SessionDB(db_path=relative_path)
    absolute = SessionDB(db_path=absolute_path)
    try:
        relative.create_session("s", "cli")
        relative.append_message("s", role="user", content="relative")
        absolute.append_message("s", role="assistant", content="absolute")
    finally:
        relative.close()
        absolute.close()

    assert _foreign_switches_to_delete(absolute_path) == "delete"

    reopened = SessionDB(db_path=relative_path)
    try:
        reopened.append_message("s", role="user", content="reopened")
        assert [message["content"] for message in reopened.get_messages("s")] == [
            "relative", "absolute", "reopened",
        ]
    finally:
        reopened.close()
