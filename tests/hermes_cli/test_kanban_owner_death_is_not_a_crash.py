"""A worker killed by its owner's death must not be booked as a crash.

Measured 2026-11-04 on the Mac Studio gateway: 18 kanban runs were booked ``crashed`` across three
gateway deaths in two hours (18:40 deliberate restart, 19:38 an unclean stop, 20:14 a launchd
teardown). Every one of those ``crashed`` runs charged the card's retry budget, and the systemic
same-error rule (>= 3 identical fingerprints in one tick) tripped the breaker on the spot — the
dispatcher logged ``crashed=7 auto_blocked=6`` and ``crashed=6 auto_blocked=3``, so nine cards were
auto-blocked for a fault that was never theirs.

Two guards are pinned here:

* ``park_owned_runs`` — the shutdown path ends in-flight runs ``interrupted`` BEFORE the process
  sweep kills their workers (no crash, no budget, no breaker);
* ``_reclaim_dead_workers`` — a death the gateway could not park (SIGKILL, OOM, launchd teardown)
  is still booked ``interrupted`` when the owner is provably gone or settled the admission
  ``interrupted``, and stays ``crashed`` when the owner is alive (a real worker fault).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_shutdown_park
from hermes_cli.kanban_owner_recovery import owner_interrupted_attempt

NOW = int(time.time())
HOST = kb._claimer_id().split(":", 1)[0]
FINGERPRINT = "6a232414-2dbf-40ab-a534-d9b71db05658:116|4162240"


def _absent_pid(base: int = 3_500_000) -> int:
    pid = base
    while psutil.pid_exists(pid):
        pid += 1
    return pid


class _Board:
    """Scratch board + the admission ledger an ``owner_admitted`` receipt points at."""

    def __init__(self, root: Path):
        self.path = root / "kanban.db"
        kbc.init_db(self.path)
        self.conn = kbc.connect(self.path)
        self.ledger = root / "state.db"
        with sqlite3.connect(self.ledger) as store:
            store.execute(
                "CREATE TABLE session_admissions "
                "(target_session_id TEXT, request_id TEXT, status TEXT, outcome TEXT)"
            )

    def close(self) -> None:
        self.conn.close()

    def dead_attempt(
        self, tid: str = "t_ghost", *, worker_pid=None, owner_pid=None,
        admission=("terminal", "interrupted"), worker_lost: str | None = "EOFError",
        status: str = "running", failures: int = 0, max_retries=None,
    ) -> int:
        """A ``running`` card whose worker died under the named owner."""
        worker_pid = _absent_pid() if worker_pid is None else worker_pid
        owner_pid = _absent_pid(base=3_600_000) if owner_pid is None else owner_pid
        lock = f"{HOST}:{owner_pid}"
        started = NOW - 600
        run_id = self.conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'default', 'running', ?, ?, ?, ?)",
            (tid, lock, NOW + 3600, worker_pid, started),
        ).lastrowid
        run = int(run_id)
        self.conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, created_at, claim_lock, claim_expires, "
            "worker_pid, worker_started_at, last_heartbeat_at, current_run_id, started_at, priority, "
            "consecutive_failures, max_retries) VALUES (?, ?, 'default', ?, ?, ?, ?, ?, ?, ?, ?, ?, 5, ?, ?)",
            (tid, tid, status, NOW - 86400, lock, NOW + 3600, worker_pid, FINGERPRINT,
             NOW - 30, run, started, failures, max_retries),
        )
        request_id = f'kanban:["default","{tid}",1]'
        self.event(tid, run, "claimed", {"source_status": "ready"}, started)
        self.event(tid, run, "spawned", {"pid": worker_pid, "started_at": FINGERPRINT}, started)
        self.event(tid, run, "owner_admitted",
                   {"db": str(self.ledger), "session_id": "sess-1", "request_id": request_id,
                    "pid": owner_pid}, started + 30)
        if worker_lost:
            self.event(tid, run, "worker_lost", {"pid": worker_pid, "reason": worker_lost}, NOW - 40)
        if admission:
            with sqlite3.connect(self.ledger) as store:
                store.execute("INSERT INTO session_admissions VALUES ('sess-1', ?, ?, ?)",
                              (request_id, admission[0], admission[1]))
        self.conn.commit()
        return run

    def event(self, tid: str, run: int, kind: str, payload: dict, ts: int) -> None:
        self.conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)", (tid, run, kind, json.dumps(payload), ts))

    def state(self, tid: str) -> dict:
        return dict(self.conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())

    def run_row(self, tid: str) -> dict:
        return dict(self.conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)).fetchone())

    def event_kinds(self, tid: str) -> list[str]:
        return [r["kind"] for r in self.conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    scratch = _Board(tmp_path)
    yield scratch
    scratch.close()


# --- reclaim path: an owner-caused death books ``interrupted``, never ``crashed`` ------------

def test_owner_death_books_interrupted_without_touching_the_retry_budget(board):
    """The 20:09 cluster: the gateway died, its worker died with it, the card kept its budget."""
    board.dead_attempt("t_owner_died", failures=2)
    assert owner_interrupted_attempt(board.conn, "t_owner_died") is True

    crashed = kbd.detect_crashed_workers(board.conn, board="default")
    assert crashed == []
    assert kbd.detect_crashed_workers._last_interrupted == ["t_owner_died"]
    assert kbd.detect_crashed_workers._last_auto_blocked == []

    row, run = board.state("t_owner_died"), board.run_row("t_owner_died")
    assert (row["status"], row["claim_lock"], row["worker_pid"]) == ("ready", None, None)
    assert row["consecutive_failures"] == 2                  # untouched, not 3
    assert (run["status"], run["outcome"]) == ("interrupted", "interrupted")
    assert "interrupted" in board.event_kinds("t_owner_died")


def test_a_whole_cluster_of_owner_deaths_cannot_trip_the_breaker(board):
    """Same-error rule fires at 3 identical fingerprints — a restart must not spend nine cards."""
    for i in range(6):
        board.dead_attempt(f"t_restart_{i}")
    assert kbd.detect_crashed_workers(board.conn, board="default") == []
    for i in range(6):
        row = board.state(f"t_restart_{i}")
        assert (row["status"], row["consecutive_failures"]) == ("ready", 0)


def test_a_worker_that_died_under_a_LIVE_owner_still_books_crashed(board):
    """A real worker fault keeps its crash and its budget: the owner never went away."""
    board.dead_attempt("t_real_crash", owner_pid=os.getpid(),
                       admission=("terminal", "completed"), failures=0)
    assert owner_interrupted_attempt(board.conn, "t_real_crash") is False

    assert kbd.detect_crashed_workers(board.conn, board="default") == ["t_real_crash"]
    row, run = board.state("t_real_crash"), board.run_row("t_real_crash")
    assert (run["status"], run["outcome"]) == ("crashed", "crashed")
    assert row["consecutive_failures"] == 1
    assert "crashed" in board.event_kinds("t_real_crash")


def test_no_worker_lost_record_means_no_owner_attribution(board):
    """Evidence, not a guess: a dead pid alone is not proof the owner took the worker down."""
    board.dead_attempt("t_no_evidence", owner_pid=_absent_pid(base=3_700_000),
                       admission=None, worker_lost=None)
    assert owner_interrupted_attempt(board.conn, "t_no_evidence") is False


# --- shutdown path: park before the workers are killed -------------------------------------

def test_park_owned_runs_ends_the_run_interrupted_and_requeues(board, monkeypatch):
    """The 18:40/20:14 shape: the shutdown knows it is about to kill the worker, so it parks."""
    me = os.getpid()
    board.dead_attempt("t_mine", owner_pid=me, admission=None, worker_lost=None, failures=2)
    board.dead_attempt("t_foreign", owner_pid=_absent_pid(base=3_610_000),
                       admission=None, worker_lost=None)
    monkeypatch.setattr(kanban_shutdown_park, "_board_slugs", lambda: ["default"])
    real_connect = kbc.connect
    monkeypatch.setattr(kbc, "connect", lambda *a, **kw: real_connect(board.path))

    parked = kanban_shutdown_park.park_owned_runs("gateway restart", owner_pid=me)

    assert [p["task_id"] for p in parked] == ["t_mine"]
    assert parked[0]["retry_status"] == "ready"
    row, run = board.state("t_mine"), board.run_row("t_mine")
    assert (row["status"], row["claim_lock"], row["worker_pid"]) == ("ready", None, None)
    assert row["consecutive_failures"] == 2                  # untouched
    assert (run["status"], run["outcome"]) == ("interrupted", "interrupted")
    assert "parked_before_shutdown" in (run["metadata"] or "")
    # Somebody else's run is not ours to end.
    assert board.state("t_foreign")["status"] == "running"


def test_park_is_a_no_op_when_nothing_is_in_flight(board, monkeypatch):
    monkeypatch.setattr(kanban_shutdown_park, "_board_slugs", lambda: ["default"])
    real_connect = kbc.connect
    monkeypatch.setattr(kbc, "connect", lambda *a, **kw: real_connect(board.path))
    assert kanban_shutdown_park.park_owned_runs("gateway shutdown") == []


def test_park_never_raises_on_a_broken_board(monkeypatch):
    """A shutdown must finish even when the board cannot be written."""
    monkeypatch.setattr(kanban_shutdown_park, "_board_slugs", lambda: ["default"])

    def _boom(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kbc, "connect", _boom)
    assert kanban_shutdown_park.park_owned_runs_best_effort("gateway restart") == []
