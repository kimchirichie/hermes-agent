"""Tests for per-run model attribution (hermes_cli.kanban_model_usage).

The join this module owns — board ``owner_admitted`` event -> profile ``session_model_usage``
— is the only place the fleet can learn which model actually served a card, so the tests
pin the three things that were wrong in the first draft: which store a session is read
from, whether auxiliary calls count as a fallback, and whether a coarse (session-row)
answer stays distinguishable from a precise (per-call) one.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_model_usage as kmu


@pytest.fixture
def board(tmp_path, monkeypatch):
    """An isolated board DB with one task, claimed, plus a matching profile store."""
    home = tmp_path / ".hermes"
    (home / "profiles" / "worker").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_TASK", "HERMES_KANBAN_BOARD",
                "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    path = kb.kanban_db_path(board="default")
    kbc.init_db(path, board="default")
    return home, path


def _profile_store(home: Path, profile: str) -> sqlite3.Connection:
    """A minimal profile ``state.db`` carrying the two tables attribution reads."""
    db = home / "profiles" / profile / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT, model TEXT, billing_provider TEXT,
            billing_base_url TEXT, input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0, started_at REAL
        );
        CREATE TABLE session_model_usage (
            session_id TEXT, model TEXT, billing_provider TEXT DEFAULT '',
            billing_base_url TEXT DEFAULT '', billing_mode TEXT DEFAULT '',
            task TEXT DEFAULT '', api_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
            PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
        );
        """
    )
    return conn


def _seed_run(path: Path, *, task_id: str, model_override: str = "", session_id: str = "",
              db_hint: str = "", profile: str = "worker", body: str = "") -> int:
    conn = kbc.connect(path)
    try:
        with kb.write_txn(conn):
            kb.create_task(conn, title="t", assignee="worker", body=body,
                           workspace_kind="scratch")
            tid = conn.execute("SELECT id FROM tasks ORDER BY rowid DESC LIMIT 1").fetchone()[0]
            conn.execute("UPDATE tasks SET id = ? WHERE id = ?", (task_id, tid))
            conn.execute("UPDATE tasks SET status='running', model_override=? WHERE id=?",
                         (model_override, task_id))
            now = int(time.time())
            cur = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) VALUES (?,?,?,?)",
                (task_id, profile, "running", now),
            )
            run_id = cur.lastrowid
            payload = {"session_id": session_id, "db": db_hint}
            kb._append_event(conn, task_id, "owner_admitted", payload, run_id=run_id)
        return run_id
    finally:
        conn.close()


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize("model,base,expected", [
    ("spark-box:deepseek-flash", "https://litellm.sinbros.xyz/v1", kmu.CLASS_LOCAL),
    ("deepseek:deepseek-flash", "https://litellm.sinbros.xyz/v1", kmu.CLASS_CLOUD),
    ("deepseek-flash", "https://api.deepseek.com/v1", kmu.CLASS_CLOUD),
    ("dario:opus", "https://litellm.sinbros.xyz/v1", kmu.CLASS_SUBSCRIPTION),
    ("claude-sonnet-5", "http://127.0.0.1:3456", kmu.CLASS_SUBSCRIPTION),
    # The base URL wins over the name: a re-pointed alias must not read as local.
    ("spark-box:deepseek-flash", "https://api.deepseek.com/v1", kmu.CLASS_CLOUD),
    ("deepseek-v4-flash-vision-exp", "http://spark-a:8888/v1/", kmu.CLASS_LOCAL),
    ("mystery-model", "", kmu.CLASS_UNKNOWN),
    ("", "", kmu.CLASS_UNKNOWN),
])
def test_classify(model, base, expected):
    assert kmu.classify(model, base) == expected


# --- attribution ------------------------------------------------------------

def test_attributes_served_model_from_usage_rows(board):
    home, path = board
    store = _profile_store(home, "worker")
    store.execute("INSERT INTO sessions VALUES ('s1','kanban','spark-box:deepseek-flash','',"
                  "'https://litellm.sinbros.xyz/v1',0,0,0)")
    store.execute("INSERT INTO session_model_usage VALUES "
                  "('s1','spark-box:deepseek-flash','custom:litellm','https://litellm.sinbros.xyz/v1',"
                  "'','',11,0,0)")
    store.commit()
    run_id = _seed_run(path, task_id="t_a", session_id="s1",
                       db_hint=str(home / "profiles" / "worker" / "state.db"))
    conn = kbc.connect(path)
    try:
        rows = kmu.attribute(conn, board="default", task_id="t_a")
    finally:
        conn.close()
    assert [a.run_id for a in rows] == [run_id]
    a = rows[0]
    assert a.model == "spark-box:deepseek-flash"
    assert a.model_class == kmu.CLASS_LOCAL
    assert a.is_precise is True
    assert a.fell_back is False


def test_mid_session_fallback_is_visible(board, monkeypatch):
    """A run served by two models must report both, and count as a fallback."""
    home, path = board
    store = _profile_store(home, "worker")
    store.execute("INSERT INTO sessions VALUES ('s2','kanban','spark-box:deepseek-flash','',"
                  "'https://litellm.sinbros.xyz/v1',0,0,0)")
    store.executemany(
        "INSERT INTO session_model_usage VALUES ('s2',?,'custom:litellm',?,'','',?,0,0)",
        [("spark-box:deepseek-flash", "https://litellm.sinbros.xyz/v1", 11),
         ("deepseek:deepseek-flash", "https://litellm.sinbros.xyz/v1", 57)],
    )
    store.commit()
    monkeypatch.setattr(kmu, "lane_default_model", lambda profile: "spark-box:deepseek-flash")
    _seed_run(path, task_id="t_b", session_id="s2",
              db_hint=str(home / "profiles" / "worker" / "state.db"))
    conn = kbc.connect(path)
    try:
        a = kmu.attribute(conn, board="default", task_id="t_b")[0]
    finally:
        conn.close()
    assert a.mix == {"spark-box:deepseek-flash": 11, "deepseek:deepseek-flash": 57}
    assert a.model == "deepseek:deepseek-flash"      # dominant by call count
    assert a.model_class == kmu.CLASS_CLOUD
    assert a.fell_back is True


def test_auxiliary_model_is_not_a_fallback(board, monkeypatch):
    """Title generation runs its own model; that must not flag the card as spilled."""
    home, path = board
    store = _profile_store(home, "worker")
    store.execute("INSERT INTO sessions VALUES ('s3','kanban','spark-box:deepseek-flash','',"
                  "'https://litellm.sinbros.xyz/v1',0,0,0)")
    store.executemany(
        "INSERT INTO session_model_usage VALUES ('s3',?,'custom:litellm',?,'',?,?,0,0)",
        [("spark-box:deepseek-flash", "https://litellm.sinbros.xyz/v1", "", 40),
         ("google/gemini-3-flash-preview", "", "title_generation", 3)],
    )
    store.commit()
    monkeypatch.setattr(kmu, "lane_default_model", lambda profile: "spark-box:deepseek-flash")
    _seed_run(path, task_id="t_c", session_id="s3",
              db_hint=str(home / "profiles" / "worker" / "state.db"))
    conn = kbc.connect(path)
    try:
        a = kmu.attribute(conn, board="default", task_id="t_c")[0]
    finally:
        conn.close()
    assert a.model_class == kmu.CLASS_LOCAL
    assert a.fell_back is False


def test_session_row_fallback_when_usage_rows_absent(board, monkeypatch):
    """GM lanes keep sessions.model but no per-model rows: coarse, still attributed."""
    home, path = board
    store = _profile_store(home, "worker")
    store.execute("INSERT INTO sessions VALUES ('s4','kanban','dario:opus','custom:litellm',"
                  "'https://litellm.sinbros.xyz/v1',1000,500,0)")
    store.commit()
    monkeypatch.setattr(kmu, "lane_default_model", lambda profile: "dario:opus")
    _seed_run(path, task_id="t_d", session_id="s4", model_override="dario:opus",
              db_hint=str(home / "profiles" / "worker" / "state.db"))
    conn = kbc.connect(path)
    try:
        a = kmu.attribute(conn, board="default", task_id="t_d")[0]
    finally:
        conn.close()
    assert a.model == "dario:opus"
    assert a.model_class == kmu.CLASS_SUBSCRIPTION
    assert a.is_precise is False          # no per-call counts behind it
    assert a.total_calls == 0             # and it must not invent any


def test_unattributed_attempts_keep_their_reason(board):
    home, path = board
    _profile_store(home, "worker").close()
    _seed_run(path, task_id="t_e")                      # never reached an owner
    _seed_run(path, task_id="t_f", session_id="gone",   # admitted, session not stored
              db_hint=str(home / "profiles" / "worker" / "state.db"))
    conn = kbc.connect(path)
    try:
        rows = {a.task_id: a for a in kmu.attribute(conn, board="default")}
    finally:
        conn.close()
    assert rows["t_e"].miss == "no-session-id"
    assert rows["t_f"].miss == "session-missing"
    assert rows["t_f"].served == []


# --- reconcile --------------------------------------------------------------

def test_reconcile_writes_the_run_row_once(board, monkeypatch):
    home, path = board
    store = _profile_store(home, "worker")
    store.execute("INSERT INTO session_model_usage VALUES "
                  "('s5','deepseek:deepseek-flash','custom:litellm',"
                  "'https://litellm.sinbros.xyz/v1','','',7,0,0)")
    store.commit()
    monkeypatch.setattr(kmu, "lane_default_model", lambda profile: "spark-box:deepseek-flash")
    _seed_run(path, task_id="t_g", session_id="s5",
              db_hint=str(home / "profiles" / "worker" / "state.db"))
    conn = kbc.connect(path)
    try:
        assert kmu.reconcile(conn, board="default") == 1
        row = conn.execute("SELECT served_model, served_class, requested_model, served_mix "
                           "FROM task_runs WHERE task_id='t_g'").fetchone()
        assert row["served_model"] == "deepseek:deepseek-flash"
        assert row["served_class"] == kmu.CLASS_CLOUD
        assert row["requested_model"] == "spark-box:deepseek-flash"
        assert json.loads(row["served_mix"]) == {"deepseek:deepseek-flash": 7}
        # Idempotent: a second pass has nothing left to write.
        assert kmu.reconcile(conn, board="default") == 0
    finally:
        conn.close()


def test_reconcile_boards_creates_its_columns_and_is_idempotent(board, monkeypatch):
    """The scheduled path opens the DB itself, so it must also add the columns on a
    board that has not been through the migration yet."""
    home, path = board
    store = _profile_store(home, "worker")
    store.execute("INSERT INTO session_model_usage VALUES "
                  "('s8','spark-box:deepseek-flash','custom:litellm',"
                  "'https://litellm.sinbros.xyz/v1','','',12,0,0)")
    store.commit()
    hint = str(home / "profiles" / "worker" / "state.db")
    _seed_run(path, task_id="t_j", session_id="s8", db_hint=hint)
    conn = kbc.connect(path)
    try:
        with kb.write_txn(conn):
            for col in ("served_model", "served_class", "served_provider",
                        "served_base_url", "served_mix", "requested_model"):
                conn.execute(f"ALTER TABLE task_runs DROP COLUMN {col}")
    finally:
        conn.close()
    monkeypatch.setattr(kmu, "_board_slugs", lambda boards: ["default"])
    monkeypatch.setattr(kmu, "board_db_path", lambda slug: path)
    rows, errors = kmu.reconcile_boards()
    assert errors == []
    assert rows == 1
    assert kmu.reconcile_boards() == (0, [])


def test_report_separates_run_share_from_call_share(board, monkeypatch):
    home, path = board
    store = _profile_store(home, "worker")
    store.execute("INSERT INTO session_model_usage VALUES "
                  "('s6','spark-box:deepseek-flash','custom:litellm',"
                  "'https://litellm.sinbros.xyz/v1','','',30,0,0)")
    store.execute("INSERT INTO sessions VALUES ('s7','kanban','dario:opus','custom:litellm',"
                  "'https://litellm.sinbros.xyz/v1',0,0,0)")
    store.commit()
    monkeypatch.setattr(kmu, "lane_default_model", lambda profile: "spark-box:deepseek-flash")
    hint = str(home / "profiles" / "worker" / "state.db")
    _seed_run(path, task_id="t_h", session_id="s6", db_hint=hint)
    _seed_run(path, task_id="t_i", session_id="s7", db_hint=hint)
    rep = kmu.report(days=1, boards=["default"])
    assert rep["runs_attributed"] == 2
    assert rep["runs_with_call_counts"] == 1
    assert rep["runs_by_class"] == {kmu.CLASS_LOCAL: 1, kmu.CLASS_SUBSCRIPTION: 1}
    # Only the precise run contributes calls, so the share is not diluted by the coarse one.
    assert rep["calls_by_class"] == {kmu.CLASS_LOCAL: 30}
    assert rep["local_share_calls"] == 1.0
    assert rep["local_share_runs"] == 0.5


# --- override audit ---------------------------------------------------------

def test_audit_flags_unexplained_pins_and_skips_the_lane_default(board, monkeypatch):
    _home, path = board
    monkeypatch.setattr(kmu, "lane_default_model", lambda profile: "spark-box:deepseek-flash")
    conn = kbc.connect(path)
    try:
        with kb.write_txn(conn):
            kb.create_task(conn, title="pinned with reason", assignee="worker",
                           body="time-sensitive: production hotfix", workspace_kind="scratch",
                           model_override="deepseek:deepseek-flash")
            with_reason = conn.execute("SELECT id FROM tasks ORDER BY rowid DESC LIMIT 1").fetchone()[0]
            kb.create_task(conn, title="pinned silently", assignee="worker", body="",
                           workspace_kind="scratch", model_override="deepseek:deepseek-pro")
            silent = conn.execute("SELECT id FROM tasks ORDER BY rowid DESC LIMIT 1").fetchone()[0]
            kb.create_task(conn, title="pinned to its own default", assignee="worker", body="",
                           workspace_kind="scratch", model_override="spark-box:deepseek-flash")
            defaulted = conn.execute("SELECT id FROM tasks ORDER BY rowid DESC LIMIT 1").fetchone()[0]
    finally:
        conn.close()
    found = {p["task_id"]: p for p in kmu.audit_overrides(boards=["default"], lanes=["worker"])}
    assert found[with_reason]["unexplained"] is False
    assert found[with_reason]["reason"] == "time-sensitive"
    assert found[silent]["unexplained"] is True
    assert found[silent]["pinned_class"] == kmu.CLASS_CLOUD
    assert defaulted not in found
