"""Tests for the multi-board kanban layer (``hermes kanban boards …``).

Covers the pieces added when boards became a first-class concept:

* Slug validation and normalisation.
* Path resolution for ``default`` (legacy ``<root>/kanban.db``) vs
  named boards (``<root>/kanban/boards/<slug>/kanban.db``).
* Current-board persistence via ``<root>/kanban/current`` and
  ``HERMES_KANBAN_BOARD`` env var.
* ``connect(board=)`` isolation — writes on one board don't leak.
* ``create_board`` / ``list_boards`` / ``remove_board`` round trip.
* CLI surface: ``hermes kanban boards list/create/switch/rm``.
* ``_default_spawn`` injects ``HERMES_KANBAN_BOARD`` into worker env.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure the worktree (not the stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior kanban state.

    The autouse hermetic conftest already nukes credentials + TZ; this
    fixture layers a per-test HERMES_HOME plus a path-init cache reset
    so each test sees a truly empty board set.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    # Also reset hermes_constants cache so get_default_hermes_root() re-reads.
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    # Kanban module-level init cache must not leak between tests.
    kb._INITIALIZED_PATHS.clear()
    return home


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------

class TestSlugValidation:
    @pytest.mark.parametrize("good", [
        "default", "atm10-server", "hermes-agent", "proj_1", "a",
        "very-long-but-still-ok-slug-with-hyphens-and-numbers-1234",
    ])
    def test_accepts_valid(self, good):
        assert kb._normalize_board_slug(good) == good


    def test_empty_returns_none(self):
        assert kb._normalize_board_slug(None) is None
        assert kb._normalize_board_slug("") is None
        assert kb._normalize_board_slug("   ") is None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestPathResolution:
    def test_default_board_legacy_path(self, fresh_home):
        """The default board's DB lives at ``<root>/kanban.db`` for back-compat."""
        assert kb.kanban_db_path() == fresh_home / "kanban.db"
        assert kb.kanban_db_path(board="default") == fresh_home / "kanban.db"

    def test_named_board_under_boards_dir(self, fresh_home):
        p = kb.kanban_db_path(board="atm10-server")
        assert p == fresh_home / "kanban" / "boards" / "atm10-server" / "kanban.db"


    def test_env_var_db_override_still_wins(self, fresh_home, tmp_path, monkeypatch):
        """``HERMES_KANBAN_DB`` pins the file regardless of ``board=`` arg for every
        execution the dispatcher fences (the 5ec6baa multi-boards isolation: workers
        physically cannot see other boards): its dispatched workers (``HERMES_KANBAN_TASK``)
        and delegated children / descendants (``HERMES_DELEGATED_CHILD_CONTEXT``). An
        explicit board that outranked the pin would also escape
        ``kanban_path_is_fenced``, which checks the pinned path / fenced root. Outside
        those fences an explicit board is the caller's own intent and wins (see
        ``test_explicit_board_trumps_env_var_db_override`` below)."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path() == forced
        assert kb.kanban_db_path(board=None) == forced
        # Dispatched worker identity: the pin still fences explicit board intent.
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fence_probe")
        assert kb.kanban_db_path(board="ignored") == forced
        # Delegated children / spawned descendants are fenced the same way.
        from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        assert kb.kanban_db_path(board="ignored") == forced

    def test_env_var_db_override_wins_when_board_not_passed(self, fresh_home, tmp_path, monkeypatch):
        """``HERMES_KANBAN_DB`` pins the file when no explicit ``board=`` is given
        (back-compat for dispatcher-spawned workers with no board override)."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path() == forced
        assert kb.kanban_db_path(board=None) == forced

    def test_explicit_board_trumps_env_var_db_override(self, fresh_home, tmp_path, monkeypatch):
        """Documented priority (module docstring, predates this test): explicit
        ``board=`` arg > ``HERMES_KANBAN_BOARD`` > ``HERMES_KANBAN_DB`` > current >
        default — for UNFENCED callers. An explicit ``board=`` must resolve to that
        board's own path even when ``HERMES_KANBAN_DB`` pins a different file: this is
        what makes cross-board ``kanban_create(board=...)`` / ``kanban_show(board=...)``
        work from a user-facing session instead of silently landing on the pinned board
        (t_3f1c63a5). Fenced callers (dispatched workers, delegated children) keep the
        pinned path — see ``test_env_var_db_override_still_wins`` above."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        p = kb.kanban_db_path(board="atm10-server")
        assert p == fresh_home / "kanban" / "boards" / "atm10-server" / "kanban.db"
        assert p != forced


# ---------------------------------------------------------------------------
# Current-board resolution
# ---------------------------------------------------------------------------

class TestCurrentBoard:



    def test_stale_file_pointer_falls_back_to_default(self, fresh_home):
        current = fresh_home / "kanban" / "current"
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text("missing-board\n", encoding="utf-8")

        assert kb.get_current_board() == "default"
        assert not kb.board_exists("missing-board")
        assert [b["slug"] for b in kb.list_boards()] == ["default"]



    def test_kanban_db_path_reads_current(self, fresh_home):
        """kanban_db_path() with no args respects the on-disk pointer."""
        kb.create_board("my-proj")
        kb.set_current_board("my-proj")
        expected = fresh_home / "kanban" / "boards" / "my-proj" / "kanban.db"
        assert kb.kanban_db_path() == expected


# ---------------------------------------------------------------------------
# Board CRUD
# ---------------------------------------------------------------------------

class TestBoardCRUD:






    @pytest.mark.parametrize("archive", [True, False])
    def test_remove_clears_init_cache_for_recreated_db(self, fresh_home, archive):
        # Regression for #23833: a poll loop that re-creates a just-removed
        # board must get a fresh schema-init pass — if _INITIALIZED_PATHS
        # still contained the resolved path, the CREATE TABLE pass would be
        # skipped and downstream readers hit `no such table: task_events`.
        # (Since #43243 connect() itself refuses to recreate a removed board,
        # so the re-creation goes through create_board, as it should.)
        kb.create_board("recycle")
        # First connect populates _INITIALIZED_PATHS for this DB.  Use
        # connect_closing: `with connect() as conn` does NOT close the fd, and
        # on Windows an open connection locks kanban.db so remove_board's
        # rename below fails with WinError 5/32.
        with kbc.connect_closing(board="recycle") as conn:
            kb.create_task(conn, title="t1", assignee="dev")
        db_path = kb.board_dir("recycle") / "kanban.db"
        assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

        kb.remove_board("recycle", archive=archive)
        # remove_board must drop the cache entry so a re-create through
        # connect() gets a fresh schema-init pass.
        assert str(db_path.resolve()) not in kb._INITIALIZED_PATHS

        # Simulate the board being re-created at the same slug: the schema
        # must be re-applied on the fresh DB file.
        kb.create_board("recycle")
        with kbc.connect_closing(board="recycle") as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert "task_events" in tables
        assert "tasks" in tables



# ---------------------------------------------------------------------------
# Connection isolation
# ---------------------------------------------------------------------------

class TestConnectionIsolation:
    def test_tasks_do_not_leak_across_boards(self, fresh_home):
        kb.create_board("alpha")
        kb.create_board("beta")

        with kbc.connect(board="alpha") as conn:
            kb.create_task(conn, title="alpha-task-1", assignee="dev")
            kb.create_task(conn, title="alpha-task-2", assignee="dev")

        with kbc.connect(board="beta") as conn:
            kb.create_task(conn, title="beta-only", assignee="dev")

        with kbc.connect(board="alpha") as conn:
            a = kb.list_tasks(conn)
        with kbc.connect(board="beta") as conn:
            b = kb.list_tasks(conn)
        with kbc.connect(board="default") as conn:
            d = kb.list_tasks(conn)

        assert {t.title for t in a} == {"alpha-task-1", "alpha-task-2"}
        assert {t.title for t in b} == {"beta-only"}
        assert d == []

    def test_connect_without_args_uses_current(self, fresh_home):
        kb.create_board("curr")
        kb.set_current_board("curr")
        with kbc.connect() as conn:
            kb.create_task(conn, title="implicit", assignee="x")
        with kbc.connect(board="curr") as conn:
            tasks = kb.list_tasks(conn)
        assert [t.title for t in tasks] == ["implicit"]

    def test_connect_env_var_overrides_current(self, fresh_home, monkeypatch):
        kb.create_board("persist")
        kb.create_board("envwin")
        kb.set_current_board("persist")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "envwin")
        with kbc.connect() as conn:
            kb.create_task(conn, title="via-env", assignee="x")
        with kbc.connect(board="envwin") as conn:
            assert [t.title for t in kb.list_tasks(conn)] == ["via-env"]
        with kbc.connect(board="persist") as conn:
            assert kb.list_tasks(conn) == []


# ---------------------------------------------------------------------------
# Worker spawn env injection
# ---------------------------------------------------------------------------

class TestWorkerSpawnEnv:
    """Ensure the dispatcher pins ``HERMES_KANBAN_BOARD`` / DB / workspaces on spawn.

    We monkey-patch ``subprocess.Popen`` to capture the child env without
    actually spawning anything.
    """

    def test_default_spawn_sets_env_vars(self, fresh_home, monkeypatch):
        captured = {}

        class FakeProc:
            pid = 12345

        def fake_popen(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env", {})
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        kb.create_board("spawntest")

        task = kb.Task(
            id="t_abc",
            title="worker test",
            body=None,
            assignee="teknium",
            status="ready",
            priority=0,
            created_by="user",
            created_at=0,
            started_at=None,
            completed_at=None,
            workspace_kind="scratch",
            workspace_path=None,
            claim_lock=None,
            claim_expires=None,
            tenant=None,
        )

        kbd._default_spawn(task, str(fresh_home / "ws"), board="spawntest")

        env = captured["env"]
        assert env["HERMES_KANBAN_BOARD"] == "spawntest"
        assert env["HERMES_KANBAN_TASK"] == "t_abc"
        # DB path should match the per-board DB, not the legacy default.
        expected_db = fresh_home / "kanban" / "boards" / "spawntest" / "kanban.db"
        assert env["HERMES_KANBAN_DB"] == str(expected_db)
        expected_ws = fresh_home / "kanban" / "boards" / "spawntest" / "workspaces"
        assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(expected_ws)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def _cli(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``hermes kanban …`` with PYTHONPATH pinned to the worktree.

    The kanban env pins are stripped first: a test process that inherits a
    dispatcher's ``HERMES_KANBAN_DB``/``HERMES_KANBAN_BOARD`` would run every
    ``--board`` assertion against a live board instead of the temp home.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_KANBAN_")}
    env["PYTHONPATH"] = str(_WORKTREE)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban"] + args,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_WORKTREE),
        timeout=30,
        check=False,
    )


class TestCLI:


    def test_per_board_task_isolation_via_cli(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "projA"], env_extra=env).returncode == 0
        assert _cli(["boards", "create", "projB"], env_extra=env).returncode == 0

        # Create one task on each via --board.
        r = _cli(["--board", "projA", "create", "Task A", "--assignee", "dev"], env_extra=env)
        assert r.returncode == 0, r.stderr
        r = _cli(["--board", "projB", "create", "Task B", "--assignee", "dev"], env_extra=env)
        assert r.returncode == 0, r.stderr

        # list on each board only shows its own.
        listA = _cli(["--board", "projA", "list", "--json"], env_extra=env)
        listB = _cli(["--board", "projB", "list", "--json"], env_extra=env)
        listD = _cli(["list", "--json"], env_extra=env)

        titlesA = [t["title"] for t in json.loads(listA.stdout)]
        titlesB = [t["title"] for t in json.loads(listB.stdout)]
        titlesD = [t["title"] for t in json.loads(listD.stdout)]

        assert titlesA == ["Task A"]
        assert titlesB == ["Task B"]
        assert titlesD == []





# ---------------------------------------------------------------------------
# Archived / deleted board resurrection (#43243)
# ---------------------------------------------------------------------------

class TestBoardResurrection:
    """Read/watch paths must not resurrect archived or deleted boards.

    Stale dashboard tabs, gateway notifiers and event-stream pollers can hand
    ``connect(board=<slug>)`` a slug whose board was archived (moved to
    ``_archived/`` with an ``archived`` tombstone) or hard-deleted
    (``rmtree``).  ``connect`` must refuse to recreate the directory/DB, and
    ``list_boards`` must not surface a DB-only stub as an active board.
    """

    def test_connect_does_not_recreate_archived_board(self, fresh_home):
        kb.create_board("gone")
        kb.remove_board("gone", archive=True)
        tombstone = kb.board_metadata_path("gone")
        assert tombstone.exists() and kb.read_board_metadata("gone")["archived"] is True
        with pytest.raises(ValueError, match="archived"):
            kbc.connect(board="gone")
        assert not (kb.board_dir("gone") / "kanban.db").exists()

    def test_connect_does_not_recreate_deleted_board(self, fresh_home):
        kb.create_board("nuked")
        kb.remove_board("nuked", archive=False)
        assert not kb.board_dir("nuked").exists()
        with pytest.raises(ValueError, match="does not exist"):
            kbc.connect(board="nuked")
        assert not kb.board_dir("nuked").exists()

    def test_connect_still_opens_live_board(self, fresh_home):
        kb.create_board("live")
        with kbc.connect_closing(board="live") as conn:
            conn.execute("SELECT 1").fetchone()
        assert kb.board_exists("live")

    def test_list_boards_ignores_db_only_stub(self, fresh_home):
        # Simulate the historical stub shape: kanban.db with no board.json.
        stub = kb.board_dir("ghost")
        stub.mkdir(parents=True)
        (stub / "kanban.db").touch()
        assert "ghost" not in [b["slug"] for b in kb.list_boards()]
        assert "ghost" not in [b["slug"] for b in kb.list_boards(include_archived=True)]

    def test_archive_tombstone_keeps_name(self, fresh_home):
        kb.create_board("keepname", name="My Board")
        kb.remove_board("keepname", archive=True)
        meta = kb.read_board_metadata("keepname")
        assert meta["archived"] is True
        assert meta["name"] == "My Board"
        assert "keepname" in [b["slug"] for b in kb.list_boards(include_archived=True)]
        assert "keepname" not in [b["slug"] for b in kb.list_boards(include_archived=False)]


# ---------------------------------------------------------------------------
# Schema-less husks: an existing DB that cannot serve `tasks` is not "empty"
# ---------------------------------------------------------------------------

def _make_husk(path: Path) -> None:
    """Write the exact shape an init leaves when it dies before SCHEMA_SQL.

    ``journal_mode=WAL`` materializes page 1, so the file is a valid 4096-byte
    SQLite database with zero tables — indistinguishable from a real board to
    everything except a table count.
    """
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)  # a real board DB would otherwise keep its schema
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.commit()
    finally:
        conn.close()
    assert path.stat().st_size > 0


class TestSchemaLessHusk:
    """A board DB with no schema must fail loudly, never list zero tasks."""

    def test_probe_matches_aborted_init_shape(self, tmp_path):
        husk = tmp_path / "kanban.db"
        _make_husk(husk)
        assert kbc._schema_less_husk_reason(husk) is not None

    def test_probe_ignores_zero_byte_db(self, tmp_path):
        empty = tmp_path / "kanban.db"
        empty.touch()
        assert kbc._schema_less_husk_reason(empty) is None

    def test_probe_ignores_real_board(self, fresh_home):
        kb.create_board("real")
        assert kbc._schema_less_husk_reason(kb.board_dir("real") / "kanban.db") is None

    def test_connect_refuses_husk_and_leaves_it_alone(self, fresh_home):
        husk = fresh_home / "kanban.db"
        _make_husk(husk)
        before = husk.read_bytes()
        with pytest.raises(kbc.KanbanDbHuskError, match="no schema"):
            kbc.connect(board="default")
        assert husk.read_bytes() == before, "the husk must not be 'repaired' into an empty board"

    def test_connect_still_initializes_a_fresh_board(self, fresh_home):
        with kbc.connect_closing(board=None) as conn:
            assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0

    def test_create_board_initializes_through_zero_byte_file(self, fresh_home):
        kb.create_board("fresh")
        with kbc.connect_closing(board="fresh") as conn:
            assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0

    def test_cli_list_marks_husk_unreadable_and_keeps_other_boards(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "healthy"], env_extra=env).returncode == 0
        _make_husk(tmp_path / "kanban" / "boards" / "healthy" / "kanban.db")

        r = _cli(["boards", "list"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "UNREADABLE" in r.stdout
        # The healthy sibling must still render — one broken board cannot hide the rest.
        assert "default" in r.stdout

    def test_cli_list_husk_is_not_reported_empty(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "husked"], env_extra=env).returncode == 0
        _make_husk(tmp_path / "kanban" / "boards" / "husked" / "kanban.db")

        r = _cli(["boards", "list", "--json"], env_extra=env)
        assert r.returncode == 0, r.stderr
        rows = {b["slug"]: b for b in json.loads(r.stdout)}
        assert rows["husked"]["counts"] == {}
        assert rows["husked"]["counts_error"], "an unreadable board must carry its error, not read as empty"

    def test_dashboard_boards_payload_carries_counts_error(self, fresh_home):
        from plugins.kanban.dashboard import plugin_api

        counts, error = plugin_api._board_counts("default")
        assert counts == {} and error is None  # no DB on disk yet: genuinely empty


class TestCreateNamesItsBoard:
    """``create`` must say where the card landed and flag a cross-board filing."""

    def test_create_prints_board_slug(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        r = _cli(["create", "Filed card", "--assignee", "dev"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "board=default" in r.stdout
        # default is the active board (no `current` file), so no cross-board note.
        assert "active board is" not in r.stderr

    def test_create_off_the_active_board_says_so(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "work"], env_extra=env).returncode == 0
        assert _cli(["boards", "switch", "work"], env_extra=env).returncode == 0

        r = _cli(["--board", "default", "create", "Off-board card", "--assignee", "dev"], env_extra=env)
        assert r.returncode == 0, r.stderr
        assert "board=default" in r.stdout
        assert "active board is 'work'" in r.stderr
        assert "boards switch default" in r.stderr

