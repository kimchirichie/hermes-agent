"""The decomposer must work on the board it is TOLD, not the ambient one.

``board=`` is what the gateway's auto-decompose tick passes so it never has to
pin ``HERMES_KANBAN_BOARD`` process-wide. These tests pin the contract: with the
``current`` pointer on ``default``, a decomposer run scoped to ``personal``
reads and writes ``personal`` and leaves ``default`` alone.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decompose_mod


@pytest.fixture
def two_boards(tmp_path, monkeypatch):
    home = tmp_path / "kanban-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK"):
        monkeypatch.delenv(key, raising=False)
    kb.init_db()
    kb.create_board("personal")
    kb.set_current_board("default")
    assert kb.get_current_board() == "default"
    return home


def _triage_on(board: str, title: str = "rough idea") -> str:
    with kbc.connect(board=board) as conn:
        return kb.create_task(conn, title=title, triage=True)


def test_list_triage_ids_reads_the_named_board_only(two_boards):
    personal_tid = _triage_on("personal")
    default_tid = _triage_on("default")

    assert decompose_mod.list_triage_ids() == [default_tid]
    assert decompose_mod.list_triage_ids(board="personal") == [personal_tid]


def test_decompose_task_writes_the_named_board_not_the_current_one(two_boards, monkeypatch):
    tid = _triage_on("personal")

    monkeypatch.setattr(
        decompose_mod, "_load_routing",
        lambda **kw: decompose_mod._Routing(
            orchestrator="mumu", default_assignee="mumu", auto_promote=True,
            roster=[], valid_names={"mumu"}),
    )
    monkeypatch.setattr(
        decompose_mod, "_call_aux",
        lambda *a, **k: (json.dumps({"fanout": False, "title": "tightened", "body": "now a spec"}), ""),
    )

    outcome = decompose_mod.decompose_task(tid, author="tester", board="personal")
    assert outcome.ok, outcome.reason

    with kbc.connect(board="personal") as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status != "triage"
    # Nothing was mirrored onto the board the `current` pointer names.
    with kbc.connect() as conn:
        assert kb.get_task(conn, tid) is None
