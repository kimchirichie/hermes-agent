"""The auto-decompose tick must never pin the board process-wide.

Regression for a live misroute: the tick used to write
``os.environ["HERMES_KANBAN_BOARD"] = slug`` around each board's decomposer run.
``os.environ`` is process-wide, not thread-local, and the tick runs in a worker
thread of the gateway that serves every live session — so for as long as the
aux LLM call lasted (up to its 180s timeout) a boardless ``kanban_create`` in any
concurrent chat turn resolved the ENV pin instead of the ``current`` file and
filed its card on the decomposer's board (observed: didi, didi, personal while
``current`` read ``default``).

The board is now passed explicitly; the environment must stay untouched.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from gateway import kanban_watchers_dispatcher as kwd


def _dispatcher():
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def test_auto_decompose_tick_leaves_process_env_untouched(monkeypatch):
    import hermes_cli

    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default", "personal"])

    seen = {"env": [], "boards": []}

    def fake_list_triage(**kw):
        seen["env"].append(os.environ.get("HERMES_KANBAN_BOARD"))
        seen["boards"].append(kw.get("board"))
        return ["t1"] if kw.get("board") == "personal" else []

    def fake_decompose(task_id, author=None, board=None):
        # Read the env at the exact moment the decomposer runs — this is the
        # window a concurrent chat turn would have been resolving through.
        seen["env"].append(os.environ.get("HERMES_KANBAN_BOARD"))
        seen["boards"].append(board)
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    fake = SimpleNamespace(list_triage_ids=fake_list_triage, decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    assert _dispatcher().auto_decompose_tick(5) == 1

    # The env stayed clear the whole way through.
    assert seen["env"] == [None, None, None]
    assert "HERMES_KANBAN_BOARD" not in os.environ
    # Each board was named explicitly, and the decomposer got the board it came from.
    assert seen["boards"] == ["default", "personal", "personal"]


def test_auto_decompose_tick_restores_a_preexisting_env_pin(monkeypatch):
    """A pin that came from the process start (CLI/TUI) is left exactly as it was."""
    import hermes_cli

    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["personal"])

    seen = {"env": []}

    def fake_decompose(task_id, author=None, board=None):
        seen["env"].append(os.environ.get("HERMES_KANBAN_BOARD"))
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    fake = SimpleNamespace(
        list_triage_ids=lambda **kw: ["t1"], decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    assert _dispatcher().auto_decompose_tick(5) == 1
    assert seen["env"] == ["default"]
    assert os.environ["HERMES_KANBAN_BOARD"] == "default"
