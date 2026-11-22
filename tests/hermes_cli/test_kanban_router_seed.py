"""A group cap starts at max(running cards, router in-flight count); unreadable means full."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_cli import kanban_router_seed as rs
from tests.hermes_cli.test_kanban_group_caps import GROUP, _dry, _mark_running, kbenv

MODEL = "spark-box:deepseek-flash"


@pytest.fixture()
def router():
    state = {"body": None, "status": 200}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(state["status"])
            self.end_headers()
            self.wfile.write(json.dumps(state["body"]).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}/spark/inflight"
    yield state
    srv.shutdown()


def _body(n, ts=None):
    return {"in_flight": n, "groups": {MODEL: n}, "ts": time.time() if ts is None else ts}


def test_reads_fresh_count(router):
    router["body"] = _body(2)
    assert rs.read_router_count(router["url"], MODEL) == 2


@pytest.mark.parametrize("body", [
    {"in_flight": 1, "groups": {}, "ts": 1e18},
    {"groups": {MODEL: "x"}, "ts": 1e18},
    {"groups": {MODEL: -1}, "ts": 1e18},
    {"groups": {MODEL: True}, "ts": 1e18},
    "garbage",
])
def test_bad_body_is_unreadable(router, body):
    router["body"] = body
    assert rs.read_router_count(router["url"], MODEL) is None


def test_stale_reading_is_unreadable(router):
    router["body"] = _body(1, ts=time.time() - 500)
    assert rs.read_router_count(router["url"], MODEL) is None


def test_unreachable_is_unreadable():
    assert rs.read_router_count("http://127.0.0.1:9/x", MODEL) is None


def test_group_seeds_unreadable_seeds_the_group_full(router):
    router["status"] = 500
    caps = {"spark": {"profiles": ["a"], "max": 3, "router_seed": {"url": router["url"], "model": MODEL}}}
    assert rs.group_seeds(caps) == {"spark": 3}


def test_group_without_router_seed_is_untouched():
    assert rs.group_seeds({"spark": {"profiles": ["a"], "max": 3}}) == {}


def test_router_count_above_cards_blocks_spawns(kbenv):
    kb, kbc, kbd = kbenv
    with kbc.connect_closing() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
    res = _dry(kbd, kbc, group_caps=GROUP, group_seed={"spark": 2})
    assert res.spawned == []


def test_router_count_below_cards_does_not_loosen(kbenv):
    kb, kbc, kbd = kbenv
    with kbc.connect_closing() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
    res = _dry(kbd, kbc, group_caps=GROUP, group_seed={"spark": 0})
    assert len(res.spawned) == 2


def test_partial_router_load_leaves_the_remainder(kbenv):
    kb, kbc, kbd = kbenv
    with kbc.connect_closing() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
    res = _dry(kbd, kbc, group_caps=GROUP, group_seed={"spark": 1})
    assert len(res.spawned) == 1


def test_urgent_card_ignores_the_router_but_not_the_card_cap(kbenv):
    kb, kbc, kbd = kbenv
    with kbc.connect_closing() as conn:
        urgent = kb.create_task(conn, title="rev", assignee="alpha", priority=8)
        kb.create_task(conn, title="normal", assignee="alpha", priority=0)
    res = _dry(kbd, kbc, group_caps=GROUP, group_seed={"spark": 2})
    assert [t[0] for t in res.spawned] == [urgent]


def test_refused_card_does_not_take_a_slot(kbenv):
    kb, kbc, kbd = kbenv
    with kbc.connect_closing() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"g{i}", assignee="gamma")
        kb.create_task(conn, title="a", assignee="alpha")
    res = _dry(kbd, kbc, group_caps=GROUP, group_seed={"spark": 1})
    assert len(res.spawned) >= 1


def test_seed_is_used_up_across_boards(kbenv):
    kb, kbc, kbd = kbenv
    for board in ("default", "other"):
        with kbc.connect_closing(board=board) as conn:
            for i in range(3):
                kb.create_task(conn, title=f"{board}{i}", assignee="alpha")
    seed = {"spark": 1}
    spawned = 0
    for board in ("default", "other"):
        with kbc.connect_closing(board=board) as conn:
            res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 12345, board=board,
                                    group_caps=GROUP, group_seed=seed)
            spawned += len(res.spawned)
            for task_id, _who, _ws in res.spawned:
                _mark_running(kb, kbc, board, task_id)
    assert spawned == 1


def test_seed_dict_is_shared_by_the_watcher(monkeypatch):
    from gateway.kanban_watchers_dispatcher import _KanbanDispatcher
    seen = []

    class FakeKb:
        pass

    d = _KanbanDispatcher.__new__(_KanbanDispatcher)
    d.settings = type("S", (), {"group_caps": None})()
    d.group_seed = {}
    d._board_slugs = lambda: ["a", "b"]
    d.tick_once_for_board = lambda slug: seen.append(d.group_seed) or None
    d.tick_once()
    assert seen[0] is seen[1]


def test_future_timestamp_is_unreadable(router):
    router["body"] = _body(1, ts=time.time() + 500)
    assert rs.read_router_count(router["url"], MODEL) is None


def test_bad_max_skips_the_group_instead_of_raising(router):
    router["body"] = _body(1)
    caps = {"spark": {"profiles": ["a"], "max": "x", "router_seed": {"url": router["url"], "model": MODEL}}}
    assert rs.group_seeds(caps) == {}


def test_router_seed_true_is_ignored_and_warns_once(caplog):
    rs._warned_malformed.clear()
    caps = {"spark": {"profiles": ["a"], "max": 3, "router_seed": True}}
    with caplog.at_level("WARNING"):
        assert rs.group_seeds(caps) == {}
        rs.group_seeds(caps)
    assert sum("not {url, model}" in r.message for r in caplog.records) == 1


def test_out_of_tick_callers_get_caps_and_seed(monkeypatch, router):
    router["body"] = _body(2)
    caps = {"spark": {"profiles": ["a"], "max": 3, "router_seed": {"url": router["url"], "model": MODEL}}}
    monkeypatch.setattr("hermes_cli.kanban_fleet_caps.fleet_caps", lambda: (None, caps))
    got_caps, seed = rs.fleet_group_caps_and_seed()
    assert got_caps is caps and seed == {"spark": 2}


def test_boards_with_the_most_urgent_ready_card_go_first(kbenv):
    from gateway.kanban_watchers_dispatcher import _KanbanDispatcher
    kb, kbc, _kbd = kbenv
    with kbc.connect_closing(board="default") as conn:
        kb.create_task(conn, title="low", assignee="alpha", priority=0)
    with kbc.connect_closing(board="other") as conn:
        kb.create_task(conn, title="high", assignee="alpha", priority=5)
    d = _KanbanDispatcher.__new__(_KanbanDispatcher)
    d._board_slugs = lambda: ["default", "other"]
    assert d._boards_by_urgency() == ["other", "default"]


def test_router_seed_still_takes_the_max_for_a_model_group(kbenv):
    kb, kbc, kbd = kbenv
    with kbc.connect_closing() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"m{i}", assignee="alpha", model_override=MODEL)
    res = _dry(kbd, kbc, group_caps={"spark": {"model": MODEL, "max": 2}}, group_seed={"spark": 1})
    assert len(res.spawned) == 1
