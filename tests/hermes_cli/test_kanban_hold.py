"""The holding pattern: hold cloud-lane cards on create, release them on the nightly drain.

Two properties carry the whole design, and most of these tests exist to keep them true:

* a held card is the ONLY thing the drain will release — ``scheduled`` is a shared column
  full of timed holds ("WAKE: ... 30-day gate", "after the snapshot limit resets"), and
  running one of those unattended at 03:00 is the failure this feature must not introduce;
* the cap counts what is actually occupying a lane, recomputed after every release, so a
  tick can never overshoot it and a card that lands ``todo`` costs no slot.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_hold as kh

CLOUD = ("ms-worker", "cs-worker", "oap-worker", "kiwii-worker",
         "ms-reviewer", "cs-reviewer", "oap-reviewer", "kiwii-reviewer", "sop-reviewer")


def _write_config(home: Path, kanban: dict) -> None:
    """``config.yaml`` with a kanban section, written before anything loads it."""
    lines = ["kanban:"]
    for key, value in kanban.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"  {key}: [{', '.join(str(v) for v in value)}]")
        else:
            lines.append(f"  {key}: {value}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp board with the fleet's hold policy: nine cloud lanes, priority 8 escapes."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_config(home, {
        "hold_profiles": list(CLOUD),
        "hold_exempt_priority": 8,
        "drain_cap": 4,
    })
    kb.init_db()
    return home


def _create(**kw) -> str:
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, **kw)


def _task(tid: str):
    with kbc.connect_closing() as conn:
        return kb.get_task(conn, tid)


def _status(tid: str) -> str:
    return _task(tid).status


def _events(tid: str) -> list:
    with kbc.connect_closing() as conn:
        return kb.list_events(conn, tid)


def _held_event(tid: str):
    hits = [e for e in _events(tid) if e.kind == kh.HOLD_EVENT]
    return hits[-1] if hits else None


def _release(cap: int = 4, dry_run: bool = False) -> kh.ReleaseResult:
    with kbc.connect_closing() as conn:
        return kh.release_due(conn, cap=cap, dry_run=dry_run)


# --- A. hold on create -------------------------------------------------------------------


def test_a_cloud_lane_card_is_born_scheduled_and_says_why(board):
    tid = _create(title="KiwiiDev gate.yml: a dismissed approval leaves a stale run",
                  assignee="kiwii-worker")
    assert _status(tid) == "scheduled"
    event = _held_event(tid)
    assert event is not None, "a held card must carry a held_for_drain event"
    assert "kiwii-worker" in (event.payload or {}).get("reason", "")


def test_the_orchestrator_and_its_gms_keep_running_live(board):
    """Rick: hold ONLY the cloud lanes. Mumu, the GMs and the local lanes run live."""
    for lane in ("mumu", "ani", "sarah", "gm-cs", "gm-ms", "gm-oap", "gm-kiwii",
                 "didi", "mockup", "worker", "reviewer"):
        tid = _create(title=f"{lane} work that must not wait for 03:00", assignee=lane)
        assert _status(tid) == "ready", lane
        assert _held_event(tid) is None, lane


def test_every_cloud_lane_is_held(board):
    for lane in CLOUD:
        tid = _create(title=f"{lane}: something to do overnight", assignee=lane)
        assert _status(tid) == "scheduled", lane


def test_an_urgent_priority_escapes_the_hold(board):
    """Guard (ii): `--priority 8` means "run now", not "run at 03:00"."""
    urgent = _create(title="Cloud lane emergency", assignee="ms-worker", priority=8)
    normal = _create(title="Cloud lane routine", assignee="ms-worker", priority=7)
    assert _status(urgent) == "ready"
    assert _status(normal) == "scheduled"


def test_a_revision_for_a_pr_in_review_is_never_held(board):
    """Guard (i): the bridge's revision/review/review-loop cards must reach the lane today."""
    revision = _create(title="[revision] [#a1b2] merge deferral predicate", assignee="cs-worker")
    review = _create(title="[review] verify revision [#a1b2]", assignee="cs-reviewer")
    from_event_loop = _create(title="Revise refund wording round 2 at abc12345", assignee="ms-worker",
                              created_by="pr_event_loop")
    for tid in (revision, review, from_event_loop):
        assert _status(tid) == "ready", tid


def test_a_card_with_an_unfinished_parent_still_lands_todo(board):
    parent = _create(title="Parent that is not done", assignee="mumu")
    child = _create(title="Child for a cloud lane", assignee="kiwii-worker", parents=[parent])
    assert _status(child) == "todo"
    assert _held_event(child) is None


def test_the_hold_is_off_until_a_profile_list_is_configured(tmp_path, monkeypatch):
    home = tmp_path / "bare"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    tid = _create(title="Nothing is configured", assignee="ms-worker")
    assert _status(tid) == "ready"


def test_a_broken_policy_never_costs_the_card(board, monkeypatch):
    """A hold feature that drops or silently un-holds a card is worse than no hold."""
    def boom(**kwargs):
        raise RuntimeError("policy exploded")

    monkeypatch.setattr(kh, "hold_reason", boom)
    tid = _create(title="Cloud card that must survive a broken policy", assignee="ms-worker")
    assert _status(tid) == "ready"
    failures = kh.take_failures()
    assert [f["kind"] for f in failures] == ["hold_unavailable"]


# --- B. the drain ------------------------------------------------------------------------


def _hold_many(n: int, *, lane: str = "kiwii-worker") -> list[str]:
    ids = []
    for index in range(n):
        ids.append(_create(title=f"Held card {index}", assignee=lane))
        time.sleep(0.001)  # created_at ordering is second-resolution in some paths
    return ids


def test_the_drain_releases_oldest_first_up_to_the_cap(board):
    ids = _hold_many(6)
    assert all(_status(tid) == "scheduled" for tid in ids)
    result = _release(cap=4)
    assert result.released == ids[:4]
    assert result.still_held == 2
    assert [_status(tid) for tid in ids] == ["ready"] * 4 + ["scheduled"] * 2


def test_a_second_tick_with_the_cap_full_releases_nothing(board):
    ids = _hold_many(6)
    _release(cap=4)
    second = _release(cap=4)
    assert second.released == []
    assert second.slots_before == 0
    assert [_status(tid) for tid in ids[4:]] == ["scheduled", "scheduled"]


def test_the_drain_respects_parents(board):
    """A released card whose parent is unfinished lands ``todo`` and costs no slot."""
    parent = _create(title="Parent still running", assignee="mumu")
    child = _create(title="Held child of an unfinished parent", assignee="kiwii-worker",
                    parents=[parent])
    # The child was parked as `todo` by the parent rule, so park it the way a hold would to
    # prove the release re-gates rather than trusting its own memory of the status.
    with kbc.connect_closing() as conn:
        kb.schedule_task(conn, child, reason="test setup")
        kh.mark_held(conn, child, "test setup")
    result = _release(cap=4)
    assert child in result.released
    assert result.landed[child] == "todo"
    assert _status(child) == "todo"


def test_the_drain_never_touches_a_timed_hold(board):
    """Trap 1: the 10 existing scheduled cards are timed holds. They stay put."""
    timed = _create(title="WAKE: t-shirt-cannon 30-day gate recheck", assignee="gm-ms")
    with kbc.connect_closing() as conn:
        assert kb.schedule_task(conn, timed, reason="WAKE: 30-day gate")
    result = _release(cap=4)
    assert result.released == []
    assert _status(timed) == "scheduled"


def test_a_held_card_re_parked_by_a_human_stays_parked(board):
    """Hold -> drain -> a later timed park must win over the old hold marker."""
    tid = _hold_many(1)[0]
    assert _release(cap=4).released == [tid]
    with kbc.connect_closing() as conn:
        assert kb.schedule_task(conn, tid, reason="after the Vercel snapshot limit resets")
    result = _release(cap=4)
    assert result.released == []
    assert _status(tid) == "scheduled"


def test_a_mumu_scheduled_card_is_not_drained(board):
    tid = _create(title="Mumu's own overnight job", assignee="mumu")
    assert _status(tid) == "ready"
    with kbc.connect_closing() as conn:
        kb.schedule_task(conn, tid, reason="safe-window job")
    assert _release(cap=4).released == []
    assert _status(tid) == "scheduled"


def test_a_dry_run_mutates_nothing(board):
    ids = _hold_many(3)
    before = [_status(tid) for tid in ids]
    result = _release(cap=4, dry_run=True)
    assert result.released == ids
    assert [_status(tid) for tid in ids] == before
    assert all(_status(tid) == "scheduled" for tid in ids)


def test_adopting_ready_cards_parks_them_and_the_drain_then_owns_them(board):
    """First deployment: cards already `ready` for a held lane join the first drain."""
    ready_card = _create(title="Already ready before the hold existed", assignee="kiwii-worker",
                         priority=0)
    with kbc.connect_closing() as conn:
        # Simulate a pre-deployment card: ready, no marker.
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (ready_card,))
        conn.commit()
    assert _status(ready_card) == "ready"
    with kbc.connect_closing() as conn:
        adopted = kh.adopt_ready(conn, dry_run=True)
    assert adopted == [ready_card]
    assert _status(ready_card) == "ready"
    with kbc.connect_closing() as conn:
        adopted = kh.adopt_ready(conn)
    assert adopted == [ready_card]
    assert _status(ready_card) == "scheduled"
    assert _release(cap=4).released == [ready_card]


def test_in_flight_counts_ready_and_running_only(board):
    ready_card = _create(title="runs now", assignee="ms-worker", priority=9)
    held = _create(title="waits", assignee="ms-worker")
    parent = _create(title="parent", assignee="mumu")
    todo_card = _create(title="child", assignee="ms-worker", parents=[parent])
    with kbc.connect_closing() as conn:
        rows = kh.in_flight(conn, kh.hold_policy().profiles)
    ids = {row["id"] for row in rows}
    assert ready_card in ids
    assert todo_card not in ids
    assert held not in ids
