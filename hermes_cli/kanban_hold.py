"""A holding pattern for cloud-lane cards: hold on create, release when there is room.

Rick (2026-11-16): every ticket that goes to a cloud worker is parked instead of claiming a
sandbox the moment it is filed. Rick (2026-11-18): the release is paced by ROOM, not by the
clock — the 03:00-07:00 window is retired and a held card goes back to the queue at ANY hour
when the spark group has room for it. GMs, Mumu and the local ``worker``/``reviewer`` lanes
keep running live; the hold is for the lanes that cost a sandbox.

Two halves, one policy:

* **hold** — :func:`hold_reason` answers "should this new card wait".  :func:`create_task`
  asks it once, in the same transaction that decides ``ready``/``todo``, and a held card is
  born ``scheduled`` with a ``held_for_drain`` event that says why.
* **drain** — :func:`release_due` walks ``scheduled`` cards oldest-first and hands the oldest
  one back through the REAL :func:`hermes_cli.kanban_db.unblock_task`, so
  ``_landing_status_after_parents`` still re-gates a card whose parent is unfinished.

The rule itself lives in ``kanban_drain.py`` (the cron job), not here. It is ONE configured
number — ``kanban.drain_target`` — compared against two READINGS: the board's count of cards
in flight on ``kanban.group_caps.spark.profiles`` (this module's :func:`in_flight`) and the
router's own in-flight count (``spark_inflight_read.py``). This module supplies the pieces
that rule is built from, so the two readings cannot drift apart:

* :func:`in_flight` — cards occupying a spark slot, with a freshness filter that drops a
  ``running`` row whose claim expired and whose heartbeat went quiet (a crashed worker must
  not hold the drain shut forever);
* :func:`held_cards` — only cards whose newest scheduling event is our marker;
* :func:`release_due` with ``limit=1`` — one card per tick, the oldest first.

``kanban.drain_cap`` stays as the backstop ceiling: the setpoint steers, the cap bounds.

The marker is what makes the drain safe. ``scheduled`` is a shared column: timed holds
("WAKE: t-shirt-cannon 30-day gate", "after the Vercel snapshot limit resets") live in it
too, and releasing one of those unattended would run a repo MOVE. So the drain only ever
touches a card whose NEWEST scheduling event is our own ``held_for_drain``; see
:func:`held_for_drain`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

#: Event kind that marks a card as OURS to release. Written once, at create time (or by the
#: one-shot adoption of cards that were already ``ready`` at deployment). Never written by
#: ``schedule_task``, so a timed hold can never be mistaken for a drain slot.
HOLD_EVENT = "held_for_drain"

#: Events that decide which kind of ``scheduled`` card this is. The newest one wins: a card
#: we held, then drained, then re-parked by a human for a timed reason must stay parked.
SCHEDULING_EVENTS = (HOLD_EVENT, "scheduled", "unblocked")

#: Cards a release hands back are counted against the cap; these are the states that occupy a
#: worker. ``todo`` does not — it waits on a parent, not on a slot.
IN_FLIGHT_STATUSES = ("running", "ready")

DEFAULT_DRAIN_CAP = 4
#: Rick 2026-11-18: the ONE configured number. The ideal number of requests in flight on the
#: spark group during a drain, ALL callers included. The two numbers it is compared against
#: are READINGS (the board count and the router's own count), never configured.
DEFAULT_DRAIN_TARGET = 3
#: The ``kanban.group_caps`` group whose members share the sparks. Its ``profiles`` list is
#: the board-side reading's population: reviewers run on dario:opus and are not counted.
SPARK_GROUP = "spark"

#: A ``running`` card whose claim has expired AND whose heartbeat is older than this is not
#: occupying a slot any more: it is a crashed or wedged worker whose row the dispatcher has
#: not reaped yet. Matches ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` (1h) in kanban_db, so
#: the drain and the reaper agree on when a row stops meaning "in flight".
HEARTBEAT_STALE_SECONDS = 60 * 60

#: A card that fixes work already in a PR must not wait a day for the drain: the PR is
#: blocked on it. The review bridge files with the profile author as ``created_by``, so the
#: filer alone cannot tell bridge cards apart — the two lane markers below do.
DEFAULT_EXEMPT_CREATORS = ("pr_event_loop", "pr_idle_clock", "kanban_review_bridge", "ship_card_guard")
DEFAULT_EXEMPT_TITLE_PREFIXES = ("[revision]", "[review]", "WATCHDOG:", "ESCALATION:", "SHIP:")

#: Where a create-time hold failure is recorded for the drain to report. A hold feature that
#: loses or silently un-holds a card is worse than no hold: the card is still created
#: ``ready`` (today's behaviour) and the failure is announced on the next drain tick.
_FAILURES_NAME = "hold_failures.jsonl"


# --- policy ----------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldPolicy:
    """The fleet's hold policy, resolved once per decision."""

    profiles: frozenset[str] = frozenset()
    exempt_priority: Optional[int] = None
    exempt_creators: frozenset[str] = frozenset(DEFAULT_EXEMPT_CREATORS)
    exempt_title_prefixes: tuple[str, ...] = DEFAULT_EXEMPT_TITLE_PREFIXES
    cap: int = DEFAULT_DRAIN_CAP
    #: The ONE configured number (``kanban.drain_target``). The drain's setpoint.
    target: int = DEFAULT_DRAIN_TARGET
    #: ``kanban.group_caps.spark.profiles``: the lanes whose cards count against the target.
    #: Deliberately NOT ``profiles`` (the hold list): reviewers are held but run on the paid
    #: dario:opus lane, so counting them would shut the drain on work that costs no spark.
    spark_profiles: frozenset[str] = frozenset()

    @property
    def enabled(self) -> bool:
        return bool(self.profiles)


_POLICY_KEYS = (
    "hold_profiles", "hold_exempt_priority", "hold_exempt_creators",
    "hold_exempt_title_prefixes", "drain_cap", "drain_target", "group_caps",
)


def _as_list(value: Any) -> list[str]:
    """A config list, or a comma/space separated string. Empty on anything else."""
    if isinstance(value, str):
        return [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(part).strip() for part in value if str(part).strip()]
    return []


def _root_kanban_section() -> dict:
    """``kanban:`` from the pre-profile (root) home's config, ``{}`` when unreadable."""
    try:
        from hermes_constants import get_default_hermes_root
        from hermes_cli.config import read_user_config_raw

        path = get_default_hermes_root() / "config.yaml"
        if not path.exists():
            return {}
        raw = read_user_config_raw(path)
        section = raw.get("kanban") if isinstance(raw, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:  # a policy read must never break card creation
        return {}


def kanban_policy_cfg() -> dict:
    """The hold keys, current profile first and the root home as the fleet default.

    A policy that has to bind EVERY profile cannot live in one profile's ``config.yaml``:
    this fleet files cards from a dozen profiles (mumu, the GMs, the decomposer, every
    worker lane), and a key missing from one of them would silently opt that profile out of
    the hold — the exact shape of a plausible-looking wrong answer. The root home is the
    pre-profile config every profile on this host starts from, so the fleet value lives
    there; a profile that sets a key still wins.
    """
    from hermes_cli.config import cfg_get, load_config

    section = cfg_get(load_config(), "kanban", default=None)
    if not isinstance(section, dict):
        section = {}
    if all(section.get(key) is not None for key in _POLICY_KEYS):
        return section
    merged = dict(_root_kanban_section())
    merged.update({k: v for k, v in section.items() if v is not None})
    return merged


def hold_policy() -> HoldPolicy:
    """Resolve ``kanban.hold_profiles`` and friends into a :class:`HoldPolicy`.

    Unset/empty ``hold_profiles`` means the hold is off, which is the only safe default for
    an install that has not asked for it. ``hold_exempt_priority`` unset means no priority
    escapes the hold; ``drain_cap`` defaults to :data:`DEFAULT_DRAIN_CAP`.

    ``drain_target`` is the Governor's ONE configured number and defaults to
    :data:`DEFAULT_DRAIN_TARGET` (Rick: 3). ``group_caps.spark.profiles`` is the population
    the board-side reading counts; it is read here so no caller has to hard-code the lanes.
    """
    cfg = kanban_policy_cfg()
    profiles = frozenset(_as_list(cfg.get("hold_profiles")))
    exempt_priority: Optional[int] = None
    raw_priority = cfg.get("hold_exempt_priority")
    if raw_priority not in (None, "", False):
        try:
            exempt_priority = int(raw_priority)
        except (TypeError, ValueError):
            exempt_priority = None
    creators = frozenset(c.lower() for c in _as_list(cfg.get("hold_exempt_creators")))
    prefixes = tuple(_as_list(cfg.get("hold_exempt_title_prefixes")))
    cap = DEFAULT_DRAIN_CAP
    try:
        if cfg.get("drain_cap") not in (None, ""):
            cap = max(1, int(cfg["drain_cap"]))
    except (TypeError, ValueError):
        cap = DEFAULT_DRAIN_CAP
    target = DEFAULT_DRAIN_TARGET
    try:
        if cfg.get("drain_target") not in (None, ""):
            target = max(1, int(cfg["drain_target"]))
    except (TypeError, ValueError):
        target = DEFAULT_DRAIN_TARGET
    return HoldPolicy(
        profiles=profiles,
        exempt_priority=exempt_priority,
        exempt_creators=creators or frozenset(DEFAULT_EXEMPT_CREATORS),
        exempt_title_prefixes=prefixes or DEFAULT_EXEMPT_TITLE_PREFIXES,
        cap=cap,
        target=target,
        spark_profiles=spark_profiles(cfg),
    )


def spark_profiles(cfg: Optional[dict] = None) -> frozenset[str]:
    """``kanban.group_caps.spark.profiles``: the lanes that share the sparks.

    Read, never hard-coded (Rick: the eligible set is the group config, set 2026-11-17 to
    worker, cs-worker, kiwii-worker, ms-worker, oap-worker, didi). A missing or malformed
    group yields an empty set, which the drain treats as "cannot count" and refuses to
    release on — fail closed, never guess a population.
    """
    if cfg is None:
        cfg = kanban_policy_cfg()
    groups = cfg.get("group_caps")
    if not isinstance(groups, dict):
        return frozenset()
    spec = groups.get(SPARK_GROUP)
    if not isinstance(spec, dict):
        return frozenset()
    return frozenset(_as_list(spec.get("profiles")))


def hold_reason(
    *, assignee: Optional[str], priority: Optional[int], created_by: Optional[str],
    title: Optional[str], policy: Optional[HoldPolicy] = None,
) -> Optional[str]:
    """Why this new card waits for the drain, or ``None`` when it may run now.

    The two urgency guards are deliberate (Rick left the choice open; both are ON):

    * a card that continues work already in a PR — a revision, a red-gate fix, a re-review,
      anything from the review bridge or the PR event loop — is never held, because the PR
      would wait a whole day for its own fix; and
    * a card at or above ``kanban.hold_exempt_priority`` is never held, so a genuinely
      urgent card can be filed with ``--priority 8`` and dispatch immediately.
    """
    pol = policy or hold_policy()
    if not pol.enabled:
        return None
    name = (assignee or "").strip().lower()
    if not name or name not in pol.profiles:
        return None
    if pol.exempt_priority is not None:
        try:
            if int(priority or 0) >= pol.exempt_priority:
                return None
        except (TypeError, ValueError):
            pass
    if (created_by or "").strip().lower() in pol.exempt_creators:
        return None
    text = (title or "").strip()
    if any(text.startswith(prefix) for prefix in pol.exempt_title_prefixes):
        return None
    return (
        f"{assignee} is a cloud lane (kanban.hold_profiles): the card waits for room on the "
        f"spark group, releasing when fewer than {pol.target} requests are in flight"
    )


# --- the marker ------------------------------------------------------------------------


def mark_held(conn: sqlite3.Connection, task_id: str, reason: str) -> None:
    """Record the hold inside the caller's transaction (``create_task``'s)."""
    from hermes_cli.kanban_db import _append_event

    _append_event(conn, task_id, HOLD_EVENT, {"reason": reason, "status": "scheduled"})


def _newest_scheduling_event(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    placeholders = ", ".join("?" for _ in SCHEDULING_EVENTS)
    row = conn.execute(
        f"SELECT kind FROM task_events WHERE task_id = ? AND kind IN ({placeholders}) "
        "ORDER BY id DESC LIMIT 1",
        (task_id, *SCHEDULING_EVENTS),
    ).fetchone()
    return row["kind"] if row is not None else None


def held_for_drain(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when the card's newest scheduling decision was OUR hold.

    Deliberately not "has a held_for_drain event": a card can be held, drained, and later
    parked again by a human for a timed reason. That later ``scheduled`` event wins, which
    is what keeps a timed hold out of the 03:00 drain.
    """
    return _newest_scheduling_event(conn, task_id) == HOLD_EVENT


# --- reading the board -----------------------------------------------------------------


def _profile_placeholders(profiles: Iterable[str]) -> tuple[str, tuple[str, ...]]:
    names = tuple(sorted(profiles))
    return ", ".join("?" for _ in names), names


def held_cards(conn: sqlite3.Connection, policy: Optional[HoldPolicy] = None) -> list[sqlite3.Row]:
    """``scheduled`` cards we are holding for the drain, oldest first.

    Ordering is ``created_at`` then the card's first event id, not its primary key: ids are
    random, so ``ORDER BY id`` is not insertion order and would make "oldest first" a lie.
    """
    pol = policy or hold_policy()
    if not pol.enabled:
        return []
    placeholders, names = _profile_placeholders(pol.profiles)
    rows = conn.execute(
        "SELECT t.id, t.assignee, t.title, t.created_at, t.priority, "
        "(SELECT MIN(e.id) FROM task_events e WHERE e.task_id = t.id) AS seq "
        "FROM tasks t "
        f"WHERE t.status = 'scheduled' AND t.assignee IN ({placeholders}) "
        "ORDER BY t.created_at ASC, seq ASC",
        names,
    ).fetchall()
    return [row for row in rows if held_for_drain(conn, row["id"])]


def in_flight(
    conn: sqlite3.Connection, profiles: Iterable[str], *, now: Optional[int] = None,
    heartbeat_stale_seconds: int = HEARTBEAT_STALE_SECONDS,
) -> list[sqlite3.Row]:
    """Cards occupying a worker slot for these lanes (``running`` + ``ready``).

    A ``running`` row is dropped when it provably cannot be alive: its claim has expired
    (or is missing, so it can never be extended) AND its heartbeat is older than
    *heartbeat_stale_seconds* or absent. A worker in one long LLM call keeps a fresh
    heartbeat, so it stays counted; a crashed or wedged worker's row would otherwise hold
    the drain shut for the whole of ``dispatch_stale_timeout_seconds``.

    ``ready`` rows always count: they are waiting for a slot, which is exactly what the
    drain's setpoint is measuring.
    """
    placeholders, names = _profile_placeholders(profiles)
    if not names:
        return []
    marks = ", ".join("?" for _ in IN_FLIGHT_STATUSES)
    rows = conn.execute(
        "SELECT id, status, assignee, claim_expires, last_heartbeat_at FROM tasks "
        f"WHERE assignee IN ({placeholders}) AND status IN ({marks}) "
        "ORDER BY created_at ASC, id ASC",
        (*names, *IN_FLIGHT_STATUSES),
    ).fetchall()
    clock = int(now if now is not None else time.time())
    live: list[sqlite3.Row] = []
    for row in rows:
        if row["status"] != "running":
            live.append(row)
            continue
        claim_expires = _epoch(row["claim_expires"])
        heartbeat = _epoch(row["last_heartbeat_at"])
        claim_dead = claim_expires is None or claim_expires < clock
        heartbeat_dead = heartbeat is None or (clock - heartbeat) > int(heartbeat_stale_seconds)
        if claim_dead and heartbeat_dead:
            continue
        live.append(row)
    return live


def _epoch(value: Any) -> Optional[int]:
    """A stored epoch column as int, or ``None`` when missing/unparseable."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# --- the drain -------------------------------------------------------------------------


@dataclass
class ReleaseResult:
    """What one board's release pass did. ``landed`` maps task id -> its new status."""

    released: list[str] = field(default_factory=list)
    landed: dict[str, str] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)
    held_before: int = 0
    slots_before: int = 0

    @property
    def still_held(self) -> int:
        return max(0, self.held_before - len(self.released))


def release_due(
    conn: sqlite3.Connection, *, cap: int, dry_run: bool = False,
    policy: Optional[HoldPolicy] = None, now: Optional[int] = None,
    busy_offset: int = 0, limit: Optional[int] = None,
) -> ReleaseResult:
    """Release the oldest held cards while slots remain; never exceed ``cap`` in flight.

    ``busy_offset`` is cards already occupying a slot on OTHER boards: the cap is fleet-wide
    (four cloud workers, not four per board), and a per-board count would quietly multiply it.

    ``limit`` bounds how many cards this pass may release. The Governor's setpoint passes
    ``limit=1``: one card per tick, oldest first, so a tick can never overshoot the target.

    Free slots are recomputed after every release, so a card that lands ``todo`` (its parent
    is unfinished) costs nothing and the next one still gets its turn. ``dry_run`` answers
    what WOULD be released without touching the board.
    """
    pol = policy or hold_policy()
    now = int(now if now is not None else time.time())
    held = held_cards(conn, pol)
    busy = int(busy_offset) + len(in_flight(conn, pol.profiles))
    result = ReleaseResult(held_before=len(held), slots_before=max(0, cap - busy))
    if not held:
        return result

    for row in held:
        if busy >= cap:
            break
        if limit is not None and len(result.released) >= max(0, int(limit)):
            break
        task_id = row["id"]
        if dry_run:
            result.released.append(task_id)
            busy += 1
            continue
        from hermes_cli.kanban_db import unblock_task

        try:
            released = unblock_task(conn, task_id)
        except Exception:  # one bad card must not stop the pass
            result.failed.append(task_id)
            continue
        if not released:
            result.failed.append(task_id)
            continue
        landed = _status_of(conn, task_id) or "unknown"
        result.released.append(task_id)
        result.landed[task_id] = landed
        if landed in IN_FLIGHT_STATUSES:
            busy += 1
    _ = now  # kept for callers that pass a clock; no time-dependent decision here
    return result


def _status_of(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return row["status"] if row is not None else None


def adopt_ready(
    conn: sqlite3.Connection, *, dry_run: bool = False, policy: Optional[HoldPolicy] = None,
    reason: str = "adopted at deployment: already ready when the drain was installed",
) -> list[str]:
    """One-shot: park ``ready`` cards for the held lanes so the first drain owns them.

    This is NOT part of the recurring tick. A release lands a card in ``ready``; a tick that
    adopted ready cards would re-park the one it just let go, and the fleet would oscillate.
    """
    pol = policy or hold_policy()
    if not pol.enabled:
        return []
    placeholders, names = _profile_placeholders(pol.profiles)
    rows = conn.execute(
        "SELECT id, title, created_at FROM tasks "
        f"WHERE status = 'ready' AND assignee IN ({placeholders}) "
        "ORDER BY created_at ASC, id ASC",
        names,
    ).fetchall()
    adopted: list[str] = []
    for row in rows:
        task_id = row["id"]
        if dry_run:
            adopted.append(task_id)
            continue
        from hermes_cli.kanban_db import write_txn

        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'scheduled', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'ready'",
                (task_id,),
            )
            if cur.rowcount != 1:
                continue
            from hermes_cli.kanban_db import _append_event

            # `scheduled` FIRST, then the hold marker: held_for_drain must be the NEWEST
            # scheduling event or the drain would treat the card as a timed hold.
            _append_event(conn, task_id, "scheduled", {"reason": reason}, run_id=None)
            mark_held(conn, task_id, reason)
        adopted.append(task_id)
    return adopted


# --- failures that must be loud ---------------------------------------------------------


def _failures_path() -> Path:
    from hermes_cli.kanban_db import kanban_home

    return kanban_home() / "kanban" / _FAILURES_NAME


def record_failure(kind: str, detail: str, *, task_id: Optional[str] = None) -> None:
    """Journal a hold failure for the drain to announce. Never raises."""
    try:
        path = _failures_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"at": int(time.time()), "kind": kind, "detail": detail[:500], "task_id": task_id}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception:  # reporting must never break the create path either
        pass


def take_failures(*, max_age_seconds: int = 2 * 86400) -> list[dict]:
    """Read the journal (dropping stale entries, truncating the file). Best effort."""
    try:
        path = _failures_path()
        if not path.exists():
            return []
        entries: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and int(entry.get("at") or 0) >= int(time.time()) - max_age_seconds:
                entries.append(entry)
        path.write_text("", encoding="utf-8")
        return entries
    except Exception:
        return []


def failures_path() -> Path:
    """Exposed for diagnostics and tests."""
    return _failures_path()
