"""Park in-flight kanban runs as ``interrupted`` before a shutdown kills their workers.

A gateway restart or signal stop takes every kanban worker subprocess with it: the workers are
children of the dispatch host, and the shutdown's process sweep (``process_registry.kill_all``)
plus the gateway's own death end them mid-turn. Booked by the crash sweep, each of those deaths
became a ``crashed`` run — it charged the card's retry budget and could trip the circuit breaker,
so one gateway restart auto-blocked a dozen cards and re-ran work that was half finished.

Parking writes the honest record while the gateway is still alive to write it: the run ends
``interrupted``, the card returns to its retry lane, and ``consecutive_failures`` is untouched.
This is the kanban analogue of ``cron.scheduler.mark_running_jobs_interrupted``, which already
does exactly this for cron jobs at the same shutdown step.

Only runs THIS process is about to kill are parked: a run whose ``owner_admitted`` receipt names
another (possibly still live) process, or a plain dispatcher worker that is not a child of this
process, belongs to somebody else and is left alone.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("gateway.kanban_shutdown_park")


def _kb():
    from hermes_cli import kanban_db as kb
    return kb


def _board_slugs() -> list[str]:
    """Live boards, default first; falls back to the default board when listing fails."""
    kb = _kb()
    try:
        from hermes_cli.kanban_db_boards import list_boards

        return [b.get("slug") or kb.DEFAULT_BOARD for b in list_boards(include_archived=False)]
    except Exception:
        return [kb.DEFAULT_BOARD]


def _owner_pid(conn, task_id: str, run_id: Optional[int]) -> Optional[int]:
    """The pid the run's ``owner_admitted`` receipt names, or None when the run has no receipt."""
    sql = (
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'owner_admitted'"
        + (" AND run_id = ?" if run_id is not None else "")
        + " ORDER BY id DESC LIMIT 1"
    )
    params: tuple = (task_id, int(run_id)) if run_id is not None else (task_id,)
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    import json

    try:
        payload = json.loads(row["payload"])
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    raw = payload.get("pid")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _is_child_of(pid: int, parent: int) -> bool:
    """True when *pid* is still a direct child of *parent* (a bare dispatcher worker of ours)."""
    try:
        import psutil

        return psutil.Process(pid).ppid() == parent
    except Exception:
        return False


def _owned_by_this_process(conn, row: Any, me: int) -> bool:
    """True when this gateway is the process that will kill the run's worker."""
    pid = _kb()._row_get(row, "worker_pid")
    try:
        worker_pid = int(pid)
    except (TypeError, ValueError):
        return False
    owner = _owner_pid(conn, row["id"], _kb()._row_get(row, "current_run_id"))
    if owner is not None:
        return owner == me
    return _is_child_of(worker_pid, me)


def _park_board(conn, slug: str, me: int, reason: str) -> list[dict]:
    kb = _kb()
    rows = conn.execute(
        "SELECT id, claim_lock, worker_pid, current_run_id FROM tasks "
        "WHERE status = 'running' AND worker_pid IS NOT NULL"
    ).fetchall()
    parked: list[dict] = []
    for row in rows:
        task_id = row["id"]
        if not _owned_by_this_process(conn, row, me):
            continue
        pid = int(kb._row_get(row, "worker_pid"))
        retry_status = kb._retry_status_for_run(conn, task_id, kb._row_get(row, "current_run_id"))
        error_text = (
            f"interrupted by {reason}: the gateway stopped while this run was in flight "
            "— parked before the worker was signalled, retry budget untouched"
        )
        payload = {
            "pid": pid,
            "reason": reason,
            "retry_status": retry_status,
            "parked_before_shutdown": True,
        }
        with kb.write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' AND claim_lock IS ?",
                (retry_status, task_id, row["claim_lock"]),
            )
            if cur.rowcount != 1:
                continue
            run_id = kb._end_run(
                conn, task_id, outcome="interrupted", status="interrupted",
                error=error_text, metadata=payload,
            )
            kb._append_event(conn, task_id, "interrupted", payload, run_id=run_id)
        parked.append({"board": slug, "task_id": task_id, "run_id": run_id,
                       "pid": pid, "retry_status": retry_status})
    return parked


def park_owned_runs(
    reason: str, *, owner_pid: Optional[int] = None, boards: Optional[list[str]] = None,
) -> list[dict]:
    """Park every in-flight kanban run this process is about to kill.

    ``reason`` names the stop for the board history ("gateway restart", "gateway shutdown").
    Returns one dict per parked run (``board``/``task_id``/``run_id``/``pid``/``retry_status``);
    empty when nothing was in flight. Never raises — a parking failure must not block a shutdown.
    """
    from hermes_cli import kanban_db_connect as kbc

    me = int(owner_pid or os.getpid())
    parked: list[dict] = []
    for slug in boards or _board_slugs():
        conn = None
        try:
            conn = kbc.connect(board=slug)
            parked.extend(_park_board(conn, slug, me, reason))
        except Exception as exc:  # noqa: BLE001 — shutdown must proceed regardless
            logger.warning("kanban park: board %s skipped (%s)", slug, exc)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    return parked


def park_owned_runs_best_effort(reason: str) -> list[dict]:
    """``park_owned_runs`` behind a blanket guard for the shutdown path."""
    try:
        return park_owned_runs(reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("kanban park: skipped (%s)", exc)
        return []
