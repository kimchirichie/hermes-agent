"""Router in-flight count used to seed a kanban group cap.

A group cap counts running cards. The sparks are also used by chat sessions and crons, so the
router's own in-flight count for the model can be higher. The dispatcher starts each group at
``max(running cards, router count)``; whichever is higher decides.

Fail closed: an unreadable, stale or malformed reading returns None and the caller treats the
group as full for this tick. Never guess a number.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)
_warned_malformed: set[str] = set()

MAX_AGE_SECONDS = 120.0
TIMEOUT_SECONDS = 2.0


def read_router_count(url: str, model: str, *, now: Optional[float] = None) -> Optional[int]:
    """The router's in-flight count for ``model``, or None when it cannot be trusted."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kanban-router-seed/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        count = body["groups"][model]
        ts = float(body["ts"])
    except Exception as exc:
        logger.warning("kanban router seed: unreadable (%s)", exc)
        return None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        logger.warning("kanban router seed: bad count %r", count)
        return None
    age = (time.time() if now is None else now) - ts
    if abs(age) > MAX_AGE_SECONDS:
        logger.warning("kanban router seed: reading is %.0fs off", age)
        return None
    return count


def group_seeds(group_caps: Optional[Mapping[str, Any]]) -> dict[str, int]:
    """``{group: starting count}`` for each group that sets ``router_seed: {url, model}``.

    An unreadable router seeds the group at its own max, so nothing new starts this tick.
    """
    seeds: dict[str, int] = {}
    for name, spec in (group_caps or {}).items():
        src = spec.get("router_seed") if isinstance(spec, Mapping) else None
        if src is None:
            continue
        if not isinstance(src, Mapping) or not src.get("url") or not src.get("model"):
            if name not in _warned_malformed:
                _warned_malformed.add(name)
                logger.warning("kanban router seed: group %r has router_seed but not {url, model}; ignored", name)
            continue
        try:
            cap = int(spec.get("max"))
        except (TypeError, ValueError):
            continue
        count = read_router_count(str(src["url"]), str(src["model"]))
        seeds[str(name)] = count if count is not None else cap
    return seeds


def fleet_group_caps_and_seed() -> tuple[Optional[Mapping[str, Any]], dict[str, int]]:
    """``(group_caps, group_seed)`` for a caller that dispatches outside the watcher tick.

    The dashboard nudge and the daemon use this so they obey the same cap and router reading
    as the gateway tick instead of spawning past it.
    """
    from hermes_cli.kanban_hold import fleet_caps

    _per_profile, caps = fleet_caps()
    return caps, group_seeds(caps)
