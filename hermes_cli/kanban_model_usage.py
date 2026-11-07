"""Which model actually served a Kanban run: attribution, reconciliation, weekly local share.

The board records what the dispatcher *asked for* — ``tasks.model_override`` or the
assignee profile's default model — but never what the provider *served*. A run that
starts on the local sparks and spills to a cloud fallback mid-session looks exactly
like a healthy local run on the board, so "did this card use cloud?" was unfalsifiable.
That matters because the fleet's worker lanes are configured local on purpose and the
only way to argue about dropping paid inference is a number that comes from the runtime.

Ground truth already exists, just not on the board: the profile's own ``state.db``
keeps ``session_model_usage``, one row per (session, model, billing route) with its API
call count and tokens. On the board, an ``owner_admitted`` event carries the session id
of exactly one attempt (``request_id`` = ``kanban:[board, task, run]``). So

    task_runs.id -> owner_admitted.payload.session_id -> session_model_usage.model

answers "which model ran this card" for every attempt whose session still exists, and
a run whose mix spans two models is a fallback mid-run (the session switched model
while it ran). :func:`reconcile` copies that answer onto the ``task_runs`` row so it
survives session pruning; :func:`report` aggregates it into the weekly local share.

The class table below is the only place a model name is turned into
local/cloud/subscription: extend it there, never at a call site.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# --- What counts as local -------------------------------------------------------
#
# A model name is not enough on its own: ``deepseek:deepseek-flash`` (cloud) and
# ``spark-box:deepseek-flash`` (the local spark pair) are one typo apart, and the
# router can re-point an alias without touching the board. Classification therefore
# reads BOTH the alias and the base_url recorded with the call, and anything it does
# not recognise stays ``unknown`` instead of being folded into a total.
CLASS_LOCAL = "local"          # the DGX Spark pair, or a box on this tailnet
CLASS_CLOUD = "cloud"          # per-token vendor API (DeepSeek, OpenRouter, ...)
CLASS_SUBSCRIPTION = "subscription"  # flat-rate paid plan behind a proxy (dario/Claude)
CLASS_UNKNOWN = "unknown"

# Aliases that the LiteLLM router resolves to the local vLLM deployments.
LOCAL_MODEL_ALIASES = frozenset({
    "spark-box:deepseek-flash",
    "deepseek-v4-flash-vision-exp",
    "deepseek-v4-flash-0731-dspark",
    "didi",
})

# Base URLs that are local by construction (host names / tailnet addresses).
LOCAL_BASE_MARKERS = (
    "spark-a", "spark-b", "spark-c", "spark-d",
    "100.92.40.121", "100.77.235.98",
    "127.0.0.1:4000", "localhost:4000",
)

# Base URLs behind a flat-rate subscription rather than metered inference.
SUBSCRIPTION_BASE_MARKERS = ("dario-proxy", "127.0.0.1:3456")


def classify(model: Optional[str], base_url: Optional[str] = None) -> str:
    """``local`` / ``cloud`` / ``subscription`` / ``unknown`` for a served model + route."""
    m = (model or "").strip()
    b = (base_url or "").strip()
    if any(marker in b for marker in LOCAL_BASE_MARKERS):
        return CLASS_LOCAL
    if any(marker in b for marker in SUBSCRIPTION_BASE_MARKERS):
        return CLASS_SUBSCRIPTION
    if b and "api.deepseek.com" in b:
        return CLASS_CLOUD
    if "openrouter" in b:
        return CLASS_CLOUD
    if not m:
        return CLASS_UNKNOWN
    if m in LOCAL_MODEL_ALIASES:
        return CLASS_LOCAL
    if m.startswith("dario:"):
        return CLASS_SUBSCRIPTION
    if m.startswith("deepseek"):
        # ``spark-box:...`` was checked above; anything else here is the vendor API.
        return CLASS_CLOUD
    return CLASS_UNKNOWN


# --- Per-run attribution --------------------------------------------------------


@dataclass
class Served:
    """One model that actually answered calls for a session.

    ``precise`` is False for a session-row fallback (the profile store kept
    ``sessions.model`` but no per-model usage rows, which is the normal shape on the GM
    lanes). A coarse row names the model and route but carries no per-call counts, so it
    decides the run's *class* and never contributes to a call-weighted total.
    """

    model: str
    provider: str
    base_url: str
    calls: int
    tokens: int
    task: str = ""   # 'title_generation', 'vision', ... ; '' = the main turn loop
    precise: bool = True

    @property
    def model_class(self) -> str:
        return classify(self.model, self.base_url)


@dataclass
class Attribution:
    """What one attempt asked for, what it was served, and by which route."""

    board: str
    task_id: str
    run_id: int
    profile: str
    assignee: str
    session_id: Optional[str]
    started_at: int
    ended_at: Optional[int] = None
    outcome: Optional[str] = None
    requested: str = ""            # task override, else the lane's default model
    request_source: str = ""       # 'override' | 'lane_default'
    served: list[Served] = field(default_factory=list)
    # Why there is no attribution, when there is none: 'no-session-id' (the attempt never
    # reached an owner — typically crashed in pre-flight) or 'session-missing' (admitted,
    # but the session row is gone from the profile store).
    miss: str = ""

    @property
    def main(self) -> list[Served]:
        """The run's own turn loop — auxiliary calls (title generation, vision,
        compression) are a separate concern and must not make a local run look mixed."""
        loop = [s for s in self.served if not s.task]
        return loop or self.served

    @property
    def model(self) -> str:
        """The model that answered the most calls — the run's identity."""
        return max(self.main, key=lambda s: s.calls).model if self.served else ""

    @property
    def route(self) -> Served:
        return max(self.main, key=lambda s: s.calls) if self.served else Served("", "", "", 0, 0)

    @property
    def model_class(self) -> str:
        return self.route.model_class

    @property
    def total_calls(self) -> int:
        """Turn-loop API calls, where the store kept per-call counts (see ``precise``)."""
        return sum(s.calls for s in self.main if s.precise)

    @property
    def is_precise(self) -> bool:
        """True when at least one served row carries real per-call counts."""
        return any(s.precise for s in self.main)

    @property
    def mix(self) -> dict[str, int]:
        return {s.model: s.calls for s in self.served}

    @property
    def fell_back(self) -> bool:
        """True when the run's own turn loop was served by something other than what it
        asked for. Auxiliary models (title generation and friends) never count: they run
        on their own configured model by design."""
        if not self.served:
            return False
        loop = self.main
        if not self.requested:
            return len({s.model for s in loop}) > 1
        return any(s.model != self.requested for s in loop)

    @property
    def calls_by_class(self) -> Counter:
        """API calls per class. A coarse row carries no counts, so it contributes none —
        and must not create a zero entry that reads like a measured class."""
        c: Counter = Counter()
        for s in self.main:
            if s.precise and s.calls:
                c[s.model_class] += s.calls
        return c


def _profile_state_db(profile: Optional[str], db_hint: Optional[str] = None) -> Optional[Path]:
    """The profile's own ``state.db`` (its sessions + per-model usage live there).

    ``db_hint`` is the authority DB recorded in the ``owner_admitted`` event — the exact
    file the owner wrote the session to, so it outranks any name-based guess. The
    profile-path fallbacks matter because a dispatched worker runs with ``HERMES_HOME``
    pinned to its own profile: ``resolve_profile_env('gm-cs')`` from inside a ``mumu``
    worker can resolve somewhere else entirely, and a wrong store silently reads as
    "no attribution" instead of an error.
    """
    if db_hint:
        hinted = Path(db_hint)
        if hinted.exists():
            return hinted
    if profile:
        try:
            from hermes_constants import get_default_hermes_root

            named = Path(get_default_hermes_root()) / "profiles" / profile / "state.db"
            if named.exists():
                return named
        except Exception:
            pass
        try:
            from hermes_cli.profiles import resolve_profile_env

            home = Path(resolve_profile_env(profile))
            db = home / "state.db"
            if db.exists():
                return db
        except Exception:
            return None
    return None


def lane_default_model(profile: Optional[str]) -> str:
    """The model the lane is configured to run, read from the profile's config.yaml.

    This is the board's *intent* for a card without an override. It is read live and
    may legitimately differ from what a past run used (the lane was re-pointed), which
    is exactly why the served model is stored per run instead of being derived later.
    """
    if not profile:
        return ""
    cfg_path = None
    try:
        from hermes_constants import get_default_hermes_root

        cfg_path = Path(get_default_hermes_root()) / "profiles" / profile / "config.yaml"
    except Exception:
        cfg_path = None
    if cfg_path is None or not cfg_path.exists():
        try:
            from hermes_cli.profiles import resolve_profile_env

            cfg_path = Path(resolve_profile_env(profile)) / "config.yaml"
        except Exception:
            return ""
    try:
        from hermes_yaml import safe_load  # type: ignore

        with open(cfg_path, encoding="utf-8-sig") as fh:
            cfg = safe_load(fh) or {}
    except Exception:
        return ""
    model = cfg.get("model")
    if isinstance(model, dict):
        return str(model.get("default") or "")
    return str(model or "")


def _event_session_id(payload: Any) -> Optional[str]:
    if not payload:
        return None
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, ValueError):
        return None
    sid = data.get("session_id") if isinstance(data, dict) else None
    return sid if isinstance(sid, str) and sid else None


def _usage_for(profile: str, session_ids: Iterable[str], db_hint: Optional[str] = None) -> dict[str, list[Served]]:
    """``{session_id: [Served, ...]}`` from that profile's usage table, in one read."""
    ids = {sid for sid in session_ids if sid}
    if not ids:
        return {}
    db = _profile_state_db(profile, db_hint)
    if db is None:
        return {}
    out: dict[str, list[Served]] = {}
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return {}
    try:
        # Chunked: a weekly window can name thousands of sessions and SQLite's
        # variable limit is 999 on older builds.
        id_list = sorted(ids)
        for i in range(0, len(id_list), 500):
            chunk = id_list[i:i + 500]
            marks = ",".join("?" for _ in chunk)
            rows = conn.execute(
                "SELECT session_id, model, COALESCE(billing_provider,''), "
                "COALESCE(billing_base_url,''), COALESCE(task,''), "
                "COALESCE(SUM(api_call_count),0), "
                "COALESCE(SUM(input_tokens+output_tokens),0) "
                f"FROM session_model_usage WHERE session_id IN ({marks}) "
                "GROUP BY session_id, model, billing_provider, billing_base_url, task",
                chunk,
            ).fetchall()
            for sid, model, provider, base, task, calls, tokens in rows:
                out.setdefault(sid, []).append(
                    Served(model=model, provider=provider, base_url=base,
                           calls=int(calls), tokens=int(tokens), task=task or "")
                )
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return out


def _sessions_fallback(db: Path, session_ids: Iterable[str]) -> dict[str, list[Served]]:
    """``sessions.model`` rows for sessions with no per-model usage rows.

    Coarse but authoritative about the *route*: the session row keeps the model and
    billing base_url the session ran on, and its token totals. It cannot show a
    mid-session switch (that needs the usage table's per-model rows), so a coarse answer
    is one model per attempt.
    """
    ids = [sid for sid in session_ids if sid]
    if not ids:
        return {}
    out: dict[str, list[Served]] = {}
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return {}
    try:
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            marks = ",".join("?" for _ in chunk)
            rows = conn.execute(
                "SELECT id, COALESCE(model,''), COALESCE(billing_provider,''), "
                "COALESCE(billing_base_url,''), COALESCE(input_tokens,0)+COALESCE(output_tokens,0) "
                f"FROM sessions WHERE id IN ({marks})",
                chunk,
            ).fetchall()
            for sid, model, provider, base, tokens in rows:
                if not model:
                    continue
                out[sid] = [Served(model=model, provider=provider, base_url=base,
                                   calls=0, tokens=int(tokens or 0), precise=False)]
    except sqlite3.Error:
        return out
    finally:
        conn.close()
    return out


def attribute(
    conn: sqlite3.Connection,
    *,
    board: str = "default",
    task_id: Optional[str] = None,
    since: Optional[int] = None,
) -> list[Attribution]:
    """Per-run attribution for one board, newest first.

    ``since`` is a UNIX timestamp floor on the run's start. Runs with no
    ``owner_admitted`` session (a transition that never reached the owner, or a
    session the profile store has since pruned) come back with ``served=[]`` —
    absence of evidence, kept visible rather than dropped.
    """
    q = """
        SELECT e.task_id, e.run_id, e.payload, e.created_at,
               r.profile, r.started_at, r.ended_at, r.outcome,
               COALESCE(t.assignee,''), COALESCE(t.model_override,''), COALESCE(t.provider_override,'')
          FROM task_events e
          JOIN task_runs r ON r.id = e.run_id
          LEFT JOIN tasks t ON t.id = e.task_id
         WHERE e.kind = 'owner_admitted'
    """
    params: list[Any] = []
    if task_id:
        q += " AND e.task_id = ?"
        params.append(task_id)
    if since is not None:
        q += " AND r.started_at >= ?"
        params.append(int(since))
    q += " ORDER BY r.started_at DESC, r.id DESC"

    raw = conn.execute(q, params).fetchall()
    by_db: dict[tuple[str, Optional[str]], list[str]] = {}
    drafts: list[Attribution] = []
    for (tid, run_id, payload, _created, profile, started, ended, outcome,
         assignee, model_override, provider_override) in raw:
        sid = _event_session_id(payload)
        db_hint = None
        try:
            db_hint = (json.loads(payload) or {}).get("db") if payload else None
        except (TypeError, ValueError):
            db_hint = None
        lane = profile or assignee
        requested = model_override or lane_default_model(lane)
        a = Attribution(
            board=board, task_id=tid, run_id=int(run_id), profile=lane or "",
            assignee=assignee, session_id=sid, started_at=int(started or 0),
            ended_at=ended, outcome=outcome, requested=requested,
            request_source="override" if model_override else "lane_default",
        )
        if provider_override:
            a.request_source = "override"
        drafts.append(a)
        if sid:
            by_db.setdefault((a.profile, db_hint if isinstance(db_hint, str) else None), []).append(sid)

    usage: dict[tuple[str, Optional[str]], dict[str, list[Served]]] = {}
    for key, sids in by_db.items():
        profile, hint = key
        if not profile:
            continue
        precise = _usage_for(profile, sids, hint)
        db = _profile_state_db(profile, hint)
        if db is not None:
            # Second rung: a session with no per-model usage rows still names its model
            # on the sessions row. GM lanes are normally in this shape.
            coarse = _sessions_fallback(db, [sid for sid in sids if sid not in precise])
            precise.update(coarse)
        usage[key] = precise

    for a in drafts:
        if not a.session_id:
            a.miss = "no-session-id"
            continue
        found = False
        for (profile, hint), table in usage.items():
            if profile != a.profile:
                continue
            if a.session_id in table:
                a.served = table[a.session_id]
                found = True
                break
        if not found:
            a.miss = "session-missing"
    return drafts


# --- Durable copy on the run row ------------------------------------------------


def reconcile(
    conn: sqlite3.Connection,
    *,
    board: str = "default",
    since: Optional[int] = None,
    force: bool = False,
) -> int:
    """Write each run's served model onto its ``task_runs`` row; return rows touched.

    Runs whose attribution is already stored are skipped unless ``force``. A run that
    never got a session keeps its NULL columns, so "unattributed" stays distinguishable
    from "served by nothing" on every later query.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(task_runs)")}
    if "served_model" not in cols:
        raise RuntimeError(
            "task_runs has no served_model column: run `hermes kanban model-usage --reconcile` "
            "through a build that carries the migration (kanban_db_connect._TASK_RUN_COLUMNS)"
        )
    guard = "" if force else " AND served_model IS NULL"
    written = 0
    for a in attribute(conn, board=board, since=since):
        if not a.served:
            continue
        cur = conn.execute(
            "UPDATE task_runs SET served_model = ?, served_class = ?, served_provider = ?, "
            "served_base_url = ?, served_mix = ?, requested_model = ? "
            f"WHERE id = ?{guard}",
            (
                a.model, a.model_class, a.route.provider, a.route.base_url,
                json.dumps(a.mix, sort_keys=True), a.requested, a.run_id,
            ),
        )
        written += cur.rowcount or 0
    conn.commit()
    return written


# Reconcile writes these columns of ``task_runs`` and nothing else. The DDL is read from
# the board migration (``_TASK_RUN_COLUMNS``) rather than copied here, so the scheduled
# reconciler and the schema cannot drift apart.

def reconcile_boards(
    boards: Optional[Iterable[str]] = None,
    *,
    since: Optional[int] = None,
) -> tuple[int, list[str]]:
    """Reconcile every board, opening each DB read-write directly. ``(rows, errors)``.

    Deliberately not through ``kanban_db_connect.connect``: a scheduled reconciler runs
    inside a delegated context, where that path hands back a READ-ONLY handle by design
    (the fence that stops an agent mutating a board it does not own). This function is
    not an agent — no model, no input, fixed code — and its whole write surface is the
    ``served_*``/``requested_model`` columns of ``task_runs``, so it stays off that fence
    without widening it. Everything else about a board is read-only here.
    """
    from hermes_cli import kanban_db_connect as kbc

    wanted = tuple(
        (name, ddl) for name, ddl in kbc._TASK_RUN_COLUMNS
        if name.startswith("served_") or name == "requested_model"
    )
    total = 0
    errors: list[str] = []
    for slug in _board_slugs(boards):
        path = board_db_path(slug)
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(str(path), timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                existing = {row[1] for row in conn.execute("PRAGMA table_info(task_runs)")}
                for name, ddl in wanted:
                    if name not in existing:
                        conn.execute(f"ALTER TABLE task_runs ADD COLUMN {ddl}")
                total += reconcile(conn, board=slug, since=since)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            errors.append(f"{slug}: {type(exc).__name__}: {exc}")
    return total, errors


# --- Reporting ------------------------------------------------------------------


def _board_slugs(boards: Optional[Iterable[str]]) -> list[str]:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_boards as kbb

    if boards:
        return [kb._slug_or_default(b) for b in boards]
    return [entry["slug"] for entry in kbb.list_boards() if not entry.get("archived")]


def board_db_path(slug: str) -> Path:
    """On-disk DB for ``slug``, resolved WITHOUT the worker's board pin.

    ``kanban_db_path()`` honours ``HERMES_KANBAN_DB`` first, so inside a dispatched
    worker every board resolves to that worker's own board. A cross-board report called
    from a worker would then read one board N times and count every run N times — the
    exact shape of a plausible-looking wrong number. Resolve from the boards root
    instead: a fleet report means every board, by construction.
    """
    from hermes_cli import kanban_db as kb

    if slug == kb.DEFAULT_BOARD:
        return kb.kanban_home() / "kanban.db"
    return kb.board_dir(slug) / "kanban.db"


def attribute_boards(
    boards: Optional[Iterable[str]] = None,
    *,
    since: Optional[int] = None,
    task_id: Optional[str] = None,
) -> list[Attribution]:
    """Attribution across boards (each board DB is opened through ``connect``)."""
    from hermes_cli import kanban_db_connect as kbc

    out: list[Attribution] = []
    for slug in _board_slugs(boards):
        path = board_db_path(slug)
        if not path.exists():
            continue
        try:
            conn = kbc.connect(path)
        except (sqlite3.Error, OSError):
            continue
        try:
            out.extend(attribute(conn, board=slug, task_id=task_id, since=since))
        except sqlite3.Error:
            continue
        finally:
            conn.close()
    return out


@dataclass
class LaneRow:
    lane: str
    runs: int = 0
    calls: int = 0
    local_runs: int = 0
    cloud_runs: int = 0
    subscription_runs: int = 0
    unknown_runs: int = 0
    fell_back: int = 0


def report(
    *,
    days: int = 7,
    boards: Optional[Iterable[str]] = None,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """The weekly number: what share of run-level inference was served locally.

    Two denominators, deliberately: ``runs`` (how many cards ran where — what a human
    reads) and ``calls`` (API calls — what actually drives spend and throughput). A
    card that starts local and spills to cloud is counted on both sides, so neither
    number can hide a fallback.
    """
    now = time.time() if now is None else now
    cutoff = int(now - days * 86400)
    rows = attribute_boards(boards, since=cutoff)
    attributed = [a for a in rows if a.served]

    by_lane: dict[str, LaneRow] = {}
    calls = Counter()
    run_classes = Counter()
    fallbacks: list[Attribution] = []
    per_day: dict[str, Counter] = {}
    precise_runs = 0

    for a in attributed:
        lane = a.assignee or a.profile or "?"
        lr = by_lane.setdefault(lane, LaneRow(lane=lane))
        lr.runs += 1
        lr.calls += a.total_calls
        precise_runs += 1 if a.is_precise else 0
        cls = a.model_class
        run_classes[cls] += 1
        if cls == CLASS_LOCAL:
            lr.local_runs += 1
        elif cls == CLASS_CLOUD:
            lr.cloud_runs += 1
        elif cls == CLASS_SUBSCRIPTION:
            lr.subscription_runs += 1
        else:
            lr.unknown_runs += 1
        if a.fell_back:
            lr.fell_back += 1
            fallbacks.append(a)
        for c, n in a.calls_by_class.items():
            calls[c] += n
        day = datetime.datetime.fromtimestamp(a.started_at).astimezone().strftime("%Y-%m-%d")
        per_day.setdefault(day, Counter())[cls] += 1

    total_calls = sum(calls.values())
    total_runs = sum(run_classes.values())
    unattributed = Counter()
    miss_outcome = Counter()
    for a in rows:
        if a.served:
            continue
        unattributed[a.miss or "unknown"] += 1
        miss_outcome[a.outcome or "(running)"] += 1
    return {
        "window_days": days,
        "since": cutoff,
        "runs_seen": len(rows),
        "runs_attributed": total_runs,
        "runs_with_call_counts": precise_runs,
        "runs_by_class": dict(run_classes),
        "unattributed_by_reason": dict(unattributed),
        "unattributed_by_outcome": dict(miss_outcome.most_common(6)),
        "calls_by_class": dict(calls),
        "local_share_runs": (run_classes[CLASS_LOCAL] / total_runs) if total_runs else None,
        "local_share_calls": (calls[CLASS_LOCAL] / total_calls) if total_calls else None,
        "lanes": [vars(r) for r in sorted(by_lane.values(), key=lambda r: -r.runs)],
        "fallback_runs": [
            {
                "board": a.board, "task_id": a.task_id, "run_id": a.run_id,
                "lane": a.assignee or a.profile, "requested": a.requested,
                "served": a.mix, "started_at": a.started_at,
            }
            for a in sorted(fallbacks, key=lambda a: a.started_at, reverse=True)
        ],
        "per_day": {day: dict(counter) for day, counter in sorted(per_day.items())},
    }


# --- Override audit -------------------------------------------------------------

# A pin away from the lane default is only legitimate when the card says why.
# Matched against the card body/title, case-insensitively.
_TIME_SENSITIVE_MARKERS = (
    "time-sensitive", "time sensitive", "urgent", "asap", "deadline",
    "faster processing", "expedite", "today only", "hotfix",
)


def format_report(rep: dict[str, Any]) -> str:
    """Render :func:`report` as the text a human reads: one headline, then the detail.

    Kept next to the numbers so the CLI and the weekly cron print the same thing —
    two formatters drift, and a report whose wording disagrees with its own arithmetic
    is worse than no report.
    """
    def pct(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    lines = [
        f"local share ({rep['window_days']}d, {len(rep['lanes'])} lanes): "
        f"{pct(rep['local_share_runs'])} of runs, {pct(rep['local_share_calls'])} of API calls",
        f"runs attributed {rep['runs_attributed']} of {rep['runs_seen']} attempts; "
        f"{rep['runs_with_call_counts']} carry per-call counts (the call share covers those)",
    ]
    if rep["unattributed_by_reason"]:
        why = ", ".join(f"{k} {v}" for k, v in sorted(rep["unattributed_by_reason"].items()))
        outs = ", ".join(f"{k} {v}" for k, v in rep["unattributed_by_outcome"].items())
        lines.append(f"unattributed: {why}  (outcomes: {outs})")
    if not rep["runs_attributed"]:
        return "\n".join(lines)
    lines.append("")
    lines.append(f"{'LANE':14s}{'RUNS':>6s}{'LOCAL':>7s}{'CLOUD':>7s}{'PAID':>6s}{'UNK':>5s}{'SPILLED':>9s}")
    for row in rep["lanes"]:
        lines.append(
            f"{row['lane']:14s}{row['runs']:6d}{row['local_runs']:7d}{row['cloud_runs']:7d}"
            f"{row['subscription_runs']:6d}{row['unknown_runs']:5d}{row['fell_back']:9d}"
        )
    if rep["fallback_runs"]:
        lines.append("")
        lines.append(f"{len(rep['fallback_runs'])} run(s) were served by a model other than the "
                     f"one they asked for (newest 8):")
        for f in rep["fallback_runs"][:8]:
            served = ", ".join(f"{m}x{c}" for m, c in f["served"].items())
            lines.append(f"  {f['board']:9s} {f['task_id']} {f['lane']:13s} "
                         f"asked={f['requested'] or '-'} served={served}")
    return "\n".join(lines)


def audit_overrides(
    *,
    boards: Optional[Iterable[str]] = None,
    lanes: Optional[Iterable[str]] = None,
    skip_reviewer_lanes: bool = True,
) -> list[dict[str, Any]]:
    """Cards running away from their lane's default model, with the stated reason.

    Rick's rule: reviews run on Opus deliberately, and every other card runs on its
    lane default unless the owner asked for faster processing because the work is
    time-sensitive — in which case the reason belongs in the card so the next reader
    sees why. This lists non-default pins and flags the ones with no reason recorded.

    Reviewer lanes are skipped by default because their pin *is* the deliberate choice;
    pass ``skip_reviewer_lanes=False`` to see them too.
    """
    from hermes_cli import kanban_db_connect as kbc

    want = {lane.strip() for lane in lanes} if lanes else None
    out: list[dict[str, Any]] = []
    for slug in _board_slugs(boards):
        path = board_db_path(slug)
        if not path.exists():
            continue
        try:
            conn = kbc.connect(path)
        except (sqlite3.Error, OSError):
            continue
        try:
            rows = conn.execute(
                "SELECT id, title, COALESCE(body,''), COALESCE(assignee,''), status, "
                "COALESCE(model_override,''), COALESCE(provider_override,'') "
                "FROM tasks WHERE COALESCE(model_override,'') != '' "
                "AND status NOT IN ('archived')"
            ).fetchall()
        except sqlite3.Error:
            continue
        finally:
            conn.close()
        for tid, title, body, assignee, status, model, provider in rows:
            if want is not None and assignee not in want:
                continue
            if want is None and skip_reviewer_lanes and "review" in (assignee or ""):
                continue
            default = lane_default_model(assignee)
            if default and model == default:
                continue  # pinning the lane default is not a pin
            text = f"{title}\n{body}".lower()
            reason = next((m for m in _TIME_SENSITIVE_MARKERS if m in text), "")
            out.append({
                "board": slug, "task_id": tid, "assignee": assignee, "status": status,
                "pinned": f"{provider}:{model}" if provider else model,
                "lane_default": default, "pinned_class": classify(model, ""),
                "reason": reason, "unexplained": not reason,
                "title": title,
            })
    return out
