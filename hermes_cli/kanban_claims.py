"""One change, one lane: what does a card claim, and who already holds it?

Measured failure (2026-11-05, board `default`): a single ``gate.yml`` change produced six cards
and three PRs across three lanes. Every lane reasoned correctly; none could see that another
lane already held the work, because a lane's sandbox cannot see another lane's *unmerged*
branch. So the subject has to be judged from shared state — the board, and GitHub — never from
local state.

This module is the single owner of the two questions:

* :func:`extract_claims` — which branches and files does this card name?
* :func:`find_conflicts` — does another *open* card of the same lane (build / review / ship)
  already hold one of them?

``create_task`` uses it to gate a new card behind the card that already holds the subject
(a dependency edge, so it resumes by itself), and the ``kanban_claim_check`` tool uses it at
work time, adding GitHub's open PRs so an unmerged branch from another lane is visible.

Precision beats recall here: a refusal costs a delayed card, a miss costs a duplicate PR and a
review lane. Every rule below was checked against the six real cards of that incident.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

BUILD, REVIEW, SHIP = "build", "review", "ship"

#: Cards in these states still hold their subject. A done or archived card does not.
OPEN_STATUSES_EXCLUDED = ("done", "archived")

#: File names that say nothing about *which* work is claimed: two cards in one repo both
#: "touch README.md" without touching each other's work. Never a claim on their own.
GENERIC_FILES = frozenset({
    "readme.md", "skill.md", "soul.md", "agents.md", "claude.md", "contributing.md",
    "changelog.md", "license", "license.md", "makefile", "dockerfile", "package.json",
    "package-lock.json", "pyproject.toml", "requirements.txt", "composer.json",
    "composer.lock", "config.yaml", "config.yml", "config.json", ".gitignore", ".env",
    ".env.example", "docker-compose.yml", "tsconfig.json", "flake.nix", "uv.lock",
})

_CODE_EXTS = frozenset({
    "py", "php", "js", "mjs", "cjs", "jsx", "ts", "tsx", "yml", "yaml", "json", "md",
    "sh", "bash", "zsh", "sql", "css", "scss", "html", "go", "rb", "java", "kt", "swift",
    "c", "h", "cpp", "hpp", "toml", "ini", "cfg", "conf", "xml", "tf", "lock", "txt",
})

#: A token that ends in a TLD is a host, not a path (`omni-pipeline.dev`, `sinbros.xyz`).
_TLDS = frozenset({"com", "net", "org", "io", "dev", "app", "xyz", "co", "us", "ai", "sh"})

_FILE_RE = re.compile(r"(?<![\w/.-])(\.?[\w@+-]+(?:/[\w@+-]+)*\.[A-Za-z][A-Za-z0-9]{0,7})(?![\w/-])")
_BRANCH_HINT_RE = re.compile(
    r"(?:branch|branches|head|base|based on|rebase[sd]?|pushed|push to|worktree)\b"
    r"[^`\n]{0,40}`([^`\n]{3,80})`", re.IGNORECASE)
_BRANCH_LABEL_RE = re.compile(r"\b(?:branch|Branch)\s*:\s*`?([A-Za-z0-9][\w./-]{2,80})`?")
#: A bare ``x/y`` token in the TITLE is a ref: titles are the card's one-line subject statement,
#: which is exactly why a body (where prose quotes other lanes' branches) is not treated the same.
_TITLE_BRANCH_RE = re.compile(r"(?<![\w/.-])([A-Za-z0-9][\w.-]*(?:/[\w.-]+)+)(?![\w/-])")
#: A bare ``x/y`` in a title is only a ref when it is shaped like one. ``red/black`` (a skirt)
#: and ``scripts/tests`` (a directory) are not branches; ``ms/gate-defer-eligible-run`` and
#: ``wt/t_22352b7f-gate-merge-retry`` are. First segment = a ref namespace, or a hyphenated /
#: underscored last segment.
_BRANCH_NAMESPACES = frozenset({
    "wt", "ms", "cs", "oap", "kiwii", "feature", "feat", "fix", "hotfix", "bugfix", "release",
    "chore", "docs", "experiment", "exp", "claude", "codex", "dependabot", "renovate",
})
#: An inline code span holding exactly one path is a claim: a card that backticks a file path is
#: talking about that file. Spans with whitespace are commands and flags, never claims.
_CODE_SPAN_RE = re.compile(r"`([^`\n]{4,140})`")
_GITHUB_REPO_RE = re.compile(r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_-][A-Za-z0-9_.-]*)")
_REPO_LABEL_RE = re.compile(r"\brepo(?:sitory)?\s*:\s*`?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)`?", re.IGNORECASE)

#: Titles that open a review or a ship lane. Anything else builds.
_SHIP_TITLE_RE = re.compile(r"^\s*(?:ship|merge)\b", re.IGNORECASE)
_REVIEW_TITLE_RE = re.compile(r"^\s*(?:re-?)?(?:review|delta review|qa|verify)\b", re.IGNORECASE)

#: ``overlap: allowed`` in the body says "this really is a second, independent change to the
#: same files" — two parallel fixes in one file are legitimate, they just have to say so. It
#: never excuses a shared branch: the same branch IS the same work.
_OVERLAP_OK_RE = re.compile(r"^\s*overlap(?:-ok)?\s*:\s*allowed\b", re.IGNORECASE | re.MULTILINE)


def overlaps_allowed(body: Optional[str]) -> bool:
    return bool(body and _OVERLAP_OK_RE.search(str(body)))


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "disabled")


def dedupe_enabled() -> bool:
    """``kanban.claim_dedupe`` (config) / ``HERMES_KANBAN_CLAIM_DEDUPE`` (env); default on."""
    try:
        from hermes_cli.config import cfg_get, load_config
        value = cfg_get(load_config(), "kanban", "claim_dedupe")
        if value is not None:
            return bool(value) if not isinstance(value, str) else value.strip().lower() not in (
                "0", "false", "no", "off", "disabled")
    except Exception:
        pass
    return _env_flag("HERMES_KANBAN_CLAIM_DEDUPE", True)


# --- roles -------------------------------------------------------------------------------

def card_role(assignee: Optional[str], title: Optional[str]) -> str:
    """Which lane this card is in: ``build``, ``review`` or ``ship``.

    Two cards in the SAME lane holding the same subject are duplicates. A build and the review
    that follows it (or the ship that merges it) legitimately share every branch and file —
    that is a dependency, not a collision — so the lane is part of the identity.
    """
    name = (assignee or "").strip().lower()
    if name == "reviewer" or name.endswith("-reviewer") or name.endswith("_reviewer"):
        return REVIEW
    text = (title or "").strip()
    if _SHIP_TITLE_RE.match(text):
        return SHIP
    if _REVIEW_TITLE_RE.match(text):
        return REVIEW
    return BUILD


# --- claims ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Claims:
    """What a card says it will act on. Frozen: it is a value, not state."""

    branches: frozenset[str] = frozenset()
    files: frozenset[str] = frozenset()
    repos: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.branches or self.files)

    def summary(self) -> str:
        parts = []
        if self.branches:
            parts.append("branch " + ", ".join(sorted(self.branches)))
        if self.files:
            parts.append("files " + ", ".join(sorted(self.files)))
        return "; ".join(parts) or "no claim"


def _norm_path(token: str) -> Optional[str]:
    tok = token.strip().strip("`'\"").rstrip(".,;:)]}>")
    if not tok or "://" in tok or tok.startswith("/") or tok.startswith("~"):
        return None
    if any(ch.isspace() for ch in tok):  # a command line, not a path
        return None
    # Only a real ``./`` prefix is noise: a leading dot is part of the path here
    # (``.github/workflows/gate.yml``), and ``lstrip("./")`` would eat it and fork the claim.
    while tok.startswith("./"):
        tok = tok[2:]
    if not tok or ("/" not in tok and "." not in tok):
        return None
    base = tok.rsplit("/", 1)[-1].lower()
    if not base or base.startswith("."):
        return None
    ext = base.rsplit(".", 1)[-1] if "." in base else ""
    if ext not in _CODE_EXTS or ext in _TLDS:
        return None
    if base in GENERIC_FILES or tok.lower() in GENERIC_FILES:
        return None
    if "*" in tok or "?" in tok or "=" in tok or "(" in tok:
        return None
    return tok


def _norm_branch(token: str, *, bare: bool = False) -> Optional[str]:
    tok = token.strip().strip("`'\"").strip()
    for prefix in ("refs/heads/", "origin/"):
        tok = tok.removeprefix(prefix)
    tok = tok.rstrip(".,;:)]}>")
    if not (3 <= len(tok) <= 80) or "/" not in tok:
        return None
    if any(ch in tok for ch in " \\*?=()[]<>,@\n"):  # globs, prose, or "@ v2": not a ref
        return None
    last = tok.rsplit("/", 1)[-1]
    if "." in last:  # a file (``docs/agents/gate-policy-tests.md``), not a branch
        return None
    if not all(re.fullmatch(r"[\w.-]+", seg) for seg in tok.split("/")):
        return None
    if bare:
        # An unlabelled ``x/y`` token: only its shape makes it a ref (see _BRANCH_NAMESPACES).
        first, last = tok.split("/")[0], tok.rsplit("/", 1)[-1]
        if len(tok) < 8 or (first not in _BRANCH_NAMESPACES and "-" not in last and "_" not in last):
            return None
    return tok.lower()


def extract_claims(
    title: Optional[str] = None,
    body: Optional[str] = None,
    *,
    branch_name: Optional[str] = None,
    extra_files: Iterable[str] = (),
    extra_branch: Optional[str] = None,
) -> Claims:
    """Branches + files this card claims, from its title, its labelled body, and its columns.

    Precision is the whole design. A 2026-11-05 card *about* the duplicate-lane incident quoted
    another lane's branch and a half-dozen files while touching none of them; a claim taken from
    that body would have gated it behind work it had nothing to do with. So:

    * **Title** — every file and every ``x/y`` ref counts. The title is the subject statement.
    * **Body** — only structural positions count: ``Branch: …``, ``branch `…``` / ``head `…```,
      and a line that *starts* with a code span holding a path. Prose is not a claim.
    * **Columns** — ``branch_name`` (the worktree's real branch) always counts.

    A token only counts as a file when it carries a source-code extension and is not one of the
    ubiquitous names in :data:`GENERIC_FILES`; only counts as a branch when it is ref-shaped
    (one or more ``/`` segments, no extension in the last one, no shell metacharacters).
    """
    branches: set[str] = set()
    files: set[str] = set()
    repos: set[str] = set()

    for source in (branch_name, extra_branch):
        normed = _norm_branch(str(source)) if source else None
        if normed:
            branches.add(normed)

    head = str(title or "")
    for match in _FILE_RE.finditer(head):
        normed = _norm_path(match.group(1))
        if normed:
            files.add(normed)
    for regex, bare in ((_BRANCH_HINT_RE, False), (_BRANCH_LABEL_RE, False), (_TITLE_BRANCH_RE, True)):
        for match in regex.finditer(head):
            normed = _norm_branch(match.group(1), bare=bare)
            if normed:
                branches.add(normed)

    text = str(body or "")
    if text:
        for regex in (_BRANCH_HINT_RE, _BRANCH_LABEL_RE):
            for match in regex.finditer(text):
                normed = _norm_branch(match.group(1))
                if normed:
                    branches.add(normed)
        for match in _CODE_SPAN_RE.finditer(text):
            normed = _norm_path(match.group(1))
            if normed:
                files.add(normed)

    for extra in extra_files:
        normed = _norm_path(str(extra))
        files.add(normed or str(extra).strip())
    _collect_repos(head, repos)
    _collect_repos(str(body or ""), repos, hints=False)
    return Claims(branches=frozenset(branches), files=frozenset(files), repos=frozenset(repos))


#: Repo names as they appear in card prose. Two cards on the same *file name* in two different
#: repos are not the same work (``gate.yml`` exists in omni, Kiwii, OAP and CS). ``kanban.
#: claim_repo_aliases`` replaces this table when a repo is added or renamed.
DEFAULT_REPO_ALIASES: dict[str, tuple[str, ...]] = {
    "omni": ("omni", "omni-pipeline"),
    "kiwii": ("kiwii", "kiwiidev"),
    "oap": ("oap", "ominousapp", "ominous app"),
    "mspv": ("mspv",),
}


def repo_aliases() -> dict[str, tuple[str, ...]]:
    """Alias table for repo hints: ``kanban.claim_repo_aliases`` if configured, else the default."""
    try:
        from hermes_cli.config import cfg_get, load_config
        configured = cfg_get(load_config(), "kanban", "claim_repo_aliases")
        if isinstance(configured, dict) and configured:
            table = {}
            for key, values in configured.items():
                items = [values] if isinstance(values, str) else list(values or [])
                if key and items:
                    table[str(key).lower()] = tuple(str(v).lower() for v in items)
            if table:
                return table
    except Exception:
        pass
    return DEFAULT_REPO_ALIASES


def _repo_hints(blob: str) -> set[str]:
    low = blob.lower()
    hints = set()
    for key, aliases in repo_aliases().items():
        for alias in aliases:
            if re.search(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])", low):
                hints.add(key)
                break
    return hints


def _collect_repos(blob: str, repos: set[str], *, hints: bool = True) -> None:
    """``owner/repo`` ids anywhere, plus repo-name hints from the title only.

    The title names the card's own subject (``OAP gate.yml: …``); the body mentions other repos
    in passing, so a body hint would make every card look like it spans every repo.
    """
    for match in _GITHUB_REPO_RE.finditer(blob):
        name = match.group(2).rstrip(".")
        repos.add(f"{match.group(1)}/{name.removesuffix('.git')}")
    for match in _REPO_LABEL_RE.finditer(blob):
        repos.add(match.group(1).rstrip("."))
    if hints:
        repos.update(_repo_hints(blob))


def _paths_overlap(a: str, b: str) -> bool:
    """Equal, or one is the tail of the other (``gate.yml`` vs ``.github/workflows/gate.yml``)."""
    a, b = a.lower(), b.lower()
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


# --- conflicts ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Conflict:
    """An open card that already holds this card's subject."""

    task_id: str
    title: str
    status: str
    assignee: Optional[str]
    kind: str        # "branch" | "file"
    subject: str
    created_at: int = 0

    def line(self) -> str:
        who = self.assignee or "unassigned"
        return f"{self.task_id} ({who}, {self.status}) holds {self.kind} {self.subject}"


def _task_claims(row: Any) -> Claims:
    return extract_claims(row["title"], row["body"], branch_name=row["branch_name"])


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def find_conflicts(
    conn: Any,
    claims: Claims,
    role: str,
    *,
    exclude: Iterable[str] = (),
    already_gated_by: Iterable[str] = (),
    allow_file_overlap: bool = False,
) -> list[Conflict]:
    """Open cards of the same ``role`` that already hold one of ``claims``.

    ``already_gated_by`` (the parents this card is being filed under) is not a conflict: the
    dependency edge is exactly the remedy. A card already linked to the new one in either
    direction is skipped for the same reason. ``allow_file_overlap`` drops file-only hits (the
    card declared ``overlap: allowed``); a shared branch is never excused.
    """
    if not claims:
        return []
    gated_by = {str(x) for x in already_gated_by if x}
    skip = {str(x) for x in exclude} | gated_by
    marks = ",".join("?" * len(OPEN_STATUSES_EXCLUDED))
    rows = conn.execute(
        f"SELECT id, title, body, status, assignee, branch_name, created_at FROM tasks "
        f"WHERE status NOT IN ({marks})",
        OPEN_STATUSES_EXCLUDED,
    ).fetchall()
    if not rows:
        return []
    # Only the declared parents (and their immediate neighbours) are "already a dependency".
    # Expanding from ``exclude`` would hide the very card we are asking about.
    related = _linked_ids(conn, gated_by) if gated_by else set()
    conflicts: list[Conflict] = []
    for row in rows:
        tid = row["id"]
        if tid in skip or tid in related:
            continue
        if card_role(_row_get(row, "assignee"), _row_get(row, "title")) != role:
            continue
        other = _task_claims(row)
        if claims.repos and other.repos and not (claims.repos & other.repos):
            continue  # same file name, different repo: not the same work
        hit = _first_overlap(claims, other)
        if hit is None:
            continue
        kind, subject = hit
        if allow_file_overlap and kind == "file":
            continue
        conflicts.append(Conflict(
            task_id=tid, title=(_row_get(row, "title") or "").strip()[:120],
            status=row["status"], assignee=_row_get(row, "assignee"),
            kind=kind, subject=subject, created_at=int(_row_get(row, "created_at") or 0),
        ))
    conflicts.sort(key=lambda c: (c.kind != "branch", -c.created_at))
    return conflicts


def _first_overlap(mine: Claims, theirs: Claims) -> Optional[tuple[str, str]]:
    shared = mine.branches & theirs.branches
    if shared:
        return "branch", min(shared)
    for a in sorted(mine.files):
        for b in sorted(theirs.files):
            if _paths_overlap(a, b):
                return "file", a if len(a) >= len(b) else b
    return None


def _linked_ids(conn: Any, ids: set[str]) -> set[str]:
    """Ids already linked to any of ``ids`` through a parent/child edge (one hop either way)."""
    if not ids:
        return set()
    marks = ",".join("?" * len(ids))
    out: set[str] = set()
    for row in conn.execute(
        f"SELECT parent_id, child_id FROM task_links WHERE parent_id IN ({marks}) "
        f"OR child_id IN ({marks})", (*ids, *ids),
    ):
        out.add(row["parent_id"])
        out.add(row["child_id"])
    return out


def reason_line(conflicts: Iterable[Conflict]) -> str:
    """One line for the board: what already holds this card's subject."""
    lines = [c.line() for c in conflicts]
    return "already in flight: " + "; ".join(lines)


# --- GitHub: the part the sandbox cannot see -------------------------------------------------

def resolve_repo(*texts: Optional[str], hint: Optional[str] = None) -> Optional[str]:
    """``owner/repo`` from an explicit hint, a github.com URL in the card, or a ``repo:`` label."""
    if hint and hint.count("/") == 1:
        return hint.removesuffix(".git")
    for text in texts:
        if not text:
            continue
        match = _GITHUB_REPO_RE.search(str(text))
        if match:
            return f"{match.group(1)}/{match.group(2).removesuffix('.git')}"
        match = _REPO_LABEL_RE.search(str(text))
        if match:
            return match.group(1)
    return None


def open_prs(repo: str, *, timeout: int = 20) -> tuple[Optional[list[dict]], str]:
    """Open PRs' ``number/headRefName/url/files`` for ``repo``.

    Returns ``(prs, note)``; ``prs is None`` with a note when GitHub cannot be asked (no gh,
    no auth, no network). Never raises: the caller degrades to board-only and says so.
    """
    cmd = ["gh", "pr", "list", "-R", repo, "--state", "open", "--limit", "50",
           "--json", "number,headRefName,url,files,title"]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, "gh not installed here"
    except subprocess.TimeoutExpired:
        return None, f"gh timed out after {timeout}s"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"gh failed: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, f"gh exit {proc.returncode}: {detail[0][:160] if detail else 'no output'}"
    try:
        data = json.loads(proc.stdout or "[]")
    except ValueError:
        return None, "gh returned non-JSON"
    prs = []
    for pr in data if isinstance(data, list) else []:
        paths = [f.get("path") for f in (pr.get("files") or []) if isinstance(f, dict) and f.get("path")]
        prs.append({
            "number": pr.get("number"), "head": pr.get("headRefName"), "url": pr.get("url"),
            "title": pr.get("title"), "files": paths,
        })
    return prs, f"{len(prs)} open"


def pr_conflicts(claims: Claims, prs: Iterable[dict]) -> list[dict]:
    """Open PRs whose head branch is a claimed branch, or whose diff touches a claimed file."""
    hits = []
    for pr in prs:
        head = (pr.get("head") or "").lower()
        by_branch = bool(head) and head in claims.branches
        touched = []
        if not by_branch:
            for path in pr.get("files") or []:
                if any(_paths_overlap(path, mine) for mine in claims.files):
                    touched.append(path)
        if by_branch or touched:
            hits.append({
                "kind": "pr", "number": pr.get("number"), "url": pr.get("url"),
                "head": pr.get("head"), "title": pr.get("title"),
                "subject": pr.get("head") if by_branch else touched[0],
                "matched_by": "branch" if by_branch else "file",
            })
    return hits


def claim_check(
    conn: Any,
    *,
    task_id: Optional[str] = None,
    repo: Optional[str] = None,
    branch: Optional[str] = None,
    files: Iterable[str] = (),
    check_github: bool = True,
    role: Optional[str] = None,
) -> dict:
    """Full preflight for one card: board conflicts plus open PRs on the same subject."""
    row = None
    if task_id:
        row = conn.execute(
            "SELECT id, title, body, status, assignee, branch_name, project_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    card_title = row["title"] if row is not None else None
    card_body = row["body"] if row is not None else None
    branch_hint = branch or (_row_get(row, "branch_name") if row is not None else None)
    file_list = [str(f) for f in files if f]
    claims = extract_claims(card_title, card_body, branch_name=branch_hint,
                            extra_branch=branch, extra_files=file_list)
    role = role or card_role(_row_get(row, "assignee") if row is not None else None,
                             _row_get(row, "title") if row is not None else None)
    body = _row_get(row, "body") if row is not None else None
    board_hits = find_conflicts(
        conn, claims, role, exclude=[row["id"]] if row is not None else (),
        allow_file_overlap=overlaps_allowed(body))
    resolved = repo or resolve_repo(card_title, card_body) or (min(claims.repos) if claims.repos else None)
    payload: dict[str, Any] = {
        "task_id": task_id, "role": role, "claims": {
            "branches": sorted(claims.branches), "files": sorted(claims.files),
            "repos": sorted(claims.repos)},
        "board": [{"kind": c.kind, "task_id": c.task_id, "title": c.title,
                   "status": c.status, "subject": c.subject} for c in board_hits],
        "github": None,
    }
    pr_hits: list[dict] = []
    if check_github and (claims.branches or claims.files):
        if not resolved:
            payload["github"] = {"checked": False, "note": "no repo on this card (add a github.com URL)"}
        else:
            prs, note = open_prs(resolved)
            if prs is None:
                payload["github"] = {"checked": False, "repo": resolved, "note": note}
            else:
                pr_hits = pr_conflicts(claims, prs)
                payload["github"] = {"checked": True, "repo": resolved, "note": note,
                                     "open_prs": len(prs), "matches": pr_hits}
    elif check_github:
        payload["github"] = {"checked": False, "note": "card names no branch or file"}
    # A board card in this lane, or an open PR on the SAME head branch, is the same work: stop.
    # An open PR that merely touches one of the same files on a different branch is a signal, not
    # a verdict — ``JourneyController.php`` is in three open PRs at any moment, and stopping on
    # that would refuse real work.
    branch_prs = [p for p in pr_hits if p.get("matched_by") == "branch"]
    file_prs = [p for p in pr_hits if p.get("matched_by") != "branch"]
    payload["verdict"] = "claimed" if (board_hits or branch_prs) else (
        "overlapping" if file_prs else "clear")
    if payload["verdict"] == "claimed":
        bits = [c.line() for c in board_hits]
        bits += [f"PR #{p['number']} ({p['url']}) is the same head branch {p['subject']}"
                 for p in branch_prs]
        payload["reason"] = "already in flight: " + "; ".join(bits)
    elif payload["verdict"] == "overlapping":
        bits = [f"PR #{p['number']} ({p['url']}) also touches {p['subject']}" for p in file_prs]
        payload["reason"] = ("open but on another branch — confirm it is not the same change: "
                             + "; ".join(bits))
    else:
        payload["reason"] = "no open card or PR on " + (
            claims.summary() if claims else "this subject (nothing to claim)")
    return payload
