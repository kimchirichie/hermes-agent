"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.

``gh`` runs as the card's assignee profile (``profile_home``), not the ambient
login: :func:`_gh_env` resolves that profile's own credentials/config for the
subprocess — a multi-profile host's default ``gh`` login cannot read another
org's private repos (#122689).
"""
from __future__ import annotations

import calendar
import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False,
         profile_home: str | None = None, token: str | None = None):
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate", "--slurp"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30,
                                check=True, env=_gh_env(profile_home, token))
    except subprocess.CalledProcessError as exc:
        # 401/403/404 = the login cannot see this repository (wrong profile identity
        # or missing grant), not a transient API failure. Persist only the status
        # code + endpoint, never gh's stderr (credentials/host details).
        denied = re.search(r"HTTP (40[134])", exc.stderr or "")
        if denied:
            raise _GateAuthError(f"HTTP {denied[1]} on {endpoint.split('?')[0]}") from None
        # GraphQL answers a permission refusal with HTTP 200 and an `errors` array, so gh
        # exits 0 on the transport and its stderr carries no status line. Without this the
        # refusal fell through to the broad `infra` handler and read as a retryable blip
        # (measured on OminousApp: the branch-protection field is admin-only).
        if re.search(r'"type"\s*:\s*"(FORBIDDEN|INSUFFICIENT_SCOPES|NOT_FOUND)"', exc.stdout or ""):
            raise _GateAuthError(f"GraphQL permission refusal on {endpoint.split('?')[0]}") from None
        if exc.returncode == 4:  # gh's authentication-required exit: this profile has no login
            raise _GateAuthError(f"gh has no login for {endpoint.split('?')[0]}") from None
        raise
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        # Same shape when gh exits 0: a partially-denied query is not evidence.
        raise _GateAuthError(f"GraphQL returned errors for {endpoint.split('?')[0]}")
    return value


class _GateAuthError(RuntimeError):
    """gh was refused at HTTP 401/403/404 (or GraphQL returned a permission error):
    this login cannot see the repo — an identity problem to fix, not an
    infrastructure blip to retry."""


# Read-only permissions the acceptance reads need. Never write: this token exists so the
# checker can PROVE a check passed, not so it can act on the repository.
_CI_PERMISSIONS = {
    "contents": "read", "pull_requests": "read", "checks": "read",
    "statuses": "read", "actions": "read", "metadata": "read",
}
_APP_ENV = ("GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY_PATH", "GITHUB_APP_INSTALLATION_ID")
# (profile_home, repo) -> (token, expiry epoch). An installation token lasts one hour; a
# single completion run may need several reads, and each is a refusal-then-retry pair.
_CI_TOKEN_CACHE: dict[tuple[str, str], tuple[str, float]] = {}
_CI_TOKEN_SKEW_SECONDS = 60


def _mint_allow_list() -> set[str]:
    """Repos this host permits the acceptance checker to mint for, from
    ``kanban.pr_acceptance.app_mint_repos``. FAIL-CLOSED: unset, empty, or unreadable means
    no mint at all.

    A GitHub App installation is commonly granted on every repository in the org, so the
    ``repositories: [repo]`` narrowing in the mint request is NOT the boundary — GitHub will
    happily mint for any repo the installation covers. Without this list the checker would
    hold a token for whatever repo a card's contract named, which is wider than the
    fine-grained PAT it is replacing. The list keeps the new path inside the same fence.
    """
    try:
        from hermes_cli.config import load_config
        configured = ((load_config() or {}).get("kanban") or {}).get("pr_acceptance") or {}
        repos = configured.get("app_mint_repos")
    except Exception:
        return set()
    if not isinstance(repos, (list, tuple)):
        return set()
    return {str(r).strip().lower() for r in repos if isinstance(r, str) and r.strip()}


def _app_credentials(profile_home: str) -> dict[str, str]:
    """The App credentials *this profile* declares (its ``.env`` + external secret sources).

    Read through the profile's own secret scope, never ``os.environ``: a multiplexed gateway
    serves many profiles from one process, and the launch profile's credentials must not
    mint a token for another profile's card."""
    from agent.secret_scope import build_profile_secret_scope
    secrets = build_profile_secret_scope(Path(profile_home))
    return {name: secrets[name] for name in _APP_ENV if secrets.get(name)}


def _mint_ci_token(profile_home: str | None, repo: str) -> str | None:
    """A repo-narrowed, read-only GitHub App installation token for *repo*, or None.

    None means "this profile declares no App identity, or the mint did not succeed" — the
    caller then reports the original refusal and names the mint, so an operator knows which
    of the two to fix. The value is returned to the caller only: it is never persisted,
    logged, or put in a receipt; it reaches exactly one child's environment.
    """
    if not profile_home:
        return None  # the ambient login is not ours to widen
    if repo.lower() not in _mint_allow_list():
        return None  # outside the configured fence: fail closed, report the original refusal
    key = (str(profile_home), repo)
    cached = _CI_TOKEN_CACHE.get(key)
    if cached and time.time() < cached[1] - _CI_TOKEN_SKEW_SECONDS:
        return cached[0]
    try:
        creds = _app_credentials(profile_home)
    except Exception:  # a scope read must never turn an acceptance read into a crash
        return None
    if not all(creds.get(n) for n in _APP_ENV[:2]):
        return None  # no App identity declared: nothing to mint with
    try:
        import jwt  # PyJWT: an optional dependency, only present for App identity
    except ImportError:
        return None
    token, expires_at = _installation_token(creds, repo, jwt)
    if token is None:
        return None
    _CI_TOKEN_CACHE[key] = (token, expires_at)
    return token


def _installation_token(creds: dict[str, str], repo: str, jwt_module) -> tuple[str | None, float]:
    """``(token, expiry_epoch)`` for a read-only, *repo*-narrowed installation token.

    ``(None, 0)`` on any failure: minting is a fallback, never the acceptance verdict."""
    import urllib.error
    import urllib.request

    owner, _, repo_name = repo.partition("/")

    try:
        key_file = Path(creds["GITHUB_APP_PRIVATE_KEY_PATH"])
        key_text = key_file.read_text(encoding="utf-8-sig")
        now = int(time.time())
        assertion = jwt_module.encode(
            {"iat": now - 60, "exp": now + 540, "iss": creds["GITHUB_APP_ID"]},
            key_text, algorithm="RS256",
        )
    except Exception:  # signing failures are a no-mint, never an acceptance verdict
        return None, 0.0

    def call(method: str, path: str, payload: dict | None = None):
        request = urllib.request.Request(
            "https://api.github.com" + path, method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": f"Bearer {assertion}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "hermes-kanban-acceptance"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, {}
        except (urllib.error.URLError, OSError, ValueError):
            return 0, {}

    installation_id = creds.get("GITHUB_APP_INSTALLATION_ID")
    if not installation_id:
        status, installations = call("GET", "/app/installations")
        if status != 200 or not isinstance(installations, list):
            return None, 0.0
        installation_id = next((str(i["id"]) for i in installations
                                if str((i.get("account") or {}).get("login", "")).lower() == owner.lower()), None)
        if not installation_id:
            return None, 0.0
    status, body = call("POST", f"/app/installations/{installation_id}/access_tokens",
                        {"repositories": [repo_name], "permissions": dict(_CI_PERMISSIONS)})
    if status != 201 or not body.get("token"):
        return None, 0.0
    try:
        # GitHub's `expires_at` is UTC (`2026-11-14T11:00:00Z`). `time.mktime` interprets a
        # struct_time as LOCAL time, so on a PST host every minted token stayed cached 8h past
        # its real expiry — the cache then handed out a dead token and the retry pass read
        # `auth` for hours after the first mint. `calendar.timegm` is the UTC-correct inverse.
        expires = float(calendar.timegm(time.strptime(body.get("expires_at", ""), "%Y-%m-%dT%H:%M:%SZ")))
    except (ValueError, TypeError, OverflowError):
        expires = time.time() + 3500.0  # undocumented shape: fall back to ~58 min
    return body["token"], expires


def _gh_env(profile_home: str | None, token: str | None = None) -> dict[str, str] | None:
    """Child env for ``gh``: the card's profile identity when one is resolvable.

    The completion boundary runs in the worker (assignee), the CLI, or a
    reviewer/dispatcher turn, so an ambient ``gh`` login is whichever process
    happened to call it (#122689). ``served_profile_child_env(inherit_credentials=True)``
    is the seam for "this child acts for that profile": it scrubs the launch
    profile's credential residue and overlays the target profile's own
    ``GH_TOKEN``/``GH_CONFIG_DIR`` (its ``.env`` + external secret sources).
    ``None`` keeps the ambient env — unassigned cards behave exactly as before.

    *token* replaces the resolved login for this ONE child (a minted App installation
    token); the profile's own values are dropped so ``gh`` cannot prefer them.
    """
    if not profile_home:
        return None
    from tools.environments.local import _is_routed_home, hermes_subprocess_env, served_profile_child_env
    base = hermes_subprocess_env(inherit_credentials=True)
    routed = _is_routed_home(profile_home)
    if routed:
        # gh's config dir decides which login `gh api` uses, yet it is a path, not a
        # credential, so no scrub list sees it; the target's own value is overlaid from its .env.
        base.pop("GH_CONFIG_DIR", None)
    env = served_profile_child_env(base=base, target_home=profile_home, inherit_credentials=True)
    if routed and not (env.keys() & {"GH_TOKEN", "GITHUB_TOKEN", "GH_CONFIG_DIR"}):
        # HOME/XDG_CONFIG_HOME are still the launch process's: without a login of its own the
        # child would fall through to ~/.config/gh/hosts.yml — the ambient login. Pin gh's config
        # to a profile-owned dir so it fails "not logged in" (exit 4 -> auth) instead.
        env["GH_CONFIG_DIR"] = str(Path(profile_home) / "gh")
    if token:
        env.pop("GITHUB_TOKEN", None)
        env["GH_TOKEN"] = token
    return env



def _assignee_profile_home(assignee: str | None) -> str | None:
    """Home whose ``gh`` login must read the contract repo — the assignee's, resolved
    exactly as the dispatcher resolves the worker's home — or None (unassigned) so the
    ambient login is used. An assigned card whose profile cannot be resolved is an
    identity failure (``auth``), never a silent fall-through to the ambient login."""
    if not assignee:
        return None
    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env
    try:
        return resolve_profile_env(normalize_profile_name(assignee))
    except (FileNotFoundError, ValueError):
        raise _GateAuthError(f"assignee profile {assignee!r} cannot be resolved") from None


def _required_checks(repo: str, branch: str, profile_home: str | None,
                     token: str | None = None) -> set[tuple[str, int | None]]:
    """``(context, app_id)`` pairs the repository requires on *branch*.

    Two sources, both readable without admin:
    - ``GET /repos/{repo}/rules/branches/{branch}`` — repository AND organization rulesets.
      Only a ``required_status_checks`` rule contributes; the other rule types are unrelated.
    - ``GET /repos/{repo}/branches/{branch}`` -> ``.protection.required_status_checks`` —
      the classic branch-protection API. The dedicated ``/protection`` endpoint needs admin,
      but this embedded copy does not.

    Deliberately NOT the GraphQL ``baseRef.branchProtectionRule`` field: that is admin-only,
    and querying it makes the whole query fail for every non-admin login.
    """
    required: set[tuple[str, int | None]] = set()
    for page in _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100",
                     paginate=True, profile_home=profile_home, token=token):
        for rule in page:
            if rule.get("type") == "required_status_checks":
                required.update((r["context"], r.get("integration_id"))
                                for r in rule["parameters"]["required_status_checks"])
    branch_info = _api(f"repos/{repo}/branches/{quote(branch, safe='')}", profile_home=profile_home,
                       token=token)
    classic = ((branch_info or {}).get("protection") or {}).get("required_status_checks") or {}
    # `checks` pins each context to an app; `contexts` is the same set without the pin. Prefer
    # `checks` when present — emitting both would double every required context in the receipt,
    # and an unpinned duplicate would then accept a check run from any app.
    if classic.get("checks"):
        for check in classic["checks"]:
            required.add((check["context"], check.get("app_id")))
    else:
        for context in classic.get("contexts") or []:
            required.add((context, None))
    return required


def _collect(repo: str, number: int, profile_home: str | None, token: str | None,
             receipt: dict) -> dict:
    """One full evidence pass as one identity. Raises ``_GateAuthError`` on any refusal, so the
    caller can decide whether a mint is available before trusting a partial receipt."""
    owner, name = repo.split("/")
    # No `baseRef.branchProtectionRule` here: GitHub serves that field to admins only, so
    # every non-admin login (a fine-grained PAT or an App installation token) gets a
    # FORBIDDEN for the whole query and the checker could never read even the head sha.
    # Required checks come from the two REST reads below instead.
    query = f'{{repository(owner:{json.dumps(owner)},name:{json.dumps(name)}){{pullRequest(number:{number:d}){{headRefOid baseRefName state}}}}}}'
    repository = _api("graphql", query=query, profile_home=profile_home, token=token)["data"]["repository"]
    if repository is None:
        # A private repo the login cannot read resolves to null, not an error.
        raise _GateAuthError(f"HTTP 404 on graphql {repo}")
    pr = repository["pullRequest"]
    sha, branch = pr["headRefOid"], pr["baseRefName"]
    receipt["head_sha"] = sha
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or pr["state"] not in {"OPEN", "MERGED"}:
        raise ValueError("PR is closed or current head is unavailable")
    required = _required_checks(repo, branch, profile_home, token)
    receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
    if not required:
        receipt["detail"] = "No repository-required checks are configured; explicitly use a local-only contract for non-CI tasks."
        return receipt
    pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest",
                 paginate=True, profile_home=profile_home, token=token)
    runs = [run for page in pages for run in page["check_runs"]]
    if len({r["id"] for r in runs}) != pages[0]["total_count"]:
        raise ValueError("Incomplete check-run pagination")
    statuses = [{**s, "sha": sha} for page in _api(
        f"repos/{repo}/commits/{sha}/statuses?per_page=100", paginate=True,
        profile_home=profile_home, token=token) for s in page]
    outcomes = []
    for context, app_id in sorted(required, key=str):
        matching = [r for r in runs if r["name"] == context and
                    (app_id in (None, -1) or r["app"]["id"] == app_id)]
        # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
        legacy = [s for s in statuses if s["context"] == context] if app_id in (None, -1) else []
        selected = matching + ([max(legacy, key=lambda s: s["id"])] if legacy else [])
        if not selected:
            outcomes.append("missing")
            receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha})
        for check in selected:
            is_run = "conclusion" in check
            outcome = check.get("conclusion") if is_run else check["state"]
            classification = _classify(check, sha, outcome, is_run)
            outcomes.append(classification)
            receipt["checks"].append({"name": context, "id": check["id"],
                "url": check.get("html_url") or check.get("target_url"),
                "head_sha": check.get("head_sha", check.get("sha")),
                "classification": classification, "conclusion": outcome})
    # Re-read after all pages: old-head successes are never transferable.
    current = _api(f"repos/{repo}/pulls/{number}", profile_home=profile_home, token=token)
    if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and not current.get("merged")):
        receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
        return receipt
    receipt["classification"] = next((x for x in outcomes if x != "success"), "missing" if not outcomes else "success")
    receipt["ok"] = receipt["classification"] == "success"
    return receipt


def collect_acceptance(contract: str, published_pr: str | None,
                       assignee: str | None = None) -> dict:
    receipt = {"ok": False, "classification": "missing", "head_sha": None,
               "pr_url": published_pr, "checks": [],
               "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
                           "Use kanban_block if human input is needed; receipts remain on the task event log."}
    try:
        profile_home = _assignee_profile_home(assignee)
        declared = _PR.fullmatch(contract)
        url = contract if declared else published_pr
        match = _PR.fullmatch(url or "")
        if not match or (not declared and match[1] != contract) or (declared and published_pr and published_pr != contract):
            receipt["detail"] = "Supply metadata.published_pr matching the persisted completion contract."
            return receipt
        repo, number = match[1], int(match[2])
        receipt["pr_url"] = url
        try:
            return _collect(repo, number, profile_home, None, receipt)
        except _GateAuthError as first:
            # The profile's own login cannot read the evidence (typically a fine-grained PAT
            # without checks/statuses). Mint a repo-narrowed, READ-ONLY App token for THIS run
            # only and redo the whole pass as that identity — a partial retry would mix two
            # identities' evidence. A refusal after a successful mint is still `auth`.
            mint = _mint_ci_token(profile_home, repo)
            if mint is None:
                raise _GateAuthError(
                    f"{first}; no read-only ci token could be minted for {repo} "
                    f"(the profile declares no GitHub App identity, or {repo!r} is not in "
                    f"kanban.pr_acceptance.app_mint_repos)") from None
            receipt["checks"] = []  # never let a partial first pass contribute outcomes
            try:
                return _collect(repo, number, profile_home, mint, receipt)
            except _GateAuthError as second:
                # The minted token was refused too: it is dead or was revoked. Drop it from the
                # cache so the next completion mints a fresh one instead of reusing the corpse
                # until its nominal expiry.
                _CI_TOKEN_CACHE.pop((str(profile_home), repo), None)
                raise _GateAuthError(f"{first}; a read-only ci token WAS minted for {repo} and "
                                     f"also refused ({second})") from None
    except _GateAuthError as exc:
        login = f"assignee profile {assignee!r}'s gh login" if assignee else "the ambient gh login"
        receipt.update(classification="auth",
                       detail=f"GitHub refused the acceptance read ({exc}) as {login}; "
                              "fix that profile's GitHub credentials/access to the repository, then retry completion.")
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        # Never persist gh stderr (credentials/host details); the failed phase is actionable.
        receipt.update(classification="infra", detail="GitHub acceptance evidence unavailable or incomplete; check gh authentication/API access and retry.")
        return receipt


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(outcome, "infra")
