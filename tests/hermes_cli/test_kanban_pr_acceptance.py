"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main",
                    "state": state.get("state", "OPEN")}}}}
            elif "/rules/branches/" in self.path:
                # Required checks as an App token can read them (no admin needed).
                value = [[]] if state.get("no_required") else [[{
                    "type": "required_status_checks",
                    "parameters": {"required_status_checks": [
                        {"context": "required", "integration_id": 1}]}}]]
            elif "/branches/" in self.path:
                value = {"protected": True, "protection": {"required_status_checks": {
                    "checks": [], "contexts": [], "enforcement_level": "off"}}}
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                value = [{"total_count": 100 + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(100)]}, {"total_count": 100 + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = [[]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            else:
                self.send_error(404)
                return
            if state.get("denied") and any(k in self.path for k in ("/check-runs", "/statuses")):
                # A fine-grained PAT without checks/statuses read: the exact refusal shape
                # gh reports, which _api must classify as `auth`, never `infra`. A minted App
                # token is accepted, so the fallback path is exercised end to end.
                auth = self.headers.get("Authorization", "")
                if auth != f"token {state.get('minted_token', 'ghs_minted_read_only')}":
                    body = json.dumps({"message": "Resource not accessible by personal access token",
                                       "status": "403"}).encode()
                    self.send_response(403)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # Mimics real `gh`: sends the credential it would use (so the fake API can refuse a PAT
    # while accepting the minted App token), and prints gh's own refusal shape on stderr —
    # `gh: <message> (HTTP 403)` — which is what `_api` classifies.
    gh.write_text(f"#!{sys.executable}\nimport json, os, sys, urllib.error, urllib.request\n"
                  f"u = 'http://127.0.0.1:{server.server_port}/' + sys.argv[2]\n"
                  "req = urllib.request.Request(u)\n"
                  "tok = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')\n"
                  "if tok:\n"
                  "    req.add_header('Authorization', 'token ' + tok)\n"
                  "try:\n"
                  "    print(urllib.request.urlopen(req).read().decode())\n"
                  "except urllib.error.HTTPError as e:\n"
                  "    body = e.read().decode()\n"
                  "    print(body)\n"
                  "    msg = json.loads(body).get('message', '')\n"
                  "    sys.stderr.write(f'gh: {msg} (HTTP {e.code})\\n')\n"
                  "    sys.exit(1)\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.platforms("linux")
def test_pr_completion_requires_current_required_evidence(github):
    with connect() as conn:
        for conclusion in ("failure", "pending", "cancelled", "timed_out", "action_required", "neutral", "skipped", None, "success"):
            github.update(conclusion=conclusion, head="a" * 40)
            tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
            ok = kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert ok is (conclusion == "success")
            task = kb.get_task(conn, tid)
            assert (task.status == "done") is ok
            receipts = [json.loads(r[0]) for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
            assert receipts and receipts[-1]["head_sha"] == "a" * 40
            if not ok:
                assert task.status in {"running", "ready", "blocked", "review"}
                assert "retry" in receipts[-1]["recovery"]
                assert receipts[-1]["checks"][0]["id"] == 42
        for fault in ("missing", "stale", "head_change"):
            github.update(conclusion="success", head="a" * 40)
            github[fault] = True
            tid = kb.create_task(conn, title=fault, completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).status != "done"
            github.pop(fault)
        # Omission and a sibling repository cannot downgrade the stored declaration.
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, summary="local green")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/other/repo/pull/7"})
        before = len(github["requests"])
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        assert kb.complete_task(conn, local, summary="https://github.com/acme/repo/pull/7 is background context")
        assert len(github["requests"]) == before


@pytest.mark.platforms("linux")
def test_acceptance_receipts_and_terminal_write_share_run_ownership(github):
    with connect() as conn:
        for conclusion in ("success", "failure"):
            tid = kb.create_task(conn, title="race", completion_contract="acme/repo")
            owner = kb.claim_task(conn, tid)
            run_id = owner.current_run_id
            def reclaim():
                with connect() as rival:
                    assert kb.block_task(rival, tid, reason="Reassigned during acceptance")
                    assert kb.unblock_task(rival, tid)
                    github["replacement"] = kb.claim_task(rival, tid).current_run_id
            github.update(conclusion=conclusion, race=reclaim)
            assert not kb.complete_task(conn, tid, result="done", expected_run_id=run_id,
                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).current_run_id == github["replacement"]
            assert github["replacement"] != run_id
            assert kb.get_task(conn, tid).status != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")


# --- #122689: acceptance must read the repo as the ASSIGNEE profile's gh login ---

@pytest.mark.platforms("posix")
def test_acceptance_runs_gh_as_the_assignee_profile(tmp_path, monkeypatch):
    """The gh child env carries the assignee's own GH credentials (its .env),
    never the ambient/launch residue, and an invisible repo is classified
    `auth` naming the repository — not a retryable `infra` failure."""
    from pathlib import Path

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GH_TOKEN=b-token\n", encoding="utf-8")
    # Ambient residue that must NOT decide the login.
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport json, os\n"
                  f"json.dump(dict(os.environ), open({str(env_dump)!r}, 'w'))\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"
        receipts = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
        assert receipts[-1]["classification"] == "auth"
        assert "acme/repo" in receipts[-1]["detail"]
    captured = json.loads(env_dump.read_text())
    assert captured["GH_TOKEN"] == "b-token"
    assert captured.get("GH_CONFIG_DIR") != "/nonexistent/launch/gh"
    assert "credentials" in (kb.get_task(conn, tid).last_failure_error or "")


@pytest.mark.platforms("posix")
def test_assignee_without_own_gh_login_never_falls_through_to_ambient_login(tmp_path, monkeypatch):
    """An assignee profile with no GH_TOKEN/GH_CONFIG_DIR of its own must not inherit the
    launch user's ~/.config/gh (HOME/XDG_CONFIG_HOME stay the launch process's): gh is pinned
    to a profile-owned config dir, its 'not logged in' exit is classified `auth` naming the profile."""
    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "launch-xdg"))

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # Real gh: GH_CONFIG_DIR wins; a config dir without hosts.yml means "not logged in" (exit 4).
    gh.write_text(f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
                  f"pathlib.Path({str(env_dump)!r}).write_text(json.dumps(dict(os.environ)), encoding='utf-8')\n"
                  "if 'GH_CONFIG_DIR' in os.environ and not os.path.exists(os.environ['GH_CONFIG_DIR']):\n"
                  "    sys.exit(4)\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "'b'" in receipt["detail"] and "no login" in receipt["detail"]
    captured = json.loads(env_dump.read_text(encoding="utf-8-sig"))
    assert captured["GH_CONFIG_DIR"] == str(assignee_home / "gh")
    assert "GH_TOKEN" not in captured and "GITHUB_TOKEN" not in captured


def test_assigned_card_with_unresolvable_profile_is_auth_not_ambient(tmp_path, monkeypatch):
    """A card assigned to a profile that no longer exists must not run gh as the completing
    process's ambient login: classification `auth` naming the profile, gh never invoked."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))  # any gh spawn would fail as infra
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-ghost", completion_contract="acme/repo", assignee="ghost")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "'ghost'" in receipt["detail"] and "cannot be resolved" in receipt["detail"]


# --- minting a read-only App token when the profile's own token cannot read checks ---

@pytest.mark.platforms("posix")
def test_pat_without_checks_scope_uses_a_minted_ci_token(github, tmp_path, monkeypatch):
    """A profile whose PAT is refused on check-runs/statuses must still reach `success` on a
    merged green PR: the checker mints a repo-narrowed read-only App token for that one read
    and never stores it. The token is not in the receipt and not in the profile's .env."""
    from hermes_cli import kanban_pr_acceptance as kpa

    launch_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    # A fine-grained PAT: reads pulls/rules, is refused on checks/statuses.
    (assignee_home / ".env").write_text("GH_TOKEN=pat-token\n", encoding="utf-8")

    minted = {"count": 0, "token": "ghs_minted_read_only"}

    def fake_mint(profile_home, repo):
        minted["count"] += 1
        assert repo == "acme/repo"
        return minted["token"]

    monkeypatch.setattr(kpa, "_mint_ci_token", fake_mint)
    # The shim sends GH_TOKEN as the request's auth marker so the server can refuse the PAT.
    github.update(denied=True, state="MERGED")
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        ok = kb.complete_task(conn, tid, result="done",
                              metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
        assert ok, receipt
        assert receipt["classification"] == "success"
        assert kb.get_task(conn, tid).status == "done"
    assert receipt["head_sha"] == "a" * 40
    assert receipt["checks"][0]["classification"] == "success"
    assert minted["count"] >= 1
    # The mint is per-call: never persisted, never in the receipt.
    assert minted["token"] not in json.dumps(receipt)
    assert "pat-token" not in json.dumps(receipt)
    assert "pat-token" in (assignee_home / ".env").read_text(encoding="utf-8")


@pytest.mark.platforms("posix")
def test_pat_without_checks_scope_and_no_mint_is_auth_naming_the_mint(github, tmp_path, monkeypatch):
    """A refusal that cannot be recovered by minting is `auth` — and the detail names the mint
    as the thing to fix, not "infra" (which would read as a retryable blip)."""
    from hermes_cli import kanban_pr_acceptance as kpa

    launch_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GH_TOKEN=pat-token\n", encoding="utf-8")
    monkeypatch.setattr(kpa, "_mint_ci_token", lambda profile_home, repo: None)
    github.update(denied=True)
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "mint" in receipt["detail"] and "ci" in receipt["detail"]
    assert "infra" not in receipt["detail"]


@pytest.mark.platforms("posix")
def test_a_red_required_check_is_failure_even_when_a_mint_succeeds(github, tmp_path, monkeypatch):
    """Minting must never soften the verdict: a red required check is `failure`."""
    from hermes_cli import kanban_pr_acceptance as kpa

    launch_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GH_TOKEN=pat-token\n", encoding="utf-8")
    monkeypatch.setattr(kpa, "_mint_ci_token", lambda profile_home, repo: "ghs_minted_read_only")
    github.update(denied=True, conclusion="failure")
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "failure"
    assert receipt["ok"] is False
    assert receipt["checks"][0]["conclusion"] == "failure"


@pytest.mark.platforms("posix")
def test_mint_is_fenced_by_the_configured_repo_allow_list(tmp_path, monkeypatch):
    """A GitHub App installation is usually granted on EVERY repo in the org, so the mint
    request's `repositories` narrowing is not the boundary. Without a configured allow-list the
    checker must mint nothing at all (fail closed) — otherwise it holds a token for whatever
    repo a card's contract names, wider than the fine-grained PAT it replaces."""
    from hermes_cli import kanban_pr_acceptance as kpa

    launch_home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GH_TOKEN=pat-token\n", encoding="utf-8")
    # No kanban.pr_acceptance.app_mint_repos configured -> nothing is mintable.
    monkeypatch.setattr(kpa, "_mint_allow_list", lambda: set())
    monkeypatch.setattr(kpa, "_app_credentials", lambda home: {"GITHUB_APP_ID": "1",
        "GITHUB_APP_PRIVATE_KEY_PATH": "/nonexistent.pem"})
    assert kpa._mint_ci_token(str(assignee_home), "acme/repo") is None
    # A repo inside the fence is allowed through to the mint attempt (which fails here only
    # because the key path is fake, proving the fence is not a blanket refusal).
    monkeypatch.setattr(kpa, "_mint_allow_list", lambda: {"acme/repo"})
    assert kpa._mint_ci_token(str(assignee_home), "acme/repo") is None  # fake key
    monkeypatch.setattr(kpa, "_installation_token", lambda creds, repo, jwt: ("ghs_x", 0.0))
    pytest.importorskip("jwt")
    assert kpa._mint_ci_token(str(assignee_home), "acme/repo") == "ghs_x"
    # ...and a sibling repo stays outside.
    assert kpa._mint_ci_token(str(assignee_home), "acme/other") is None


@pytest.mark.platforms("posix")
def test_minted_token_expiry_is_read_as_utc_not_local_time(tmp_path, monkeypatch):
    """GitHub's `expires_at` is UTC. Parsing it with `time.mktime` (which reads a struct_time as
    LOCAL) cached a dead token for the host's UTC offset — 8h on a PST host — so every completion
    in that window reused an expired token and read `auth`. This drives the REAL
    `_installation_token` parser and asserts the returned expiry equals the UTC instant."""
    import calendar
    import urllib.request

    from hermes_cli import kanban_pr_acceptance as kpa

    expires_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600))
    expected = float(calendar.timegm(time.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ")))

    class FakeJwt:
        @staticmethod
        def encode(payload, key, algorithm):
            return "signed-jwt"

    class FakeResponse:
        status = 201

        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps({"token": "ghs_tz", "expires_at": expires_at}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    key = tmp_path / "key.pem"
    key.write_text("key", encoding="utf-8")
    creds = {"GITHUB_APP_ID": "1", "GITHUB_APP_PRIVATE_KEY_PATH": str(key),
             "GITHUB_APP_INSTALLATION_ID": "99"}

    # tests/conftest.py pins TZ=UTC for determinism, which HIDES this bug: mktime and timegm
    # agree at UTC. Switch the process to a real non-UTC zone for this test, then restore.
    original_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        assert (time.mktime(time.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ")) - expected) != 0, \
            "this host's TZ must actually differ from UTC for the regression to bite"
        token, expires = kpa._installation_token(creds, "acme/repo", FakeJwt)
    finally:
        if original_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original_tz)
        time.tzset()

    assert token == "ghs_tz"
    assert expires == expected, "expiry must be the UTC instant, not the local-clock reading"
    # A one-hour token must never be cached as if it expired later than ~1h from now.
    assert expires < time.time() + 7200, "cache would outlive a 1h token"

