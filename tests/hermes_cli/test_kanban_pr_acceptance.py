"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import sys
import threading
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
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                value = [[]]
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
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.request\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "print(urllib.request.urlopen(u).read().decode())\n")
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


# --- one verdict per required context: `filter=latest` returns several runs at a head ---
#
# Fixture ids below are the REAL ones GitHub served for omni-pipeline #933 (head
# d58d012a…) on 2026-11-14: a re-run (43383656911) left a `skipped` pair beside the
# earlier passing attempt (43383286708). Taking the worst of both read `infra` on a
# merged, fully-green PR.

_SHA_933 = "d58d012aeb208b30eea15f4005776c2d966ba0d6"
_APP_933 = 15368
_REQUIRED_933 = [("npm run gate:ui", _APP_933), ("php artisan test", _APP_933)]
_RUNS_933 = [
    # required contexts: skipped re-run supersedes the earlier success
    {"id": 131745793445, "name": "npm run gate:ui", "app": {"id": _APP_933},
     "head_sha": _SHA_933, "status": "completed", "conclusion": "success"},
    {"id": 131746976051, "name": "npm run gate:ui", "app": {"id": _APP_933},
     "head_sha": _SHA_933, "status": "completed", "conclusion": "skipped"},
    {"id": 131745793128, "name": "php artisan test", "app": {"id": _APP_933},
     "head_sha": _SHA_933, "status": "completed", "conclusion": "success"},
    {"id": 131747025300, "name": "php artisan test", "app": {"id": _APP_933},
     "head_sha": _SHA_933, "status": "completed", "conclusion": "skipped"},
    # second-order effect from the same head: non-required contexts carry the same pairs
    {"id": 131746976326, "name": "merge approved head", "app": {"id": _APP_933},
     "head_sha": _SHA_933, "status": "completed", "conclusion": "skipped"},
    {"id": 131756036377, "name": "merge approved head", "app": {"id": _APP_933},
     "head_sha": _SHA_933, "status": "completed", "conclusion": "success"},
    {"id": 131746977254, "name": "red-main issue", "app": {"id": _APP_933},
     "head_sha": _SHA_933, "status": "completed", "conclusion": "skipped"},
]


def _acceptance(monkeypatch, runs, *, required=_REQUIRED_933, statuses=(), state="MERGED",
                sha=_SHA_933):
    """Run ``collect_acceptance`` against a canned GitHub, no gh, no network."""
    from hermes_cli import kanban_pr_acceptance as acc

    def fake_api(endpoint, *, query=None, paginate=False, profile_home=None):
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": sha, "baseRefName": "main", "state": state}}}}
        if "/rules/branches/" in endpoint:
            return [[{"type": "required_status_checks", "parameters": {
                "required_status_checks": [{"context": c, "integration_id": a} for c, a in required]}}]]
        if "/check-runs" in endpoint:
            return [{"total_count": len(runs), "check_runs": runs}]
        if "/statuses" in endpoint:
            return [list(statuses)]
        if "/pulls/" in endpoint:
            return {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "closed", "merged": True}
        raise AssertionError(f"unexpected endpoint {endpoint}")

    monkeypatch.setattr(acc, "_api", fake_api)
    url = "https://github.com/acme/repo/pull/933"
    return acc.collect_acceptance(url, url)


def test_skipped_rerun_does_not_poison_a_green_success(monkeypatch):
    """#933's measured shape: every required context has a `skipped` re-run at a HIGHER id
    than its `success`. A merged, fully-passing PR must read `success`, not `infra`."""
    receipt = _acceptance(monkeypatch, _RUNS_933)
    assert receipt["classification"] == "success", receipt
    assert receipt["ok"] is True
    assert receipt["head_sha"] == _SHA_933
    # One verdict per required context — the passing run, with the skipped re-run recorded
    # as context rather than folded into the outcome.
    by_name = {c["name"]: c for c in receipt["checks"]}
    assert set(by_name) == {"npm run gate:ui", "php artisan test"}
    assert by_name["npm run gate:ui"]["id"] == 131745793445
    assert by_name["npm run gate:ui"]["classification"] == "success"
    assert by_name["npm run gate:ui"]["skipped_runs"] == [131746976051]
    assert by_name["php artisan test"]["id"] == 131745793128
    assert by_name["php artisan test"]["classification"] == "success"
    assert by_name["php artisan test"]["skipped_runs"] == [131747025300]


def test_a_context_whose_only_run_is_skipped_is_missing_not_infra(monkeypatch):
    """`skipped` means the job did not run: no evidence about this head. It must read
    `missing` — never `infra`, which would blame GitHub for a workflow's own filter."""
    only_skipped = [
        {"id": 131746976051, "name": "npm run gate:ui", "app": {"id": _APP_933},
         "head_sha": _SHA_933, "status": "completed", "conclusion": "skipped"},
        {"id": 131745793128, "name": "php artisan test", "app": {"id": _APP_933},
         "head_sha": _SHA_933, "status": "completed", "conclusion": "success"},
    ]
    receipt = _acceptance(monkeypatch, only_skipped)
    assert receipt["classification"] == "missing", receipt
    assert receipt["ok"] is False
    by_name = {c["name"]: c for c in receipt["checks"]}
    assert by_name["npm run gate:ui"]["classification"] == "missing"
    assert by_name["npm run gate:ui"]["skipped_runs"] == [131746976051]
    assert by_name["php artisan test"]["classification"] == "success"


def test_newest_run_failure_wins_over_an_older_success(monkeypatch):
    """The highest-id executed run is the verdict: a re-run that failed is never masked by
    the pass it superseded, even on a MERGED PR."""
    failed = [
        {"id": 131745793445, "name": "npm run gate:ui", "app": {"id": _APP_933},
         "head_sha": _SHA_933, "status": "completed", "conclusion": "success"},
        {"id": 131746976051, "name": "npm run gate:ui", "app": {"id": _APP_933},
         "head_sha": _SHA_933, "status": "completed", "conclusion": "failure"},
        {"id": 131745793128, "name": "php artisan test", "app": {"id": _APP_933},
         "head_sha": _SHA_933, "status": "completed", "conclusion": "success"},
    ]
    receipt = _acceptance(monkeypatch, failed)
    assert receipt["classification"] == "failure", receipt
    assert receipt["ok"] is False
    by_name = {c["name"]: c for c in receipt["checks"]}
    assert by_name["npm run gate:ui"]["id"] == 131746976051
    assert by_name["npm run gate:ui"]["conclusion"] == "failure"


def test_legacy_status_survives_a_newer_skipped_duplicate(monkeypatch):
    """A legacy commit status is separate evidence, not a re-run of a check-run: a newer
    `skipped` duplicate must not discard it, and a legacy failure still decides."""
    required = [("legacy-pass", None), ("legacy-fail", None)]
    statuses = [
        {"id": 900, "context": "legacy-pass", "state": "success", "sha": _SHA_933},
        {"id": 901, "context": "legacy-fail", "state": "failure", "sha": _SHA_933},
    ]
    runs = [
        {"id": 131746976051, "name": "legacy-pass", "app": {"id": _APP_933},
         "head_sha": _SHA_933, "status": "completed", "conclusion": "skipped"},
        {"id": 131746976052, "name": "legacy-fail", "app": {"id": _APP_933},
         "head_sha": _SHA_933, "status": "completed", "conclusion": "skipped"},
    ]
    receipt = _acceptance(monkeypatch, runs, required=required, statuses=statuses)
    by_name = {c["name"]: c for c in receipt["checks"]}
    assert by_name["legacy-pass"]["classification"] == "success"
    assert by_name["legacy-pass"]["id"] == 900
    assert by_name["legacy-pass"]["skipped_runs"] == [131746976051]
    assert by_name["legacy-fail"]["classification"] == "failure"
    assert by_name["legacy-fail"]["id"] == 901
    assert receipt["classification"] == "failure"
    assert receipt["ok"] is False


def test_skipped_run_never_classifies_as_infra():
    """The mapping is explicit, not a fallthrough: `skipped` is no evidence, while a
    cancelled/timed-out run is still a real failure of the job."""
    from hermes_cli.kanban_pr_acceptance import _classify
    run = {"head_sha": _SHA_933, "status": "completed"}
    assert _classify(run, _SHA_933, "skipped", True) == "missing"
    assert _classify(run, _SHA_933, "success", True) == "success"
    assert _classify(run, _SHA_933, "failure", True) == "failure"
    assert _classify(run, _SHA_933, "cancelled", True) != "success"
    assert _classify(run, _SHA_933, "error", True) == "infra"
