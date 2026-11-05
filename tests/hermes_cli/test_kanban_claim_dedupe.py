"""One change, one lane: a card must not act when another lane already holds its subject.

The measured failure (2026-11-05) was six cards and three PRs for one ``gate.yml`` edit. Every
lane reasoned correctly; none could see that another lane already held the work, because a
sandbox cannot see another lane's unmerged branch. These tests replay that sequence and pin the
two properties that make the fix safe:

* the second and third cards for the same change come out of filing gated behind the first,
  naming it, with no work refused; and
* cards touching different files still proceed untouched — the false-refusal guard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_claims as kc_claims
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_DEDUPE", raising=False)
    kb.init_db()
    return home


def _create(**kw) -> str:
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, **kw)


def _task(tid: str):
    with kbc.connect_closing() as conn:
        return kb.get_task(conn, tid)


def _events(tid: str) -> list:
    with kbc.connect_closing() as conn:
        return kb.list_events(conn, tid)


def _claim_conflict(tid: str) -> dict | None:
    hits = [e for e in _events(tid) if e.kind == "claim_conflict"]
    return hits[-1].payload if hits else None


# --- the replay ---------------------------------------------------------------------------

BUILD_TITLE = "gate.yml: the merge job defers to a run that cannot merge"
BUILD_BODY = (
    "`.github/workflows/gate.yml`, jobs `php`/`ui`/`merge approved head`.\n"
    "Fix the deferral predicate so the merge job only defers to a run that can merge.\n"
)
REVISION_BODY = (
    "The deferral fix on branch `ms/gate-defer-eligible-run` is incomplete: the template moved,"
    " `.github/workflows/gate.yml` did not.\n"
)


def test_replay_files_one_build_and_gates_the_other_two(kanban_home):
    """Three cards for one change: one runs, two wait on it by name."""
    first = _create(title=BUILD_TITLE, body=BUILD_BODY, assignee="ms-worker")
    second = _create(title="REVISION: gate.yml deferral fix is template-only",
                     body=REVISION_BODY, assignee="ms-worker")
    third = _create(title="Apply gate.yml for ms/gate-defer-eligible-run and open the PR",
                    assignee="mumu")

    assert _task(first).status == "ready"
    with kbc.connect_closing() as conn:
        for later in (second, third):
            assert _task(later).status == "todo", f"{later} must not start beside {first}"
            assert first in kb.parent_ids(conn, later)
        payload = _claim_conflict(later)
        assert payload is not None
        assert first in payload["holders"]
        assert "already in flight" in payload["reason"]


def test_replay_review_card_still_files_once(kanban_home):
    """A review card for the same branch is a different lane: it files and runs."""
    build = _create(title=BUILD_TITLE, body=BUILD_BODY, assignee="ms-worker")
    review = _create(title="REVIEW PR #849: gate defers only to a run that can merge",
                     body=REVISION_BODY, assignee="reviewer")
    assert _task(build).status == "ready"
    assert _task(review).status == "ready"
    assert _claim_conflict(review) is None


def test_second_review_card_for_the_same_branch_gates(kanban_home):
    """The pr-bridge filing a second review card for one branch is the duplicate to stop."""
    first = _create(title="Review PR #849: gate defers only to a run that can merge",
                    body=REVISION_BODY, assignee="reviewer")
    second = _create(title="Review PR #850 delta: gate deferral",
                     body=REVISION_BODY, assignee="reviewer")
    assert _task(first).status == "ready"
    assert _task(second).status == "todo"
    assert first in (_claim_conflict(second) or {}).get("holders", [])


# --- the false-refusal guard --------------------------------------------------------------

def test_two_cards_on_different_files_both_proceed(kanban_home):
    a = _create(title="Add retry to scripts/gate_helpers.sh", assignee="worker")
    b = _create(title="Fix tenancy config in config/tenancy.php", assignee="worker")
    assert (_task(a).status, _task(b).status) == ("ready", "ready")
    assert _claim_conflict(b) is None


def test_same_file_in_a_different_repo_both_proceed(kanban_home):
    """Every repo has a gate.yml; two repos' gate.yml are not one change."""
    oap = _create(title="OAP gate.yml: the merge job has no checkout", assignee="oap-worker")
    kiwii = _create(title="KiwiiDev gate.yml: derive the test host from config",
                    assignee="kiwii-worker")
    assert (_task(oap).status, _task(kiwii).status) == ("ready", "ready")


def test_overlap_allowed_keeps_a_parallel_change_to_one_file_ready(kanban_home):
    _create(title="Add bounded transport retry to gate.yml merge step", assignee="worker")
    second = _create(
        title="Add a second, unrelated assertion to gate.yml merge step",
        body="overlap: allowed\nThis is an independent change to the same job.\n",
        assignee="worker")
    assert _task(second).status == "ready"


def test_a_finished_card_holds_nothing(kanban_home):
    done = _create(title=BUILD_TITLE, body=BUILD_BODY, assignee="ms-worker")
    with kbc.connect_closing() as conn:
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (done,))
        conn.commit()
    later = _create(title="Apply gate.yml for ms/gate-defer-eligible-run", assignee="mumu")
    assert _task(later).status == "ready"


def test_quoting_another_lane_is_not_a_claim(kanban_home):
    """Prose that names another lane's branch/file must not gate this card behind it."""
    _create(title=BUILD_TITLE, body=BUILD_BODY, assignee="ms-worker")
    meta = _create(
        title="Stop duplicate lanes: a card must refuse to act when an open PR already covers "
              "its branch or files",
        body=("Measured: one gate.yml change produced six cards. t_581981a5 built the fix on "
              "ms/gate-defer-eligible-run and t_cd72d406 opened PR #849.\n"),
        assignee="mumu")
    assert _task(meta).status == "ready"
    assert _claim_conflict(meta) is None


def test_explicit_parent_is_the_documented_remedy(kanban_home):
    first = _create(title=BUILD_TITLE, body=BUILD_BODY, assignee="ms-worker")
    second = _create(title="Apply gate.yml for ms/gate-defer-eligible-run", assignee="mumu",
                     parents=[first])
    assert _task(second).status == "todo"
    assert _claim_conflict(second) is None  # a deliberate dependency is not a collision


def test_dedupe_switch_off_restores_the_old_behaviour(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_DEDUPE", "off")
    _create(title=BUILD_TITLE, body=BUILD_BODY, assignee="ms-worker")
    second = _create(title="Apply gate.yml for ms/gate-defer-eligible-run", assignee="mumu")
    assert _task(second).status == "ready"


# --- claim extraction ---------------------------------------------------------------------

def test_extraction_reads_the_title_and_labelled_body_only():
    claims = kc_claims.extract_claims(
        "Apply gate.yml for ms/gate-defer-eligible-run",
        "- Branch: `ms/gate-defer-eligible-run`\nDiff: 3 files, `scripts/tests/test_x.py`\n"
        "Prose mentioning app/Foo.php and red/black fabric without a label.\n",
    )
    assert claims.branches == frozenset({"ms/gate-defer-eligible-run"})
    assert "gate.yml" in claims.files
    # A backticked single path in the body is structural; bare prose paths are not.
    assert "scripts/tests/test_x.py" in claims.files
    assert "app/Foo.php" not in claims.files


def test_generic_filenames_and_prose_slashes_are_not_claims():
    claims = kc_claims.extract_claims(
        "Update README.md and the red/black swatch", "see omni-pipeline.dev for the docs")
    assert claims.files == frozenset()   # README.md is ubiquitous, the rest is prose
    assert claims.branches == frozenset()


def test_card_role_splits_the_lanes():
    assert kc_claims.card_role("reviewer", "anything") == kc_claims.REVIEW
    assert kc_claims.card_role("ms-reviewer", "anything") == kc_claims.REVIEW
    assert kc_claims.card_role("mumu", "REVIEW PR #849: gate deferral") == kc_claims.REVIEW
    assert kc_claims.card_role("mumu", "SHIP open PR #850") == kc_claims.SHIP
    assert kc_claims.card_role("worker", "Add retry to gate.yml") == kc_claims.BUILD


# --- the acting-side preflight --------------------------------------------------------------

def test_claim_check_names_the_holder_and_declines_github_without_a_repo(kanban_home):
    first = _create(title=BUILD_TITLE, body=BUILD_BODY, assignee="ms-worker")
    mine = _create(title="Apply gate.yml for ms/gate-defer-eligible-run", assignee="mumu",
                   body="no repo here")
    with kbc.connect_closing() as conn:
        payload = kc_claims.claim_check(conn, task_id=mine)
    assert payload["verdict"] == "claimed"
    assert payload["board"][0]["task_id"] == first
    assert payload["github"]["checked"] is False
    assert "no repo" in payload["github"]["note"]


def test_claim_check_degrades_when_gh_is_missing(kanban_home, monkeypatch):
    _create(title=BUILD_TITLE, body=BUILD_BODY, assignee="ms-worker")
    monkeypatch.setattr(kc_claims.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("gh")))
    with kbc.connect_closing() as conn:
        payload = kc_claims.claim_check(conn, repo="Sin-Brothers-LLC/omni-pipeline",
                                        branch="ms/other-branch", check_github=True)
    assert payload["verdict"] == "clear"
    assert payload["github"] == {"checked": False, "repo": "Sin-Brothers-LLC/omni-pipeline",
                                 "note": "gh not installed here"}


def test_a_pr_on_the_same_file_but_another_branch_is_a_signal_not_a_verdict(kanban_home, monkeypatch):
    """JourneyController.php sits in three open PRs at any moment; that must not refuse work."""
    prs = [{"number": 840, "headRefName": "wt/t_57894799-care-sop", "url": "u840", "title": "t",
            "files": [{"path": "app/Http/Controllers/JourneyController.php"}]}]
    monkeypatch.setattr(kc_claims.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(prs)))
    with kbc.connect_closing() as conn:
        payload = kc_claims.claim_check(
            conn, repo="Sin-Brothers-LLC/omni-pipeline", role="build",
            files=["app/Http/Controllers/JourneyController.php"])
    assert payload["verdict"] == "overlapping"
    assert payload["github"]["matches"][0]["matched_by"] == "file"
    assert "confirm it is not the same change" in payload["reason"]


def test_a_pr_on_the_same_head_branch_is_a_stop(kanban_home, monkeypatch):
    prs = [{"number": 850, "headRefName": "ms/gate-defer-eligible-run", "url": "u850",
            "title": "t", "files": [{"path": ".github/workflows/gate.yml"}]}]
    monkeypatch.setattr(kc_claims.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(prs)))
    with kbc.connect_closing() as conn:
        payload = kc_claims.claim_check(
            conn, repo="Sin-Brothers-LLC/omni-pipeline", branch="ms/gate-defer-eligible-run")
    assert payload["verdict"] == "claimed"
    assert payload["board"] == []
    assert "PR #850" in payload["reason"]


def test_open_prs_and_pr_conflicts_read_the_github_shape(monkeypatch):
    prs = [{"number": 850, "headRefName": "ms/gate-defer-eligible-run", "url": "u850",
            "title": "t", "files": [{"path": ".github/workflows/gate.yml"}]}]
    monkeypatch.setattr(kc_claims.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps(prs)))
    got, note = kc_claims.open_prs("Sin-Brothers-LLC/omni-pipeline")
    assert note == "1 open" and got[0]["files"] == [".github/workflows/gate.yml"]
    by_branch = kc_claims.pr_conflicts(
        kc_claims.extract_claims("Apply gate.yml for ms/gate-defer-eligible-run"), got)
    assert by_branch and by_branch[0]["number"] == 850
    assert by_branch[0]["matched_by"] == "branch"
    by_file = kc_claims.pr_conflicts(kc_claims.extract_claims("gate.yml: deferral fix"), got)
    assert by_file[0]["matched_by"] == "file"


# --- CLI ------------------------------------------------------------------------------------

def test_cli_claim_check_exits_3_when_claimed_and_0_when_clear(kanban_home, monkeypatch, capsys):
    _create(title=BUILD_TITLE, body=BUILD_BODY, assignee="ms-worker")
    mine = _create(title="Apply gate.yml for ms/gate-defer-eligible-run", assignee="mumu")

    def run(argv):
        root = argparse.ArgumentParser(prog="hermes")
        kc.build_parser(root.add_subparsers())
        return kc._cmd_claim_check(root.parse_args(["kanban", "claim-check", *argv]))

    assert run([mine, "--no-github"]) == 3
    assert "claimed" in capsys.readouterr().out
    assert run(["--branch", "nothing/here", "--no-github"]) == 0
