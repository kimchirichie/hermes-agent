"""Shared group caps: profiles in one group share a single running budget across all boards."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def kbenv(monkeypatch):
    home = tempfile.mkdtemp(prefix="kanban_group_cap_test_")
    for prof in ("alpha", "beta", "gamma", "default"):
        os.makedirs(os.path.join(home, "profiles", prof), exist_ok=True)
        with open(os.path.join(home, "profiles", prof, "config.yaml"), "w") as fh:
            fh.write("{}\n")
    monkeypatch.setenv("HERMES_HOME", home)
    for mod in list(sys.modules):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    kb.create_board(slug="default", name="Default")
    kb.create_board(slug="other", name="Other")
    return kb, kbc, kbd


def _spawn(*a, **k):
    return 12345


def _mark_running(kb, kbc, board, task_id):
    with kbc.connect_closing(board=board) as conn:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (task_id,))


def _dry(kbd, kbc, **kw):
    with kbc.connect_closing() as conn:
        return kbd.dispatch_once(conn, spawn_fn=_spawn, dry_run=True, **kw)


GROUP = {"spark": {"profiles": ["alpha", "beta"], "max": 2}}


def test_group_cap_limits_members_combined(kbenv):
    kb, kbc, kbd = kbenv
    with kbc.connect_closing() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
            kb.create_task(conn, title=f"b{i}", assignee="beta")
    res = _dry(kbd, kbc, group_caps=GROUP)
    assert len(res.spawned) == 2
    assert len(res.skipped_per_profile_capped) == 4


def test_group_cap_counts_other_boards(kbenv):
    kb, kbc, kbd = kbenv
    with kbc.connect_closing(board="other") as conn:
        t1 = kb.create_task(conn, title="r1", assignee="alpha")
        t2 = kb.create_task(conn, title="r2", assignee="beta")
    _mark_running(kb, kbc, "other", t1)
    _mark_running(kb, kbc, "other", t2)
    with kbc.connect_closing() as conn:
        kb.create_task(conn, title="x", assignee="alpha")
    res = _dry(kbd, kbc, group_caps=GROUP)
    assert res.spawned == []
    assert len(res.skipped_per_profile_capped) == 1


def test_non_members_unaffected(kbenv):
    kb, kbc, kbd = kbenv
    with kbc.connect_closing() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"g{i}", assignee="gamma")
    res = _dry(kbd, kbc, group_caps=GROUP)
    assert len(res.spawned) == 3


def test_no_group_caps_is_noop(kbenv):
    kb, kbc, kbd = kbenv
    with kbc.connect_closing() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
    assert len(_dry(kbd, kbc).spawned) == 3
    assert len(_dry(kbd, kbc, group_caps={}).spawned) == 3


def test_normalize_drops_invalid_entries():
    from hermes_cli import kanban_db_dispatch as kbd
    out = kbd.normalize_group_caps({
        "ok": {"profiles": "a, b", "max": "3"},
        "zero": {"profiles": ["a"], "max": 0},
        "empty": {"profiles": [], "max": 2},
        "junk": "x",
        "nomax": {"profiles": ["a"]},
    })
    assert out == [kbd.GroupCap("ok", None, frozenset({"a", "b"}), 3)]
    assert kbd.normalize_group_caps(None) == []


SPARK = "spark-box:deepseek-flash"
BY_MODEL = {"spark": {"model": SPARK, "max": 2}}


@pytest.fixture()
def profile_models(monkeypatch, kbenv):
    defaults: dict = {}
    monkeypatch.setattr(kbenv[2], "_profile_default_model", lambda name: defaults.get(name))
    return defaults


def test_cloud_card_on_a_spark_profile_does_not_count(kbenv, profile_models):
    kb, kbc, kbd = kbenv
    profile_models["alpha"] = SPARK
    with kbc.connect_closing() as conn:
        for i in range(2):
            t = kb.create_task(conn, title=f"c{i}", assignee="alpha", model_override="dario:sonnet")
            _mark_running(kb, kbc, "default", t)
        for i in range(3):
            kb.create_task(conn, title=f"s{i}", assignee="alpha")
    res = _dry(kbd, kbc, group_caps=BY_MODEL, reconcile_orphans=False)
    assert len(res.spawned) == 2


def test_spark_card_on_any_profile_counts(kbenv, profile_models):
    kb, kbc, kbd = kbenv
    profile_models["beta"] = "dario:sonnet"
    with kbc.connect_closing() as conn:
        for i in range(2):
            t = kb.create_task(conn, title=f"g{i}", assignee="gamma", model_override=SPARK)
            _mark_running(kb, kbc, "default", t)
        kb.create_task(conn, title="next", assignee="beta", model_override=SPARK)
    res = _dry(kbd, kbc, group_caps=BY_MODEL, reconcile_orphans=False)
    assert res.spawned == []
    assert len(res.skipped_per_profile_capped) == 1


def test_unset_model_resolves_via_profile_default(kbenv, profile_models):
    kb, kbc, kbd = kbenv
    profile_models["alpha"] = SPARK
    profile_models["beta"] = "dario:sonnet"
    with kbc.connect_closing() as conn:
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
            kb.create_task(conn, title=f"b{i}", assignee="beta")
    res = _dry(kbd, kbc, group_caps=BY_MODEL, reconcile_orphans=False)
    assert sorted(a for _, a, _ in res.spawned) == ["alpha", "alpha", "beta", "beta", "beta"]


@pytest.mark.parametrize("model_cfg,expected", [
    ({"default": SPARK}, SPARK),
    ({"model": "m"}, "m"),
    (SPARK, SPARK),
    ({}, None),
    (None, None),
])
def test_profile_default_model_parsing(monkeypatch, model_cfg, expected):
    import hermes_cli.config as cfg
    from hermes_cli import profiles
    from hermes_cli import kanban_db_dispatch as kbd
    monkeypatch.setattr(cfg, "read_user_config_raw", lambda path: {"model": model_cfg})
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: __import__("pathlib").Path("/x") / name)
    assert kbd._profile_default_model("alpha") == expected
