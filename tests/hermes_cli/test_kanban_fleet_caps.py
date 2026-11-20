"""The spark caps are one fleet number, read from the root home and never a profile's own."""

from hermes_cli import kanban_hold


def test_fleet_caps_come_from_the_root_section(monkeypatch):
    monkeypatch.setattr(
        kanban_hold, "_root_kanban_section",
        lambda: {"max_in_progress_per_profile": {"worker": 8}, "group_caps": {"spark": {"profiles": ["a"], "max": 3}}},
    )

    per_profile, groups = kanban_hold.fleet_caps()

    assert per_profile == {"worker": 8}
    assert groups == {"spark": {"profiles": ["a"], "max": 3}}


def test_a_profiles_own_copy_does_not_override_the_root(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
        "kanban": {"group_caps": {"spark": {"profiles": ["a"], "max": 6}}, "max_in_progress_per_profile": {"a": 9}},
    })
    monkeypatch.setattr(
        kanban_hold, "_root_kanban_section",
        lambda: {"group_caps": {"spark": {"profiles": ["a"], "max": 3}}},
    )

    per_profile, groups = kanban_hold.fleet_caps()

    assert groups["spark"]["max"] == 3
    assert per_profile is None


def test_an_unreadable_root_means_no_caps_and_no_crash(monkeypatch):
    monkeypatch.setattr(kanban_hold, "_root_kanban_section", dict)

    assert kanban_hold.fleet_caps() == (None, None)
