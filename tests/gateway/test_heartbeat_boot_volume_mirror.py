"""The gateway heartbeat also exists on the boot volume, so a wedged /Volumes cannot hide it.

``state/gateway.heartbeat`` lives under the Hermes home, which on this fleet is on another volume
(``~/.hermes`` -> ``/Volumes/Agents/hermes``). A removable-volume wedge — a deadlocked ``tccd`` is
the observed one — makes that file unreadable exactly when an agent wants to restart the gateway,
and the restart runner refuses a request without a fresh readiness snapshot. The gateway therefore
keeps a copy on the boot volume, and ``gateway-restart`` reads it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import gateway.shutdown_watchdog as shutdown_watchdog_module
from gateway.shutdown_watchdog import (
    get_loop_heartbeat_mirror_paths,
    get_loop_heartbeat_path,
    write_loop_heartbeat,
)


@pytest.fixture
def external_home(tmp_path, monkeypatch):
    """A home on another volume, with the boot volume being everything else."""
    home = tmp_path / "external-home"
    (home / "state").mkdir(parents=True)
    mirrors = (tmp_path / "boot-a" / "gateway.heartbeat", tmp_path / "boot-b" / "gateway.heartbeat")
    monkeypatch.setattr(shutdown_watchdog_module, "_HEARTBEAT_MIRRORS", mirrors)
    monkeypatch.setattr("hermes_constants.path_is_on_boot_volume",
                        lambda path: not str(Path(path)).startswith(str(home)))
    return home, mirrors


def test_boot_volume_home_needs_no_mirror(tmp_path, monkeypatch):
    home = tmp_path / "local-home"
    (home / "state").mkdir(parents=True)
    monkeypatch.setattr(shutdown_watchdog_module, "_HEARTBEAT_MIRRORS", (tmp_path / "boot" / "gateway.heartbeat",))
    monkeypatch.setattr("hermes_constants.path_is_on_boot_volume", lambda path: True)

    assert get_loop_heartbeat_mirror_paths(home) == ()


def test_external_home_mirrors_onto_the_boot_volume(external_home):
    home, mirrors = external_home

    assert get_loop_heartbeat_mirror_paths(home) == mirrors


def test_only_the_boot_volume_mirrors_are_kept(external_home, monkeypatch):
    home, _ = external_home
    still_on_the_volume = home / "state" / "mirror-candidate"
    monkeypatch.setattr(shutdown_watchdog_module, "_HEARTBEAT_MIRRORS",
                        (still_on_the_volume, Path("/boot-volume/gateway.heartbeat")))

    # A "mirror" that is itself on the volume buys nothing: it is exactly as unreadable as the original.
    assert get_loop_heartbeat_mirror_paths(home) == (Path("/boot-volume/gateway.heartbeat"),)


def test_unjudgeable_home_yields_no_mirrors(tmp_path, monkeypatch):
    def explode(path):
        raise OSError("stat failed")

    monkeypatch.setattr(shutdown_watchdog_module, "_HEARTBEAT_MIRRORS", (tmp_path / "boot" / "gateway.heartbeat",))
    monkeypatch.setattr("hermes_constants.path_is_on_boot_volume", explode)

    assert get_loop_heartbeat_mirror_paths(tmp_path / "home") == ()


def test_heartbeat_write_copies_every_mirror(external_home):
    home, mirrors = external_home

    write_loop_heartbeat(pid=4242, start_time=1700000000.0, home=home)

    canonical = json.loads(get_loop_heartbeat_path(home).read_text(encoding="utf-8"))
    assert canonical["pid"] == 4242
    for mirror in mirrors:
        copied = json.loads(mirror.read_text(encoding="utf-8"))
        assert copied["pid"] == 4242
        assert copied["start_time"] == 1700000000.0


def test_a_broken_mirror_never_costs_the_canonical_heartbeat(tmp_path, monkeypatch):
    home = tmp_path / "external-home"
    (home / "state").mkdir(parents=True)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    # mkdir under a regular file raises; the canonical heartbeat must still be written.
    monkeypatch.setattr(shutdown_watchdog_module, "_HEARTBEAT_MIRRORS",
                        (blocker / "gateway.heartbeat", tmp_path / "boot" / "gateway.heartbeat"))
    monkeypatch.setattr("hermes_constants.path_is_on_boot_volume",
                        lambda path: not str(Path(path)).startswith(str(home)))

    write_loop_heartbeat(pid=7, start_time=1700000000.0, home=home)

    assert json.loads(get_loop_heartbeat_path(home).read_text(encoding="utf-8"))["pid"] == 7
    assert json.loads((tmp_path / "boot" / "gateway.heartbeat").read_text(encoding="utf-8"))["pid"] == 7
