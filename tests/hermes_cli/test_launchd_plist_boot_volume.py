"""launchd opens a job's own paths before the job runs, so they must live on the boot volume.

A Hermes home on a removable/external volume (``~/.hermes`` -> ``/Volumes/...``) used to put
``WorkingDirectory`` and ``StandardOutPath``/``StandardErrorPath`` on that volume. macOS then denies
the removable volume to a background job that can never answer an approval prompt, the spawn dies
EX_CONFIG 78, and nothing is written anywhere — so the service is down with a silent, unlogged
failure. These tests pin the generator to boot-volume paths while leaving the gateway's own process
logs under the Hermes home.
"""
import os
import plistlib
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_cli

pytestmark = pytest.mark.platforms("macos")


def _generated(**kwargs) -> dict:
    return plistlib.loads(gateway_cli.generate_launchd_plist().encode())


def _fake_stat(st_dev_by_prefix):
    """``os.stat`` stand-in that reports a fake ``st_dev`` for paths under each prefix."""
    real_stat = os.stat

    def fake(path, *args, **kwargs):
        resolved = str(Path(path).expanduser().resolve()) if not isinstance(path, int) else str(path)
        for prefix, dev in st_dev_by_prefix.items():
            if resolved == prefix or resolved.startswith(prefix.rstrip("/") + "/"):
                result = real_stat(path, *args, **kwargs)
                return os.stat_result((result.st_mode, result.st_ino, dev, result.st_nlink,
                                       result.st_uid, result.st_gid, result.st_size,
                                       result.st_atime, result.st_mtime, result.st_ctime))
        return real_stat(path, *args, **kwargs)

    return fake


@pytest.fixture(autouse=True)
def _local_python(monkeypatch):
    """Resolve the launcher without reading the INSTALL's manifest.

    ``resolve_store_python(PROJECT_ROOT)`` reads ``<home>/manifest.json``, and a checkout installed
    inside the production home (the normal layout) is refused by the real-home I/O guard. Which
    interpreter the launcher names is irrelevant to which paths the plist declares, so stub both
    launcher builders; ``launchd_program_arguments`` (the osascript wrapping) stays real.
    """
    import hermes_cli._launchers as launchers

    monkeypatch.setattr(gateway_cli, "get_python_path", lambda: "/usr/bin/false")
    monkeypatch.setattr(
        launchers, "installation_command",
        lambda root, args=(), **kwargs: ["/fake/hermes", *args],
    )
    monkeypatch.setattr(
        launchers, "runtime_command",
        lambda root, args=(), **kwargs: ["/fake/hermes", *args],
    )


@pytest.fixture
def external_home(tmp_path, monkeypatch):
    """A Hermes home on another volume, with the boot volume mapped to the real home."""
    home = tmp_path / "external-home"
    (home / "logs").mkdir(parents=True)
    boot = str(Path.home().resolve())

    def on_boot_volume(path) -> bool:
        resolved = Path(path).expanduser().resolve()
        return not str(resolved).startswith(str(home.resolve())) and str(resolved).startswith(boot)

    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: home)
    monkeypatch.setattr(gateway_cli, "_path_is_on_boot_volume", on_boot_volume)
    return home


def test_external_home_keeps_launchd_paths_on_the_boot_volume(external_home):
    plist = _generated()

    for key in ("WorkingDirectory", "StandardOutPath", "StandardErrorPath"):
        assert not str(plist[key]).startswith(str(external_home)), (key, plist[key])
    assert plist["WorkingDirectory"] == str(Path.home())
    assert plist["StandardOutPath"] == str(Path.home() / "Library" / "Logs" / "hermes" / "gateway.log")
    assert plist["StandardErrorPath"] == str(Path.home() / "Library" / "Logs" / "hermes" / "gateway.error.log")
    # The gateway process still logs to the Hermes home: only launchd's own files moved.
    assert str(external_home / "logs" / "gateway.log") in " ".join(plist["ProgramArguments"])
    # HERMES_HOME is only ever a string to launchd, so it keeps pointing at the real home.
    assert plist["EnvironmentVariables"]["HERMES_HOME"] == str(external_home.resolve())


def test_boot_volume_home_is_unchanged(tmp_path, monkeypatch):
    home = tmp_path / "local-home"
    (home / "logs").mkdir(parents=True)
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: home)
    monkeypatch.setattr(gateway_cli, "_path_is_on_boot_volume", lambda path: True)

    plist = _generated()

    assert plist["WorkingDirectory"] == str(home.resolve())
    assert plist["StandardOutPath"] == str(home.resolve() / "logs" / "gateway.log")
    assert plist["StandardErrorPath"] == str(home.resolve() / "logs" / "gateway.error.log")


def test_path_is_on_boot_volume_compares_devices(monkeypatch):
    monkeypatch.setattr(gateway_cli.os, "stat", _fake_stat({"/Volumes": 424242}))
    assert gateway_cli._path_is_on_boot_volume("/Users/example") is True
    assert gateway_cli._path_is_on_boot_volume("/Volumes/Agents/hermes") is False


def test_path_is_on_boot_volume_judges_missing_leaf_by_ancestor(monkeypatch, tmp_path):
    # ~/.hermes/logs is on the external volume through the symlink even before logs/ exists.
    root = tmp_path / "volume-root"
    root.mkdir()
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(gateway_cli.os, "stat", _fake_stat({str(root): 424242}))
    assert gateway_cli._path_is_on_boot_volume(link / "not-created-yet") is False
