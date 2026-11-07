"""The one-off runtime job's plist and cwd must be on the boot volume, not on the Hermes home.

``start_gateway_job`` is the macOS fallback that keeps an unmanaged gateway's Local Network identity
alive by running it as a transient launchd job. launchd opens the plist and ``chdir()``s to
``WorkingDirectory`` *before* the job runs, and a removable volume is denied to a background job that
can never answer an approval prompt: with the plist or the cwd under ``/Volumes`` the bootstrap fails
``5: Input/output error``, or the spawn dies EX_CONFIG 78 with nothing logged. A Hermes home that lives
on another volume (``~/.hermes`` -> ``/Volumes/...``) made this fallback dead.
"""
from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import hermes_cli.gateway_launchd_runtime as runtime
from tests.waiting import wait_until


def _capture_bootstrap(monkeypatch, captured, returncode=0):
    """Stub ``launchctl`` and record the plist bytes at the moment launchd would read them."""
    def fake_launchctl(*args, timeout):
        captured.append(args)
        if args[0] == "bootstrap":
            plist = Path(args[2])
            captured.append(("plist", str(plist), plist.read_bytes() if plist.exists() else None))
        return subprocess.CompletedProcess(list(args), returncode, "", "")

    monkeypatch.setattr(runtime, "_launchctl", fake_launchctl)


def _as_external(monkeypatch, external: Path):
    monkeypatch.setattr(runtime, "_path_is_on_boot_volume",
                        lambda path: not Path(path).expanduser().resolve().is_relative_to(external))


def _bootstrapped_plist(captured) -> tuple[Path, dict]:
    path, body = next(entry for entry in captured if entry[:1] == ("plist",))[1:]
    return Path(path), plistlib.loads(body)


def _off_volume_staging() -> Path | None:
    """A fresh scratch dir on a volume that is NOT the boot volume (and not the Hermes home)."""
    home = Path(os.path.expanduser("~/.hermes")).resolve()
    for entry in sorted(Path("/Volumes").iterdir()):
        if entry.name in ("Macintosh HD", "Time Machine"):
            continue
        try:
            if runtime._path_is_on_boot_volume(entry) or home.is_relative_to(entry.resolve()):
                continue
            return Path(tempfile.mkdtemp(prefix="runtime-job-off-volume-", dir=entry))
        except (OSError, RuntimeError):
            continue
    return None


def test_off_volume_caller_gets_a_boot_volume_plist(tmp_path, monkeypatch):
    external = (tmp_path / "external").resolve()
    (external / "logs").mkdir(parents=True)
    _as_external(monkeypatch, external)
    captured = []
    _capture_bootstrap(monkeypatch, captured)

    job = runtime.start_gateway_job(["true"], env={}, cwd=external,
                                    log_path=external / "logs" / "gateway-stdio.log")

    assert job is not None
    plist_path, plist = _bootstrapped_plist(captured)
    assert plist_path.parent.parent == runtime._BOOT_VOLUME_TEMP_ROOT, plist_path
    assert plist_path.parent.name.startswith("hermes-"), plist_path
    # The caller's own cwd is kept: chdir() to /Volumes works (measured), only launchd's file opens fail.
    assert plist["WorkingDirectory"] == str(external)
    # The launch environment (env vars, the plist itself) must not outlive its job.
    assert not plist_path.exists()
    assert not plist_path.parent.exists(), "the staging dir must be removed again"


def test_unusable_cwd_falls_back_to_the_boot_volume(tmp_path, monkeypatch):
    """An unmounted or unreadable cwd fails the spawn EX_CONFIG 78 with nothing logged."""
    external = (tmp_path / "external").resolve()
    (external / "logs").mkdir(parents=True)
    _as_external(monkeypatch, external)
    captured = []
    _capture_bootstrap(monkeypatch, captured)

    job = runtime.start_gateway_job(["true"], env={}, cwd=external / "gone",
                                    log_path=external / "logs" / "gateway-stdio.log")

    assert job is not None
    _, plist = _bootstrapped_plist(captured)
    assert plist["WorkingDirectory"] == str(runtime._BOOT_VOLUME_TEMP_ROOT)


def test_boot_volume_caller_keeps_its_own_plist_location(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "gateway-stdio.log"
    log_path.parent.mkdir(parents=True)
    captured = []
    _capture_bootstrap(monkeypatch, captured)

    job = runtime.start_gateway_job(["true"], env={}, cwd=tmp_path, log_path=log_path)

    assert job is not None
    plist_path, plist = _bootstrapped_plist(captured)
    assert plist_path.parent == log_path.parent
    assert plist["WorkingDirectory"] == str(tmp_path)


@pytest.mark.platforms("macos")
def test_job_started_from_an_off_volume_caller_really_runs(tmp_path):
    """End to end against real launchd, with cwd and log path genuinely on another volume."""
    staging = _off_volume_staging()
    if staging is None:
        pytest.skip("no writable volume other than the boot volume to place the caller on")
    report = tmp_path / "report.json"
    release = tmp_path / "release"
    child = (
        "import json, os, pathlib, time; "
        f"pathlib.Path({str(report)!r}).write_text(json.dumps({{'cwd': os.getcwd(), 'ppid': os.getppid()}})); "
        f"[time.sleep(0.1) for _ in range(300) if not pathlib.Path({str(release)!r}).exists()]"
    )
    try:
        job = runtime.start_gateway_job([sys.executable, "-c", child], env={**os.environ},
                                        cwd=staging, log_path=staging / "gateway-stdio.log")
        assert job is not None, "launchd refused a job whose caller sits on another volume"
        try:
            wait_until(lambda: report.exists() or job.poll() is not None, timeout=30)
            if not report.exists():
                state = subprocess.run(["launchctl", "print", job.target], capture_output=True, text=True,
                                       encoding="utf-8", check=False).stdout
                pytest.fail(f"the job never ran; launchctl print said:\n{state[:3000]}")
            # The job really runs, from a cwd on another volume, with its plist on the boot volume.
            assert json.loads(report.read_text(encoding="utf-8"))["cwd"] == str(staging.resolve())
            assert job.poll() is None
        finally:
            release.write_text("", encoding="utf-8")
        assert wait_until(lambda: job.poll() is not None, timeout=30)
        assert job.poll() == 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)
