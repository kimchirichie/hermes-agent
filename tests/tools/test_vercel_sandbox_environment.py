"""Unit tests for the Vercel Sandbox terminal backend."""

from __future__ import annotations

import importlib
import io
import os
import re
import sys
import tarfile
import threading
import types
from dataclasses import dataclass
from enum import StrEnum

import pytest


class _FakeRunResult:
    """Mirror of the 0.12 sync SDK's ``CompletedProcess`` (stdout/stderr/returncode)."""

    def __init__(self, stdout: str | bytes = "", *, returncode: int = 0, stderr: str | bytes = ""):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _FakeSandboxStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    ABORTED = "aborted"
    SNAPSHOTTING = "snapshotting"


@dataclass(frozen=True)
class _FakeSnapshot:
    id: str


class _FakeBatch:
    """Mirror of ``SyncSandboxFilesystem.batch()``: stages writes, uploads them on exit."""

    def __init__(self, filesystem: _FakeFilesystem):
        self._filesystem = filesystem
        self._staged: list[dict[str, object]] = []

    def __enter__(self) -> _FakeBatch:
        return self

    def __exit__(self, *_exc) -> bool:
        self._filesystem._record(self._staged)
        return False

    def write_bytes(self, path: str, data: bytes, *, mode: int | None = None) -> None:
        entry: dict[str, object] = {"path": path, "content": data}
        if mode is not None:
            entry["mode"] = mode
        self._staged.append(entry)


class _FakeFilesystem:
    """Mirror of ``SyncSandboxFilesystem`` (``sandbox.fs``)."""

    def __init__(self, sandbox: _FakeSandbox):
        self._sandbox = sandbox

    def _record(self, files: list[dict[str, object]]) -> None:
        self._sandbox.write_files_calls.append(files)
        self._sandbox.events.append(("upload", tuple(str(f.get("path")) for f in files)))
        if self._sandbox.write_files_side_effects:
            effect = self._sandbox.write_files_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                effect(files)

    def write_bytes(self, path: str, data: bytes, *, mode: int | None = None) -> None:
        entry: dict[str, object] = {"path": path, "content": data}
        if mode is not None:
            entry["mode"] = mode
        self._record([entry])

    def batch(self) -> _FakeBatch:
        return _FakeBatch(self)

    def read_bytes(self, path: str) -> bytes:
        self._sandbox.read_bytes_calls.append(path)
        if self._sandbox.read_bytes_side_effects:
            effect = self._sandbox.read_bytes_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(path)
        return self._sandbox.read_bytes_content


class _FakeSandbox:
    def __init__(
        self,
        *,
        cwd: str = "/vercel/sandbox",
        home: str = "/home/vercel",
        status: _FakeSandboxStatus = _FakeSandboxStatus.RUNNING,
        name: str = "sb-123",
    ):
        self.cwd = cwd
        self.name = name
        self.status = status
        self.home = home
        self.fs = _FakeFilesystem(self)
        self.events: list[tuple[str, object]] = []
        self.run_process_calls: list[tuple[str, list[str], dict]] = []
        self.run_process_side_effects: list[object] = []
        self.write_files_calls: list[list[dict[str, object]]] = []
        self.write_files_side_effects: list[object] = []
        self.read_bytes_calls: list[str] = []
        self.read_bytes_side_effects: list[object] = []
        self.read_bytes_content = b""
        self.stop_calls: list[tuple[tuple, dict]] = []
        self.snapshot_calls: list[tuple[tuple, dict]] = []
        self.snapshot_side_effects: list[object] = []
        self.snapshot_id = "snap_default"

    def run_process(self, cmd: str, args: list[str] | None = None, **kwargs):
        args = list(args or [])
        self.run_process_calls.append((cmd, args, kwargs))
        self.events.append(("run", args[1] if len(args) > 1 else ""))
        if self.run_process_side_effects:
            effect = self.run_process_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(cmd, args, kwargs)
            return effect
        script = args[1] if len(args) > 1 else ""
        if 'printf %s "$HOME"' in script:
            return _FakeRunResult(self.home)
        return _FakeRunResult("")

    def stop(self, *args, **kwargs) -> None:
        self.stop_calls.append((args, kwargs))

    def snapshot(self, *args, **kwargs):
        self.snapshot_calls.append((args, kwargs))
        if self.snapshot_side_effects:
            effect = self.snapshot_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(*args, **kwargs)
            if isinstance(effect, str):
                return _FakeSnapshot(effect)
            return effect
        return _FakeSnapshot(self.snapshot_id)


@dataclass(frozen=True)
class _FakeResources:
    vcpus: float | None = None
    memory: int | None = None


@dataclass(frozen=True)
class _FakeSnapshotSource:
    snapshot_id: str
    type: str = "snapshot"


class _FakeSDK:
    def __init__(self):
        self.create_kwargs: list[dict[str, object]] = []
        self.create_side_effects: list[object] = []
        self.sandboxes: list[_FakeSandbox] = []
        self.get_sandbox_calls: list[str] = []
        self.get_sandbox_side_effects: list[object] = []

    @property
    def current(self) -> _FakeSandbox:
        return self.sandboxes[-1]

    def create_sandbox(self, **kwargs):
        self.create_kwargs.append(kwargs)
        if self.create_side_effects:
            effect = self.create_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if isinstance(effect, _FakeSandbox):
                self.sandboxes.append(effect)
                return effect
        sandbox = _FakeSandbox()
        self.sandboxes.append(sandbox)
        return sandbox

    def get_sandbox(self, *, name: str):
        """Stand-in for ``sync.get_sandbox``: the 0.12 refresh-by-name read."""
        self.get_sandbox_calls.append(name)
        if self.get_sandbox_side_effects:
            effect = self.get_sandbox_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if isinstance(effect, _FakeSandbox):
                return effect
        return self.sandboxes[-1]


def _cwd_result(body: str = "", *, cwd: str = "/vercel/sandbox", returncode: int = 0):
    def _result(_cmd: str, args: list[str], _kwargs: dict):
        script = args[1] if len(args) > 1 else ""
        match = re.search(r"__HERMES_CWD_[A-Za-z0-9]+__", script)
        marker = match.group(0) if match else "__HERMES_CWD_MISSING__"
        prefix = f"{body}\n\n" if body else "\n"
        return _FakeRunResult(f"{prefix}{marker}{cwd}{marker}\n", returncode=returncode)

    return _result


def _tar_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture()
def vercel_sdk(monkeypatch):
    fake_sdk = _FakeSDK()
    sync_mod = types.ModuleType("vercel.sandbox.sync")
    sync_mod.SyncSandbox = object
    sync_mod.SyncSnapshot = object
    sync_mod.SandboxResources = _FakeResources
    sync_mod.SandboxStatus = _FakeSandboxStatus
    sync_mod.SnapshotSource = _FakeSnapshotSource
    sync_mod.create_sandbox = fake_sdk.create_sandbox
    sync_mod.get_sandbox = fake_sdk.get_sandbox

    sandbox_mod = types.ModuleType("vercel.sandbox")
    sandbox_mod.sync = sync_mod

    vercel_mod = types.ModuleType("vercel")
    vercel_mod.sandbox = sandbox_mod

    monkeypatch.setitem(sys.modules, "vercel", vercel_mod)
    monkeypatch.setitem(sys.modules, "vercel.sandbox", sandbox_mod)
    monkeypatch.setitem(sys.modules, "vercel.sandbox.sync", sync_mod)
    return fake_sdk


@pytest.fixture()
def vercel_module(vercel_sdk, monkeypatch):
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", list)
    monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kwargs: [])
    monkeypatch.setattr("tools.credential_files.iter_cache_files", lambda **kwargs: [])

    module = importlib.import_module("tools.environments.vercel_sandbox")
    module = importlib.reload(module)
    # These tests fully fake the vercel SDK via sys.modules (vercel_sdk
    # fixture) — the real package is never exercised, so the lazy-install
    # probe must not run: _ensure_vercel_sdk checks installed DISTRIBUTION
    # metadata (not sys.modules), and on a host without the vercel dist +
    # with lazy installs disabled (any hermetic env) it raises ImportError
    # before the fake SDK is ever reached. 16 tests failed exactly this way
    # in CI while passing on dev boxes that happened to have vercel==0.7.2
    # installed.
    monkeypatch.setattr(module, "_ensure_vercel_sdk", lambda: None)
    return module


@pytest.fixture()
def make_env(vercel_module, request):
    envs = []

    def _cleanup_envs():
        for env in envs:
            env._sync_manager = None
            env.cleanup()

    request.addfinalizer(_cleanup_envs)

    def _factory(**kwargs):
        kwargs.setdefault("runtime", "node22")
        kwargs.setdefault("cwd", vercel_module.DEFAULT_VERCEL_CWD)
        kwargs.setdefault("timeout", 30)
        kwargs.setdefault("task_id", "task-123")
        env = vercel_module.VercelSandboxEnvironment(**kwargs)
        envs.append(env)
        return env

    return _factory


class TestStartup:
    def test_default_cwd_tracks_remote_workspace_root(self, make_env, vercel_sdk):
        sandbox = _FakeSandbox(cwd="/workspace")
        vercel_sdk.create_side_effects.append(sandbox)

        env = make_env()

        assert env.cwd == "/workspace"

    def test_tilde_cwd_resolves_against_remote_home(self, make_env, vercel_sdk):
        sandbox = _FakeSandbox(home="/home/custom")
        vercel_sdk.create_side_effects.append(sandbox)

        env = make_env(cwd="~")

        assert env.cwd == "/home/custom"

    def test_pending_sandbox_raises_descriptive_error(
        self, make_env, vercel_sdk
    ):
        sandbox = _FakeSandbox(status=_FakeSandboxStatus.PENDING)
        vercel_sdk.create_side_effects.append(sandbox)

        with pytest.raises(RuntimeError, match="Sandbox did not reach running state"):
            make_env()


class TestFileSync:
    def test_initial_sync_uploads_managed_files_under_remote_home(
        self, make_env, vercel_sdk, monkeypatch, tmp_path
    ):
        src = tmp_path / "token.txt"
        src.write_text("secret-token")
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [
                {
                    "host_path": str(src),
                    "container_path": "/root/.hermes/credentials/token.txt",
                }
            ],
        )
        monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kwargs: [])
        monkeypatch.setattr("tools.credential_files.iter_cache_files", lambda **kwargs: [])

        make_env()

        uploaded = vercel_sdk.current.write_files_calls[0]
        assert uploaded == [
            {
                "path": "/home/vercel/.hermes/credentials/token.txt",
                "content": b"secret-token",
            }
        ]

    def test_execute_resyncs_changed_managed_files(
        self, make_env, vercel_sdk, monkeypatch, tmp_path
    ):
        src = tmp_path / "token.txt"
        src.write_text("secret-token")
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [
                {
                    "host_path": str(src),
                    "container_path": "/root/.hermes/credentials/token.txt",
                }
            ],
        )
        monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kwargs: [])
        monkeypatch.setattr("tools.credential_files.iter_cache_files", lambda **kwargs: [])

        env = make_env()
        src.write_text("updated-secret-token")
        monkeypatch.setenv("HERMES_FORCE_FILE_SYNC", "1")
        vercel_sdk.current.run_process_side_effects.append(_cwd_result("hello"))

        result = env.execute("echo hello")

        assert result["output"] == "hello\n"
        assert result["returncode"] == 0
        assert vercel_sdk.current.write_files_calls[-1] == [
            {
                "path": "/home/vercel/.hermes/credentials/token.txt",
                "content": b"updated-secret-token",
            }
        ]

    def test_cleanup_syncs_back_snapshots_closes_and_is_idempotent(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        src = tmp_path / "token.txt"
        src.write_text("host-token")
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [
                {
                    "host_path": str(src),
                    "container_path": "/root/.hermes/credentials/token.txt",
                }
            ],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_skills_files",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_cache_files",
            lambda **kwargs: [],
        )
        env = make_env()
        sandbox = vercel_sdk.current
        sandbox.snapshot_id = "snap_cleanup"
        vercel_sdk.current.read_bytes_content = _tar_bytes(
            {
                "home/vercel/.hermes/credentials/token.txt": b"remote-token",
                "home/vercel/.hermes/credentials/new.txt": b"new-remote",
                "home/vercel/.hermes/unmapped/skip.txt": b"skip",
            }
        )

        env.cleanup()
        env.cleanup()

        # The remote tar must skip live sockets (gateway.sock) instead of failing the download.
        tar_scripts = [args[1] for cmd, args, _ in sandbox.run_process_calls
                       if cmd == "bash" and args and args[1].startswith("tar cf ")]
        assert tar_scripts and all("--exclude='*.sock'" in script for script in tar_scripts)

        # Credential mounts are upload-only since bcfc7458fa ("fix remote
        # sync-back credential overwrite"): the sandbox must never rewrite a
        # host credential file, so token.txt keeps its host content, and the
        # credential mapping cannot serve as an inference anchor for new
        # remote files either — new.txt/skip.txt stay remote-only.
        assert src.read_text() == "host-token"
        assert not (tmp_path / "new.txt").exists()
        assert not (tmp_path / "skip.txt").exists()
        assert len(sandbox.snapshot_calls) == 1
        assert len(sandbox.stop_calls) == 1  # always stop after snapshot to avoid resource leaks
        assert vercel_module._load_snapshots() == {"task-123": "snap_cleanup"}

    def test_cleanup_sync_back_failure_from_download_does_not_block_snapshot(
        self, make_env, vercel_sdk, monkeypatch, tmp_path
    ):
        src = tmp_path / "token.txt"
        src.write_text("host-token")
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [
                {
                    "host_path": str(src),
                    "container_path": "/root/.hermes/credentials/token.txt",
                }
            ],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_skills_files",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_cache_files",
            lambda **kwargs: [],
        )
        env = make_env()
        sandbox = vercel_sdk.current
        sandbox.run_process_side_effects.extend(
            [
                _FakeRunResult("tar failed", returncode=2),
                _FakeRunResult(""),
                _FakeRunResult("tar failed", returncode=2),
                _FakeRunResult(""),
                _FakeRunResult("tar failed", returncode=2),
                _FakeRunResult(""),
            ]
        )
        monkeypatch.setattr("tools.environments.file_sync.time.sleep", lambda _delay: None)

        env.cleanup()

        assert src.read_text() == "host-token"
        assert len(sandbox.snapshot_calls) == 1
        assert len(sandbox.read_bytes_calls) == 0


class TestExecute:

    @pytest.mark.parametrize("unhealthy", ["terminal-state", "refresh-failure"])
    def test_execute_recreates_unhealthy_sandbox_before_running_command(
        self, make_env, vercel_sdk, unhealthy
    ):
        env = make_env()
        original = vercel_sdk.current
        if unhealthy == "terminal-state":
            original.status = _FakeSandboxStatus.STOPPED
        else:
            # 0.12 has no handle.refresh(); the backend re-reads the sandbox by name.
            vercel_sdk.get_sandbox_side_effects.append(RuntimeError("refresh failed"))

        replacement = _FakeSandbox()
        replacement.run_process_side_effects.extend(
            [
                _FakeRunResult(replacement.home),
                _cwd_result("hello"),
            ]
        )
        vercel_sdk.create_side_effects.append(replacement)

        result = env.execute("echo hello")

        assert result["output"] == "hello\n", unhealthy
        assert result["returncode"] == 0, unhealthy
        assert vercel_sdk.get_sandbox_calls == [original.name]
        assert vercel_sdk.current is replacement

    def test_run_bash_handle_uses_captured_sandbox_for_exec_and_cancel(
        self, make_env
    ):
        env = make_env()
        original = env._sandbox
        assert original is not None
        replacement = _FakeSandbox()
        started = threading.Event()
        release = threading.Event()

        def blocking_command(_cmd: str, _args: list[str], _kwargs: dict):
            started.set()
            release.wait(timeout=5)
            return _FakeRunResult("done")

        original.run_process_side_effects.append(blocking_command)

        handle = env._run_bash("echo done")
        assert started.wait(timeout=1)

        env._sandbox = replacement
        handle.kill()
        release.set()

        assert handle.wait(timeout=2) == 0
        assert len(original.stop_calls) == 1
        assert replacement.stop_calls == []
        cmd, args, kwargs = original.run_process_calls[-1]
        assert cmd == "bash"
        assert args == ["-c", "echo done"]
        assert kwargs["cwd"] == "/vercel/sandbox"
        assert kwargs["capture_output"] is True


    def test_payload_stdin_is_staged_byte_exact_not_in_argv(self, make_env):
        env = make_env()
        sandbox = env._sandbox
        raw = (b"A" * (160 * 1024)) + b"\x00\xff\xfeTAIL"
        writes_before = len(sandbox.write_files_calls)

        handle = env._run_bash("cat > /tmp/out", stdin_data=raw.decode("utf-8", "surrogateescape"))

        assert handle.wait(timeout=2) == 0
        (staged,) = sandbox.write_files_calls[writes_before]
        assert staged["content"] == raw
        assert staged["mode"] == 0o600
        cmd, args, _ = sandbox.run_process_calls[-1]
        assert cmd == "bash" and args[0] == "-c"
        assert args[1].startswith(f"exec 0< {staged['path']} || exit $?\n")
        assert args[1].endswith("\ncat > /tmp/out")
        assert "AAAAAAAAAAAAAAAA" not in " ".join(args)


class TestSnapshotPersistence:
    def test_create_restores_from_saved_snapshot(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        vercel_module._store_snapshot("task-123", "snap_saved")
        restored = _FakeSandbox(cwd="/restored")
        vercel_sdk.create_side_effects.append(restored)

        env = make_env()

        assert env.cwd == "/restored"
        assert vercel_sdk.create_kwargs[0]["source"].snapshot_id == "snap_saved"
        assert vercel_sdk.create_kwargs[0]["source"].type == "snapshot"
        assert vercel_module._load_snapshots() == {"task-123": "snap_saved"}

    def test_restore_failure_prunes_snapshot_and_falls_back_to_fresh_sandbox(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        vercel_module._store_snapshot("task-123", "snap_stale")
        fresh = _FakeSandbox(cwd="/fresh")
        vercel_sdk.create_side_effects.extend(
            [RuntimeError("snapshot missing"), fresh]
        )

        env = make_env()

        assert env.cwd == "/fresh"
        assert vercel_sdk.create_kwargs[0]["source"].snapshot_id == "snap_stale"
        assert "source" not in vercel_sdk.create_kwargs[1]
        assert vercel_module._load_snapshots() == {}

    def test_cleanup_stops_when_snapshot_fails_without_storing_metadata(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        env = make_env()
        sandbox = vercel_sdk.current
        sandbox.snapshot_side_effects.append(RuntimeError("snapshot failed"))

        env.cleanup()

        assert len(sandbox.snapshot_calls) == 1
        assert len(sandbox.stop_calls) == 1
        assert vercel_module._load_snapshots() == {}

    def test_non_persistent_cleanup_stops_without_snapshot(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        env = make_env(persistent_filesystem=False)
        sandbox = vercel_sdk.current

        env.cleanup()

        assert sandbox.snapshot_calls == []
        assert len(sandbox.stop_calls) == 1
        assert vercel_module._load_snapshots() == {}

    def test_persistent_cleanup_without_task_id_stops_without_snapshot(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        env = make_env(task_id="")
        sandbox = vercel_sdk.current

        env.cleanup()

        assert sandbox.snapshot_calls == []
        assert len(sandbox.stop_calls) == 1
        assert vercel_module._load_snapshots() == {}


class TestCleanup:
    def test_cleanup_continues_when_sync_back_raises(self, make_env, vercel_sdk):
        env = make_env()
        sandbox = vercel_sdk.current

        class FailingSyncManager:
            def sync_back(self):
                raise RuntimeError("download failed")

        env._sync_manager = FailingSyncManager()

        env.cleanup()

        assert len(sandbox.snapshot_calls) == 1
        assert len(sandbox.stop_calls) == 1


class TestImageSelection:
    def test_image_always_sent_and_legacy_runtime_ignored(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        """vercel==0.12.0 dropped the ``runtime`` kwarg (runtimes were deprecated by Vercel), so a
        legacy ``terminal.vercel_runtime`` pin is warned about and dropped: the image is always sent,
        and a snapshot restore sends no image because the snapshot carries its own base."""
        from datetime import timedelta

        from hermes_cli.config_defaults import DEFAULT_VERCEL_IMAGE

        make_env(runtime=None)
        make_env(runtime=None, image="vercel/sandbox/python:3.14")
        make_env(runtime="node22", image="vercel/sandbox/python:3.14")
        fresh, custom, pinned = vercel_sdk.create_kwargs[-3:]
        assert fresh["image"] == DEFAULT_VERCEL_IMAGE
        assert custom["image"] == "vercel/sandbox/python:3.14"
        assert pinned["image"] == "vercel/sandbox/python:3.14"
        assert all("runtime" not in kwargs for kwargs in (fresh, custom, pinned))
        # ``timeout`` -> ``execution_time_limit`` (floored at Vercel's 5-minute minimum).
        assert fresh["execution_time_limit"] == timedelta(minutes=5)
        assert fresh["resources"] == _FakeResources(vcpus=1, memory=5120)

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        vercel_module._store_snapshot("task-123", "snap_saved")
        make_env(runtime="node22")
        restore = vercel_sdk.create_kwargs[-1]
        assert restore["source"].snapshot_id == "snap_saved"
        assert "runtime" not in restore and "image" not in restore


class TestSetupScript:
    """``terminal.vercel_setup_script``: a loud, ordered pre-launch hook, run once per fresh box."""

    @staticmethod
    def _write_script(tmp_path, body: str = "#!/bin/bash\necho ready\n"):
        script = tmp_path / "setup.sh"
        script.write_text(body)
        return script

    @staticmethod
    def _setup_runs(sandbox, remote: str) -> list[int]:
        return [i for i, (kind, script) in enumerate(sandbox.events)
                if kind == "run" and remote in str(script)]

    @staticmethod
    def _init_session_runs(sandbox) -> list[int]:
        return [i for i, (kind, script) in enumerate(sandbox.events)
                if kind == "run" and "__HERMES_CWD_" in str(script)]

    @staticmethod
    def _setup_uploads(sandbox, remote: str) -> list[dict[str, object]]:
        return [f for files in sandbox.write_files_calls for f in files if f["path"] == remote]

    def test_unset_setup_script_makes_no_extra_call(self, make_env, vercel_module, vercel_sdk):
        env = make_env()
        sandbox = vercel_sdk.current
        remote = vercel_module._SETUP_SCRIPT_REMOTE

        assert env._setup_script is None
        assert self._setup_uploads(sandbox, remote) == []
        assert self._setup_runs(sandbox, remote) == []
        # Unchanged behaviour: home detection, then init_session, and nothing else.
        runs = [script for kind, script in sandbox.events if kind == "run"]
        assert len(runs) == 2
        assert "__HERMES_CWD_" in runs[-1]

    def test_setup_script_uploaded_and_run_before_init_session(
        self, make_env, vercel_module, vercel_sdk, tmp_path
    ):
        script = self._write_script(tmp_path, "#!/bin/bash\ngit clone https://example.test/repo\n")
        make_env(setup_script=str(script))
        sandbox = vercel_sdk.current
        remote = vercel_module._SETUP_SCRIPT_REMOTE

        uploads = self._setup_uploads(sandbox, remote)
        assert len(uploads) == 1
        assert uploads[0]["content"] == script.read_bytes()

        upload_at = sandbox.events.index(("upload", (remote,)))
        setup_runs, init_runs = self._setup_runs(sandbox, remote), self._init_session_runs(sandbox)
        assert len(setup_runs) == 1 and len(init_runs) == 1
        assert upload_at < setup_runs[0] < init_runs[0]

    def test_failing_setup_raises_and_leaves_no_half_set_up_box(
        self, make_env, vercel_module, vercel_sdk, tmp_path
    ):
        script = self._write_script(tmp_path)
        sandbox = _FakeSandbox()
        sandbox.run_process_side_effects.extend(
            [_FakeRunResult(sandbox.home), _FakeRunResult("fatal: could not clone", returncode=128)]
        )
        vercel_sdk.create_side_effects.append(sandbox)

        with pytest.raises(RuntimeError, match="Vercel setup failed: fatal: could not clone"):
            make_env(setup_script=str(script))

        assert len(sandbox.stop_calls) == 1, "a failed setup must not leave the sandbox running"
        assert self._init_session_runs(sandbox) == [], "the box was never marked ready"

    def test_recreate_runs_setup_again_on_the_new_box(
        self, make_env, vercel_module, vercel_sdk, tmp_path
    ):
        script = self._write_script(tmp_path)
        env = make_env(setup_script=str(script))
        remote = vercel_module._SETUP_SCRIPT_REMOTE
        first = vercel_sdk.current
        assert len(self._setup_runs(first, remote)) == 1

        first.status = _FakeSandboxStatus.STOPPED
        replacement = _FakeSandbox()
        replacement.run_process_side_effects.extend(
            [_FakeRunResult(replacement.home), _FakeRunResult(""), _cwd_result("hello")]
        )
        vercel_sdk.create_side_effects.append(replacement)

        result = env.execute("echo hello")

        assert result["output"] == "hello\n"
        assert len(self._setup_runs(replacement, remote)) == 1
        assert len(self._setup_uploads(replacement, remote)) == 1

    def test_setup_timeout_reaches_the_sdk_kill_after(
        self, make_env, vercel_module, vercel_sdk, tmp_path
    ):
        script = self._write_script(tmp_path)
        make_env(setup_script=str(script), setup_timeout=42)
        remote = vercel_module._SETUP_SCRIPT_REMOTE

        setup_calls = [call for call in vercel_sdk.current.run_process_calls
                       if len(call[1]) > 1 and remote in call[1][1]]
        assert len(setup_calls) == 1
        cmd, args, kwargs = setup_calls[0]
        assert cmd == "bash" and args[0] == "-lc"
        assert kwargs["cwd"] == "/vercel/sandbox"
        assert kwargs["kill_after"] == 42

    def test_failure_output_is_tail_capped_and_carries_no_env_values(
        self, make_env, vercel_module, vercel_sdk, tmp_path, monkeypatch, caplog
    ):
        secret = "ghp_SUPERSECRETVALUE"
        token = tmp_path / "gh-token.env"
        token.write_text(f"GH_TOKEN={secret}\n")
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [{"host_path": str(token),
                      "container_path": "/root/.hermes/credentials/gh-token.env"}],
        )
        monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kwargs: [])
        monkeypatch.setattr("tools.credential_files.iter_cache_files", lambda **kwargs: [])
        script = self._write_script(
            tmp_path, '#!/bin/bash\nsource "$HOME/.hermes/credentials/gh-token.env"\ngit clone repo\n')
        remote = vercel_module._SETUP_SCRIPT_REMOTE
        noisy = ("x" * 20000) + "\nfatal: could not read Username for 'https://example.test'"
        sandbox = _FakeSandbox()
        sandbox.run_process_side_effects.extend(
            [_FakeRunResult(sandbox.home), _FakeRunResult(noisy, returncode=1)]
        )
        vercel_sdk.create_side_effects.append(sandbox)

        with pytest.raises(RuntimeError) as excinfo:
            make_env(setup_script=str(script))

        message = str(excinfo.value)
        assert message.startswith("Vercel setup failed: ")
        assert message.rstrip().endswith("fatal: could not read Username for 'https://example.test'")
        assert "…(truncated)…" in message
        assert len(message) < vercel_module._SETUP_OUTPUT_TAIL + 200
        # The script reads the credential from a mounted file; Hermes never passes or echoes it.
        assert secret not in message
        assert secret not in caplog.text
        assert all(secret not in " ".join(args) for _cmd, args, _kw in sandbox.run_process_calls)
        assert all(secret.encode() not in f["content"] for f in self._setup_uploads(sandbox, remote))


class TestSetupScriptConfigBridge:
    def test_config_bridge_parses_setup_keys_through_get_env_config(self, monkeypatch):
        """config.yaml -> env bridge -> _get_env_config(), the real path terminal_tool reads."""
        from hermes_constants import get_hermes_home
        from tools import terminal_tool

        home = get_hermes_home()
        home.mkdir(parents=True, exist_ok=True)
        config_path = home / "config.yaml"
        config_path.write_text(
            "terminal:\n"
            "  backend: vercel_sandbox\n"
            "  vercel_setup_script: scripts/clone.sh\n"
            "  vercel_setup_timeout: 900\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)
        for name in ("TERMINAL_ENV", "TERMINAL_CWD",
                     "TERMINAL_VERCEL_SETUP_SCRIPT", "TERMINAL_VERCEL_SETUP_TIMEOUT"):
            monkeypatch.delenv(name, raising=False)
        try:
            config = terminal_tool._get_env_config()
        finally:
            config_path.unlink(missing_ok=True)

        assert config["env_type"] == "vercel_sandbox"
        assert config["vercel_setup_script"] == "scripts/clone.sh"
        assert config["vercel_setup_timeout"] == 900
        assert os.environ["TERMINAL_VERCEL_SETUP_SCRIPT"] == "scripts/clone.sh"
        assert os.environ["TERMINAL_VERCEL_SETUP_TIMEOUT"] == "900"
