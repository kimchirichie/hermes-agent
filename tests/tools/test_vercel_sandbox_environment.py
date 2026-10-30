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


class _FakeCredentials:
    def __init__(self, token, team_id, project_id):
        self.token, self.team_id, self.project_id = token, team_id, project_id


class _FakeServiceOptions:
    def __init__(self, credentials_factory=None, **_kw):
        self.credentials_factory = credentials_factory


class _FakeClient:
    """Stand-in for ``SyncSandboxClient``: the credentials arrive through the factory, never os.environ."""
    instances: list = []

    def __init__(self, sdk, options):
        self._sdk, self.options, self.closed = sdk, options, False
        _FakeClient.instances.append(self)

    @classmethod
    def factory(cls, sdk):
        def create(*, options=None, **_kw):
            return cls(sdk, options)
        return create

    def resolved_credentials(self):
        return self.options.credentials_factory()

    def create_sandbox(self, **kwargs):
        self.resolved_credentials()
        return self._sdk.create_sandbox(**kwargs)

    def get_sandbox(self, *, name):
        self.resolved_credentials()
        return self._sdk.get_sandbox(name=name)

    def close(self):
        self.closed = True


@pytest.fixture()
def vercel_sdk(monkeypatch):
    fake_sdk = _FakeSDK()
    _FakeClient.instances = []
    for name in ("VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("agent.secret_scope.get_secret",
                        lambda name, default=None: {"VERCEL_TOKEN": "tok-A", "VERCEL_PROJECT_ID": "prj-A",
                                                    "VERCEL_TEAM_ID": "team-A"}.get(name, default))
    sync_mod = types.ModuleType("vercel.sandbox.sync")
    sync_mod.SyncSandbox = object
    sync_mod.SyncSnapshot = object
    sync_mod.SandboxResources = _FakeResources
    sync_mod.SandboxStatus = _FakeSandboxStatus
    sync_mod.SnapshotSource = _FakeSnapshotSource
    sync_mod.SandboxCredentials = _FakeCredentials
    sync_mod.SandboxServiceOptions = _FakeServiceOptions
    sync_mod.SyncSandboxClient = types.SimpleNamespace(create=_FakeClient.factory(fake_sdk))

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
        # The shared defaults (cpu 1, memory 5120) are ILLEGAL at Vercel (memory must be
        # vcpus * 2048), so the backend derives the memory the API accepts instead of failing
        # every create with HTTP 400.
        assert fresh["resources"] == _FakeResources(vcpus=1, memory=2048)

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


class TestToolDefaultCwd:
    """The terminal tool's default cwd for ``vercel_sandbox`` must follow the sandbox's own directory.

    The stock image's working directory and HOME are ``/vercel`` and ``/vercel/sandbox`` does not exist,
    so a literal default wrapped every command that omitted ``workdir`` in ``cd /vercel/sandbox`` and it
    died with exit 126. ``~`` is expanded by the backend against the sandbox's detected HOME, so it is
    right on this image and on any future one whose home differs.
    """

    @staticmethod
    def _tool_default() -> str:
        from tools import terminal_tool

        return terminal_tool._backend_default_cwd("vercel_sandbox")

    def test_default_is_the_sandbox_home_not_a_literal(self, monkeypatch):
        from tools import terminal_tool

        monkeypatch.delenv("TERMINAL_CWD", raising=False)

        assert self._tool_default() == "~"
        assert terminal_tool._resolve_config_cwd("vercel_sandbox", False) == ("~", None)

    def test_shipped_config_default_falls_back_to_the_sandbox_home(self, monkeypatch):
        """``DEFAULT_CONFIG['terminal']['cwd']`` is ``"."``: a relative host path no sandbox can cd
        into, so it must resolve to the backend default — not to the dead ``/vercel/sandbox`` literal."""
        from tools import terminal_tool

        monkeypatch.setenv("TERMINAL_CWD", ".")

        assert terminal_tool._resolve_config_cwd("vercel_sandbox", False) == ("~", None)

    def test_explicit_cwd_is_honoured(self, monkeypatch):
        from tools import terminal_tool

        monkeypatch.setenv("TERMINAL_CWD", "/tmp")

        assert terminal_tool._resolve_config_cwd("vercel_sandbox", False) == ("/tmp", None)

    def test_legacy_literal_still_reaches_the_real_workspace_root(self, make_env, vercel_sdk, monkeypatch):
        """A profile that explicitly set ``terminal.cwd: /vercel/sandbox`` keeps working: the backend
        maps the legacy literal to the sandbox's real workspace root instead of the missing path."""
        from tools import terminal_tool

        monkeypatch.setenv("TERMINAL_CWD", "/vercel/sandbox")
        assert terminal_tool._resolve_config_cwd("vercel_sandbox", False) == ("/vercel/sandbox", None)

        sandbox = _FakeSandbox(cwd="/vercel", home="/vercel")
        vercel_sdk.create_side_effects.append(sandbox)

        env = make_env(cwd="/vercel/sandbox")

        assert env.cwd == "/vercel"

    def test_default_cwd_resolves_to_the_real_sandbox_dir(self, make_env, vercel_sdk):
        sandbox = _FakeSandbox(cwd="/vercel", home="/vercel")
        vercel_sdk.create_side_effects.append(sandbox)

        env = make_env(cwd=self._tool_default())

        assert env.cwd == "/vercel"

    def test_default_cwd_still_works_on_an_image_whose_cwd_is_the_old_literal(self, make_env, vercel_sdk):
        sandbox = _FakeSandbox(cwd="/vercel/sandbox", home="/vercel/sandbox")
        vercel_sdk.create_side_effects.append(sandbox)

        env = make_env(cwd=self._tool_default())

        assert env.cwd == "/vercel/sandbox"

    def test_default_cwd_wraps_the_command_with_tilde(self, make_env, vercel_sdk, monkeypatch):
        """The per-command ``cd`` must be ``~`` (expanded in the sandbox), never the dead literal."""
        from tools import terminal_tool

        sandbox = _FakeSandbox(cwd="/vercel", home="/vercel")
        vercel_sdk.create_side_effects.append(sandbox)
        env = make_env(cwd=self._tool_default())
        monkeypatch.setattr(terminal_tool, "get_session_cwd", lambda _key: None)
        command_cwd = terminal_tool._resolve_command_cwd(
            workdir=None, default_cwd=self._tool_default(), session_key="tool-default",
            env_type="vercel_sandbox", env=env)
        sandbox.run_process_side_effects.append(_cwd_result(cwd="/vercel"))

        env.execute("pwd", cwd=command_cwd)

        assert command_cwd == "~"
        scripts = [args[1] for _cmd, args, _kw in sandbox.run_process_calls if len(args) > 1]
        cd_lines = [line for script in scripts for line in script.splitlines()
                    if line.startswith("builtin cd --")]
        assert cd_lines, scripts
        assert cd_lines[-1] == "builtin cd -- ~ || exit 126", cd_lines
        assert "/vercel/sandbox" not in cd_lines[-1]

    def test_create_environment_through_the_terminal_tool_path(self, vercel_module, vercel_sdk, monkeypatch):
        """The exact path the terminal tool takes: resolve the config cwd, then build the backend."""
        from tools import terminal_tool
        from tools.terminal_tool_backends import _create_environment

        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        cwd, host_cwd = terminal_tool._resolve_config_cwd("vercel_sandbox", False)
        sandbox = _FakeSandbox(cwd="/vercel", home="/vercel")
        vercel_sdk.create_side_effects.append(sandbox)

        env = _create_environment("vercel_sandbox", image="", cwd=cwd, timeout=30, task_id="tool-default")
        try:
            assert host_cwd is None
            assert env.cwd == "/vercel"
        finally:
            env.cleanup()


class TestVercelProfileScopedCredentials:
    """The vercel SDK reads credentials only from os.environ. A multiplexing gateway keeps a
    profile's .env in a context-local secret scope instead, so the backend must hand the SDK the
    scoped values explicitly and must never write them into os.environ."""

    def _env(self, vercel_module):
        return vercel_module.VercelSandboxEnvironment(
            cwd="/vercel/sandbox", timeout=60, persistent_filesystem=False, task_id="t")

    def test_credentials_come_from_the_scope_when_environ_has_none(self, vercel_module):
        env = self._env(vercel_module)
        creds = _FakeClient.instances[-1].resolved_credentials()
        assert (creds.token, creds.team_id, creds.project_id) == ("tok-A", "team-A", "prj-A")
        env.cleanup()

    def test_os_environ_is_never_written(self, vercel_module):
        env = self._env(vercel_module)
        for name in ("VERCEL_TOKEN", "VERCEL_PROJECT_ID", "VERCEL_TEAM_ID"):
            assert name not in os.environ
        env.cleanup()

    def test_missing_credential_names_the_variable_and_hides_values(self, vercel_module, monkeypatch):
        monkeypatch.setattr("agent.secret_scope.get_secret",
                            lambda name, default=None: None if name == "VERCEL_TEAM_ID" else "secret-value")
        with pytest.raises(RuntimeError) as exc:
            self._env(vercel_module)
        assert "VERCEL_TEAM_ID" in str(exc.value)
        assert "secret-value" not in str(exc.value)

    def test_two_profiles_on_two_threads_never_see_each_others_token(self, vercel_module, monkeypatch):
        import threading
        scopes = {"A": {"VERCEL_TOKEN": "tok-A", "VERCEL_PROJECT_ID": "prj-A", "VERCEL_TEAM_ID": "team-A"},
                  "B": {"VERCEL_TOKEN": "tok-B", "VERCEL_PROJECT_ID": "prj-B", "VERCEL_TEAM_ID": "team-B"}}
        local = threading.local()
        monkeypatch.setattr("agent.secret_scope.get_secret",
                            lambda name, default=None: scopes[local.profile].get(name, default))
        seen, gate = {}, threading.Barrier(2)

        def run(profile):
            local.profile = profile
            gate.wait()
            env = self._env(vercel_module)
            seen[profile] = _FakeClient.instances[-1] if False else env._vercel_client().resolved_credentials()
            env.cleanup()

        threads = [threading.Thread(target=run, args=(p,)) for p in ("A", "B")]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert (seen["A"].token, seen["A"].project_id) == ("tok-A", "prj-A")
        assert (seen["B"].token, seen["B"].project_id) == ("tok-B", "prj-B")

    def test_refresh_and_cleanup_use_the_same_scoped_client(self, vercel_module, vercel_sdk):
        env = self._env(vercel_module)
        client = env._vercel_client()
        env._ensure_sandbox_ready()
        assert env._vercel_client() is client and vercel_sdk.get_sandbox_calls
        env.cleanup()
        assert client.closed is True


class TestSnapshotExpiration:
    """``terminal.vercel_snapshot_expiration``: Vercel snapshots a stopping sandbox and keeps it
    for 30 days unless creation asked for less. One project-wide snapshot quota, so abandoned
    ephemeral sandboxes exhausted it (HTTP 402 on 2026-10-29). An ephemeral sandbox gets the
    configured lifetime; a persistent one keeps the platform default so its own restore works."""

    def test_ephemeral_create_sends_the_configured_expiration(self, make_env, vercel_sdk):
        from datetime import timedelta

        make_env(persistent_filesystem=False, snapshot_expiration="1d")

        assert vercel_sdk.create_kwargs[-1]["snapshot_expiration"] == timedelta(days=1)

    def test_configured_duration_units_and_float_seconds(self, make_env, vercel_sdk):
        from datetime import timedelta

        make_env(persistent_filesystem=False, snapshot_expiration="36h")
        make_env(persistent_filesystem=False, snapshot_expiration="2w")
        make_env(persistent_filesystem=False, snapshot_expiration=86400)

        assert vercel_sdk.create_kwargs[-3]["snapshot_expiration"] == timedelta(hours=36)
        assert vercel_sdk.create_kwargs[-2]["snapshot_expiration"] == timedelta(days=14)
        assert vercel_sdk.create_kwargs[-1]["snapshot_expiration"] == timedelta(days=1)

    def test_zero_means_never_expire(self, make_env, vercel_sdk):
        from datetime import timedelta

        make_env(persistent_filesystem=False, snapshot_expiration="0")

        assert vercel_sdk.create_kwargs[-1]["snapshot_expiration"] == timedelta(0)

    def test_persistent_create_keeps_todays_behaviour(self, make_env, vercel_sdk):
        """A persistent task restores its own snapshot later, so nothing about its lifetime
        changes: the kwargs carry no expiration at all."""
        make_env(persistent_filesystem=True, snapshot_expiration="1d")

        assert "snapshot_expiration" not in vercel_sdk.create_kwargs[-1]

    def test_unset_expiration_leaves_the_platform_default(self, make_env, vercel_sdk):
        make_env(persistent_filesystem=False, snapshot_expiration="")
        make_env(persistent_filesystem=False, snapshot_expiration=None)

        assert "snapshot_expiration" not in vercel_sdk.create_kwargs[-2]
        assert "snapshot_expiration" not in vercel_sdk.create_kwargs[-1]

    @pytest.mark.parametrize("bad", ["1h", "30m", "-1d", "1d4", "day", "12", "1y2w", "3651d"])
    def test_bad_duration_raises_before_any_sandbox_call(self, make_env, vercel_sdk, bad):
        """A value Vercel would reject fails here, with no sandbox created and nothing to clean up."""
        with pytest.raises(ValueError, match="snapshot expiration"):
            make_env(persistent_filesystem=False, snapshot_expiration=bad)

        assert vercel_sdk.create_kwargs == []


class TestSnapshotExpirationConfig:
    """Config parsing: the value is a duration string, and a bad one fails at load, not later."""

    @pytest.mark.parametrize("raw,expected_days", [("1d", 1), ("36h", 1.5), ("2w", 14), ("1y", 365)])
    def test_durations_parse_to_timedeltas(self, vercel_module, raw, expected_days):
        from datetime import timedelta

        from tools.terminal_tool_config import parse_snapshot_expiration

        assert parse_snapshot_expiration(raw) == timedelta(days=expected_days)

    def test_empty_and_none_mean_platform_default(self):
        from tools.terminal_tool_config import parse_snapshot_expiration

        assert parse_snapshot_expiration("") is None
        assert parse_snapshot_expiration(None) is None

    def test_bad_value_is_a_config_error_at_startup(self):
        """``hermes`` prints validate_config_structure issues on startup, so a typo is visible
        there instead of surfacing when the first sandbox is created."""
        from hermes_cli.config import validate_config_structure

        issues = validate_config_structure({"terminal": {"vercel_snapshot_expiration": "1h"}})

        errors = [i for i in issues if i.severity == "error"]
        assert any("vercel_snapshot_expiration" in i.message for i in errors)

    def test_valid_value_is_not_flagged(self):
        from hermes_cli.config import validate_config_structure

        issues = validate_config_structure({"terminal": {"vercel_snapshot_expiration": "1d"}})

        assert not [i for i in issues if "vercel_snapshot_expiration" in i.message]

    def test_env_builder_rejects_a_bad_value_with_its_name(self, monkeypatch):
        from tools.terminal_tool_config import parse_snapshot_expiration
        from tools.terminal_tool_config import _parse_env_var

        raw = {"TERMINAL_VERCEL_SNAPSHOT_EXPIRATION": "1h"}
        monkeypatch.setattr("tools.terminal_tool_config._tenv", lambda name, default="": raw.get(name, default))

        with pytest.raises(ValueError, match="TERMINAL_VERCEL_SNAPSHOT_EXPIRATION"):
            _parse_env_var("TERMINAL_VERCEL_SNAPSHOT_EXPIRATION", "1d", parse_snapshot_expiration,
                           "a duration like 1d")


class TestPersistentFlag:
    """``persistent`` must be sent, not left to the platform default.

    Vercel's default is ``persistent: True``, so a Hermes sandbox created for a profile with
    ``container_persistent: false`` contradicted itself: every stop — and, more often, every hit of
    the sandbox's ``execution_time_limit`` — left an automatic ~800 MB snapshot behind, and the
    project's single snapshot quota filled up (live: the API reported ``persistent: True`` for every
    Hermes sandbox while the profile said false).
    """

    def test_ephemeral_create_sends_persistent_false(self, make_env, vercel_sdk):
        make_env(persistent_filesystem=False)

        assert vercel_sdk.create_kwargs[-1]["persistent"] is False

    def test_persistent_create_sends_persistent_true(self, make_env, vercel_sdk):
        make_env(persistent_filesystem=True)

        assert vercel_sdk.create_kwargs[-1]["persistent"] is True

    def test_restore_keeps_the_flag(self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path):
        from datetime import timedelta

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        vercel_module._store_snapshot("task-123", "snap_saved")

        make_env(persistent_filesystem=True)

        restore = vercel_sdk.create_kwargs[-1]
        assert restore["source"].snapshot_id == "snap_saved"
        assert restore["persistent"] is True
        assert "image" not in restore
        assert vercel_sdk.create_kwargs[-1]["execution_time_limit"] >= timedelta(minutes=5)


class TestResourcePair:
    """Vercel accepts ONE resource shape: ``memory == vcpus * 2048``.

    Live: creating with Hermes's shared defaults (``container_cpu 1``, ``container_memory 5120``)
    answers ``HTTP 400: `resources.memory` should equal `resources.vcpus` * 2048
    (code=bad_request)``. Every existing vercel profile only worked because it overrode the pair in
    ``.env``; a new profile on defaults could not start at all.
    """

    def test_default_pair_is_derived_to_the_legal_memory(self, make_env, vercel_sdk, caplog):
        make_env(cpu=1, memory=5120)

        assert vercel_sdk.create_kwargs[-1]["resources"] == _FakeResources(vcpus=1, memory=2048)
        assert "not legal" in caplog.text and "2048" in caplog.text

    def test_legal_pair_is_sent_untouched_and_unwarned(self, make_env, vercel_sdk, caplog):
        make_env(cpu=4, memory=8192)

        assert vercel_sdk.create_kwargs[-1]["resources"] == _FakeResources(vcpus=4, memory=8192)
        assert "not legal" not in caplog.text

    def test_fractional_cpu_floors_then_derives(self, make_env, vercel_sdk):
        make_env(cpu=2.7, memory=8192)

        assert vercel_sdk.create_kwargs[-1]["resources"] == _FakeResources(vcpus=2, memory=4096)

    def test_memory_only_config_derives_the_vcpu_count(self, make_env, vercel_sdk):
        make_env(cpu=0, memory=6144)

        assert vercel_sdk.create_kwargs[-1]["resources"] == _FakeResources(vcpus=3, memory=6144)

    def test_unsatisfiable_pair_sends_no_resources_rather_than_an_illegal_one(
        self, make_env, vercel_sdk
    ):
        make_env(cpu=0, memory=5000)

        assert vercel_sdk.create_kwargs[-1]["resources"] is None

    def test_no_positive_resources_means_no_resources_argument(self, make_env, vercel_sdk):
        make_env(cpu=0, memory=0)

        assert vercel_sdk.create_kwargs[-1]["resources"] is None
