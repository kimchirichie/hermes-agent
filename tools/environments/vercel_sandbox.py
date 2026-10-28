"""Vercel Sandbox execution environment.

Runs commands in Vercel cloud sandboxes through the shared ``BaseEnvironment``
shell contract. With persistence enabled, task-scoped snapshot ids are stored
under ``HERMES_HOME`` and new sandboxes are restored from them on task reuse.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import shlex
import threading
import time
from datetime import timedelta
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from hermes_cli.config_defaults import DEFAULT_VERCEL_IMAGE
from hermes_constants import get_hermes_home
from tools.environments.base import BaseEnvironment, _load_json_store, _update_json_store
from tools.environments.base_output import _ThreadedProcessHandle
from tools.environments.file_sync import FileSyncManager, iter_sync_files, quoted_rm_command

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vercel.sandbox.sync import SandboxStatus, SyncSandbox, SyncSnapshot

DEFAULT_VERCEL_CWD = "/vercel/sandbox"
_DEFAULT_CONTAINER_DISK_MB = 51200
_CREATE_RETRY_ATTEMPTS = 3
_TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_RUNNING_WAIT_TIMEOUT = timedelta(seconds=30)


def _ensure_vercel_sdk() -> None:
    """Lazy-install vercel SDK on demand. Idempotent."""
    # The SDK (>=0.7) ships default-on telemetry; Hermes policy is opt-in only, so disable it
    # before the SDK is imported. setdefault: an explicit user value is never overridden.
    os.environ.setdefault("VERCEL_TELEMETRY_DISABLED", "1")
    try:
        from pm import ensure_import as _lazy_ensure
    except ImportError:
        return  # pm unavailable — vercel's own import sites decide
    try:
        _lazy_ensure("vercel")
    except Exception as e:
        raise ImportError(str(e))
_WRITE_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_STEP = timedelta(milliseconds=100)
_MIN_SANDBOX_TIMEOUT = timedelta(minutes=5)
_MIN_RUNNING_WAIT = timedelta(seconds=1)
_RUNNING_WAIT_POLL_INTERVAL = timedelta(milliseconds=250)
_STOP_TIMEOUT = timedelta(seconds=15)
_STOP_POLL_INTERVAL = timedelta(milliseconds=500)
_SNAPSHOT_STORE_NAME = "vercel_sandbox_snapshots.json"
_SETUP_SCRIPT_REMOTE = "/tmp/hermes_vercel_setup.sh"
_SETUP_OUTPUT_TAIL = 4000
DEFAULT_VERCEL_SETUP_TIMEOUT = 300


def _resolve_setup_script(raw: str) -> Path:
    """Absolute host path, or one resolved against the profile home."""
    path = Path(os.path.expanduser(raw.strip()))
    return path if path.is_absolute() else get_hermes_home() / path


def _tail_cap(text: str, limit: int | None) -> str:
    if not limit or len(text) <= limit:
        return text
    return "…(truncated)…\n" + text[-limit:]


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _extract_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    for value in (getattr(exc, "status_code", None), getattr(response, "status_code", None)):
        if isinstance(value, int):
            return value
    return None

@cache
def _transport_error_types() -> tuple[type[BaseException], ...]:
    """Network exceptions of every HTTP stack the SDK may raise from.

    vercel >= 0.12 moved the sandbox client from httpx to httpx2, a separate
    distribution whose exceptions do not subclass httpx's.
    """
    types: list[type[BaseException]] = [httpx.NetworkError, httpx.ProtocolError, httpx.ReadError]
    try:
        import httpx2
    except ImportError:
        return tuple(types)
    return (*types, httpx2.NetworkError, httpx2.ProtocolError, httpx2.ReadError)


def _is_transient_vercel_error(exc: BaseException) -> bool:
    """True when any exception in the cause/context chain looks retryable."""
    seen: set[int] = set()
    error: BaseException | None = exc
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        codes = (getattr(error, "status_code", None), getattr(getattr(error, "response", None), "status_code", None))
        status = next((c for c in codes if isinstance(c, int)), None)
        name = type(error).__name__.lower()
        if (status in _TRANSIENT_STATUS_CODES
                or isinstance(error, _transport_error_types())
                or "ratelimit" in name or "servererror" in name):
            return True
        error = error.__cause__ or error.__context__
    return False


def _retry_vercel_call(label: str, callback, *, attempts: int):
    for attempt in range(1, attempts + 1):
        try:
            return callback()
        except Exception as exc:
            if attempt >= attempts or not _is_transient_vercel_error(exc):
                raise
            logger.warning("Vercel: %s failed (%s); retrying %d/%d", label, exc, attempt, attempts)
            time.sleep(0.1 * attempt)


def _result_parts(result: Any) -> tuple[str, int]:
    """(output, returncode) from an SDK ``CompletedProcess``; tolerates raw strings.

    The 0.12 sync SDK returns stdout/stderr separately (no ``output()`` helper),
    so both streams are concatenated to keep the single-output contract.
    """
    if isinstance(result, str):
        return result, 0
    stdout, stderr = getattr(result, "stdout", None), getattr(result, "stderr", None)
    parts: list[str] = []
    for value in (stdout, stderr):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if value:
            parts.append(str(value))
    exit_code = getattr(result, "returncode", None)
    return "".join(parts), (exit_code if isinstance(exit_code, int) else 1)


def _snapshot_store() -> Path:
    return get_hermes_home() / "vercel_sandbox_snapshots.json"


def _load_snapshots() -> dict:
    return _load_json_store(_snapshot_store())


def _store_snapshot(task_id: str, snapshot_id: str) -> None:
    if task_id and snapshot_id:
        _update_json_store(_snapshot_store(), lambda snapshots: snapshots.update({task_id: snapshot_id}))


def _delete_snapshot(task_id: str, snapshot_id: str) -> None:
    """Drop the stored id for ``task_id`` only if it still equals ``snapshot_id``."""
    def update(snapshots: dict) -> None:
        if task_id and snapshots.get(task_id) == snapshot_id:
            snapshots.pop(task_id)

    _update_json_store(_snapshot_store(), update)


def _extract_snapshot_id(snapshot: Any) -> str | None:
    """Accept SDK objects or raw dicts; attribute lookup first, then dict keys."""
    getters = [lambda k: getattr(snapshot, k, None)] + ([snapshot.get] if isinstance(snapshot, dict) else [])
    candidates = (get(key) for get in getters for key in ("snapshot_id", "snapshotId", "id"))
    return next((v for v in candidates if isinstance(v, str) and v), None)


@cache
def _sandbox_status_type() -> type[SandboxStatus]:
    _ensure_vercel_sdk()
    from vercel.sandbox.sync import SandboxStatus
    return SandboxStatus


def _is_terminal(status: Any) -> bool:
    S = _sandbox_status_type()
    return status in {S.ABORTED, S.FAILED, S.STOPPED}


class VercelSandboxEnvironment(BaseEnvironment):
    """Vercel cloud sandbox backend."""

    _stdin_mode = "payload"

    def __init__(self, runtime: str | None = None, cwd: str = DEFAULT_VERCEL_CWD, timeout: int = 60,
                 cpu: float = 1, memory: int = 5120, disk: int = _DEFAULT_CONTAINER_DISK_MB,
                 persistent_filesystem: bool = True, task_id: str = "default",
                 image: str | None = None, setup_script: str | None = None,
                 setup_timeout: int = DEFAULT_VERCEL_SETUP_TIMEOUT):
        super().__init__(cwd=cwd, timeout=timeout)
        if disk not in {0, _DEFAULT_CONTAINER_DISK_MB}:
            raise ValueError(
                "Vercel Sandbox does not support configurable container_disk. "
                "Use the default shared setting.")
        self._persistent, self._task_id, self._requested_cwd = persistent_filesystem, task_id, cwd
        self._setup_script = (setup_script or "").strip() or None
        self._setup_timeout = setup_timeout
        self._lock = threading.Lock()
        self._sandbox: SyncSandbox | None = None
        self._workspace_root = self._remote_home = DEFAULT_VERCEL_CWD
        self._sync_manager: FileSyncManager | None = None
        _ensure_vercel_sdk()
        from vercel.sandbox.sync import SandboxResources
        vcpus, memory_mb = (math.floor(cpu) if cpu > 0 else None), (memory if memory > 0 else None)
        # Vercel deprecated runtimes (Aug 2026) and the 0.12 SDK no longer accepts one:
        # a pinned runtime is logged and dropped, and the image is always sent instead.
        if runtime:
            logger.warning("Vercel: terminal.vercel_runtime is deprecated by Vercel and unsupported by "
                           "vercel==0.12.0; ignoring %r and using terminal.vercel_image (default %s)",
                           runtime, DEFAULT_VERCEL_IMAGE)
        self._create_kwargs = {
            "execution_time_limit": max(timedelta(seconds=max(self.timeout, 0)), timedelta(minutes=5)),
            "image": image or DEFAULT_VERCEL_IMAGE,
            "resources": SandboxResources(vcpus=vcpus, memory=memory_mb) if (vcpus, memory_mb) != (None, None) else None}
        self._attach_fresh_sandbox(cwd)
        self.init_session()

    def _require_sandbox(self) -> SyncSandbox:
        if self._sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        return self._sandbox

    def _remote_hermes_dir(self) -> str:
        return f"{self._remote_home.rstrip('/')}/.hermes"

    def _create_sandbox(self) -> SyncSandbox:
        _ensure_vercel_sdk()
        from vercel.sandbox.sync import SnapshotSource, create_sandbox
        snapshot_id = _load_snapshots().get(self._task_id) if self._persistent and self._task_id else None
        if isinstance(snapshot_id, str) and snapshot_id:
            try:
                source = SnapshotSource(snapshot_id=snapshot_id)
                # A snapshot carries its own filesystem: the image would describe a base the
                # snapshot already replaced, so it is dropped from the restore call.
                restore_kwargs = {k: v for k, v in self._create_kwargs.items() if k != "image"}
                return _retry_vercel_call(
                    "sandbox restore", lambda: create_sandbox(**restore_kwargs, source=source),
                    attempts=_CREATE_RETRY_ATTEMPTS)
            except Exception as exc:
                logger.warning("Vercel: failed to restore snapshot %s for task %s; falling back to a fresh sandbox: %s",
                               snapshot_id, self._task_id, exc)
                _delete_snapshot(self._task_id, snapshot_id)
        return _retry_vercel_call("sandbox create", lambda: create_sandbox(**self._create_kwargs),
                                  attempts=_CREATE_RETRY_ATTEMPTS)

    def _attach_fresh_sandbox(self, requested_cwd: str) -> None:
        """Create a sandbox, wait until it runs, then wire cwd/home, sync the managed files and
        run the optional setup script before any session is initialised."""
        self._sandbox = self._create_sandbox()
        self._wait_for_running()
        cwd = self._require_sandbox().cwd or ""
        self._workspace_root = cwd if cwd.startswith("/") else DEFAULT_VERCEL_CWD
        self._remote_home = self._detect_remote_home()
        container_base = self._remote_hermes_dir()
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(container_base),
            upload_fn=lambda host_path, remote_path: self._vercel_bulk_upload([(host_path, remote_path)]),
            delete_fn=self._vercel_delete,
            bulk_upload_fn=self._vercel_bulk_upload, bulk_download_fn=self._vercel_bulk_download)
        self.cwd = {"~": self._remote_home, "": self._workspace_root,
                    DEFAULT_VERCEL_CWD: self._workspace_root}.get(requested_cwd, requested_cwd)
        self._sync_manager.sync(force=True)
        self._run_setup_script()

    def _run_setup_script(self) -> None:
        """Upload and run terminal.vercel_setup_script once per fresh sandbox; it must be re-runnable."""
        if not self._setup_script:
            return
        host_path = _resolve_setup_script(self._setup_script)
        try:
            host_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Vercel setup failed: cannot read {host_path}: {exc}") from exc
        self._vercel_bulk_upload([(str(host_path), _SETUP_SCRIPT_REMOTE)])
        try:
            self._run_checked(f"bash {shlex.quote(_SETUP_SCRIPT_REMOTE)}", "setup",
                              tail=_SETUP_OUTPUT_TAIL, timeout=self._setup_timeout)
        except Exception:
            # Construction raises, so cleanup() never runs: stop the half-set-up box here.
            self._stop_sandbox(self._sandbox)
            self._sandbox = None
            raise

    def _detect_remote_home(self) -> str:
        try:
            result = self._require_sandbox().run_process(
                "sh", ["-lc", 'printf %s "$HOME"'], cwd=self._workspace_root, capture_output=True)
        except Exception as exc:
            logger.debug("Vercel: home detection failed for task %s: %s", self._task_id, exc)
            return self._workspace_root
        home = _result_parts(result)[0].strip()
        return home if home.startswith("/") else self._workspace_root

    def _wait_for_running(self, timeout: timedelta = _RUNNING_WAIT_TIMEOUT) -> None:
        """Assert the sandbox is running; the 0.12 sync SDK's create/restore call already waits."""
        del timeout  # the SDK blocks inside create_sandbox; there is no poll-until-ready helper
        sandbox, running = self._require_sandbox(), _sandbox_status_type().RUNNING
        status = sandbox.status
        if status is None or status == running:
            return
        if _is_terminal(status):
            raise RuntimeError(f"Sandbox entered terminal state: {status}")
        raise RuntimeError(f"Sandbox did not reach running state (last status: {status})")

    @staticmethod
    def _stop_sandbox(sandbox: SyncSandbox | None) -> None:
        if sandbox is None:
            return
        with contextlib.suppress(Exception):
            sandbox.stop()

    def _snapshot_sandbox(self, sandbox: SyncSandbox) -> None:
        if not self._persistent or not self._task_id:
            return
        try:
            snapshot_id = _extract_snapshot_id(sandbox.snapshot())
        except Exception as exc:
            logger.warning("Vercel: filesystem snapshot failed for task %s: %s", self._task_id, exc)
            return
        if not snapshot_id:
            logger.warning("Vercel: filesystem snapshot for task %s did not return a snapshot id", self._task_id)
            return
        _store_snapshot(self._task_id, snapshot_id)
        logger.info("Vercel: saved filesystem snapshot %s for task %s", snapshot_id, self._task_id)

    def _ensure_sandbox_ready(self) -> None:
        """Reuse a healthy sandbox; recreate when the refresh fails or it hit a terminal state."""
        sandbox = self._sandbox
        requested_cwd = self.cwd or self._requested_cwd or DEFAULT_VERCEL_CWD
        if sandbox is not None:
            try:
                # 0.12 has no handle.refresh(); re-fetching by name is the equivalent read.
                from vercel.sandbox.sync import get_sandbox
                self._sandbox = sandbox = get_sandbox(name=sandbox.name)
            except Exception as exc:
                logger.warning("Vercel: sandbox refresh failed for task %s: %s; recreating", self._task_id, exc)
            else:
                status = sandbox.status
                if not _is_terminal(status):
                    self._wait_for_running()
                    return
                logger.warning("Vercel: sandbox entered state %s for task %s; recreating", status, self._task_id)
            self._mark_recreated()
        self._attach_fresh_sandbox(requested_cwd)

    def _run_checked(self, script: str, label: str, *, tail: int | None = None,
                     timeout: int | None = None) -> None:
        """Run *script* under ``bash -lc``; ``timeout`` maps to the SDK's ``kill_after`` seconds."""
        kwargs = {"cwd": self._workspace_root, "capture_output": True}
        if timeout is not None:
            kwargs["kill_after"] = timeout
        output, returncode = _result_parts(
            self._require_sandbox().run_process("bash", ["-lc", script], **kwargs))
        if returncode != 0:
            raise RuntimeError(f"Vercel {label} failed: {_tail_cap(output.strip(), tail)}")

    def _vercel_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        if not files:
            return
        payload = [(remote_path, Path(host_path).read_bytes()) for host_path, remote_path in files]
        sandbox = self._require_sandbox()

        def upload() -> None:
            with sandbox.fs.batch() as batch:
                for remote_path, content in payload:
                    batch.write_bytes(remote_path, content)

        _retry_vercel_call("write_files", upload, attempts=3)

    def _vercel_delete(self, remote_paths: list[str]) -> None:
        if remote_paths:
            self._run_checked(quoted_rm_command(remote_paths), "delete")

    def _vercel_bulk_download(self, dest_tar_path: Path) -> None:
        archive_member = self._remote_hermes_dir().lstrip("/")
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"  # no-tmp: ok — remote sandbox path
        sandbox = self._require_sandbox()
        try:
            # --exclude: live sockets cannot be archived ("socket ignored") and must not fail the download.
            self._run_checked(f"tar cf {shlex.quote(remote_tar)} --exclude='*.sock' -C / {shlex.quote(archive_member)}",
                              "bulk download")
            # 0.12 exposes no download_file(); the archive is read back through the filesystem handle.
            dest_tar_path.write_bytes(sandbox.fs.read_bytes(remote_tar))
        finally:
            with contextlib.suppress(Exception):
                sandbox.run_process("bash", ["-lc", f"rm -f {shlex.quote(remote_tar)}"],
                                    cwd=self._workspace_root, capture_output=True)

    def _before_execute(self) -> None:
        with self._lock:
            self._ensure_sandbox_ready()
            if self._sync_manager is not None:
                self._sync_manager.sync()

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120, stdin_data: str | None = None):
        """``timeout`` is enforced by the base ``_wait_for_process`` via ``cancel_fn`` (the SDK has no
        per-exec timeout). Payload stdin is staged through the SDK so it never becomes a shell argv."""
        del timeout
        sandbox, workspace_root, lock = self._require_sandbox(), self._workspace_root, self._lock

        # Guarded by ``lock`` so cancel() and dispatch agree on whether the shell
        # has taken ownership of (opened + unlinked) the staged stdin file.
        state = {"cancelled": False, "staged": None, "dispatched": False}

        def scrub_staged() -> None:  # caller holds ``lock``
            if state["staged"] and not state["dispatched"]:
                # Staged but never dispatched: scrub the payload (may hold a
                # sudo password). Once dispatched the user shell unlinks it
                # itself, so kill() sends no extra command.
                with contextlib.suppress(Exception):
                    sandbox.fs.write_bytes(state["staged"], b"", mode=0o600)
                state["staged"] = None

        def cancel() -> None:
            with lock:
                state["cancelled"] = True
                scrub_staged()
                self._stop_sandbox(sandbox)

        def exec_fn() -> tuple[str, int]:
            command = cmd_string
            if stdin_data:  # empty stdin == no stdin, as on base (heredoc skipped it)
                with lock:
                    if state["cancelled"]:
                        return ("", 130)
                remote_stdin = self._staged_stdin_path()
                stdin_payload = stdin_data.encode("utf-8", "surrogateescape")
                _retry_vercel_call(
                    "stdin upload",
                    lambda: sandbox.fs.write_bytes(remote_stdin, stdin_payload, mode=0o600),
                    attempts=_WRITE_RETRY_ATTEMPTS,
                )
                command = self._redirect_stdin_from_file(cmd_string, remote_stdin)
                with lock:
                    state["staged"] = remote_stdin
            with lock:
                if state["cancelled"]:
                    # cancel() may have run mid-upload, before ``staged`` was set.
                    scrub_staged()
                    return ("", 130)
                state["dispatched"] = True
            return _result_parts(sandbox.run_process(
                "bash", ["-lc" if login else "-c", command], cwd=workspace_root, capture_output=True))
        return _ThreadedProcessHandle(exec_fn, cancel_fn=cancel)

    def cleanup(self):
        with self._lock:
            sandbox, sync_manager = self._sandbox, self._sync_manager
            if sandbox is not None and sync_manager is not None:
                try:
                    sync_manager.sync_back()
                except Exception as exc:
                    logger.warning("Vercel: sync_back failed for task %s: %s", self._task_id, exc)
            self._sandbox = None
            self._sync_manager = None
        if sandbox is None:
            return
        self._snapshot_sandbox(sandbox)
        # Always stop the sandbox during cleanup to avoid resource leaks (matches Modal/Daytona).
        self._stop_sandbox(sandbox)
