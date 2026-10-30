"""End-of-run release for cloud sandbox terminal envs.

A cloud sandbox env (Vercel) is cached under the SESSION key (``session:<key>``) or the shared
``"default"`` — never under the turn's own task id — so the per-turn ``cleanup_vm(task_id)`` could
not find it. A long-lived gateway serving one-shot runs therefore left the sandbox running after the
CLI exited 0 (live: four ``hermes -p oap-reviewer -z`` runs, each still ``running`` minutes later),
and with the platform's default ``persistent: true`` each abandoned box also produced a snapshot.

These tests pin every guard on ``release_run_env``: sabotage one and its test fails.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

# Imported at COLLECTION time on purpose: importing agent.chat_completion_helpers boots
# hermes_bootstrap (it resolves the runtime dir), which the per-test home_io_guard forbids.
from agent.chat_completion_helpers import cleanup_task_resources
from tools import terminal_tool as tt
from tools import terminal_tool_lifecycle as lifecycle


class _FakeCloudEnv:
    """A non-persistent Vercel-style env: declares that a finished run releases it."""

    env_type = "vercel_sandbox"
    _release_when_run_finishes = True

    def __init__(self, *, persistent: bool = False):
        self._persistent = persistent
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class _FakeOrdinaryEnv:
    """A backend that does NOT declare release-when-finished (docker/local/persistent-style)."""

    env_type = "docker"

    def __init__(self):
        self._persistent = False
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    """Importing agent.chat_completion_helpers boots hermes_bootstrap: never the real home."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))


@pytest.fixture(autouse=True)
def _clean_terminal_state():
    for store in (tt._active_environments, tt._last_activity, tt._calls_in_flight,
                  tt._env_users, tt._env_created_by, tt._env_one_shot, tt._creation_locks):
        store.clear()
    yield
    for store in (tt._active_environments, tt._last_activity, tt._calls_in_flight,
                  tt._env_users, tt._env_created_by, tt._env_one_shot, tt._creation_locks):
        store.clear()


@pytest.fixture()
def session_key(monkeypatch):
    """Make the env key resolve to ``session:s1``, the way a gateway turn's key does."""
    monkeypatch.setattr(tt, "_current_session_key", lambda: "s1")
    return "session:s1"


def _cache(key: str, env, *, created_by: str | None, users: tuple[str, ...], one_shot=None):
    tt._active_environments[key] = env
    tt._last_activity[key] = 0.0
    if created_by is not None:
        tt._env_created_by[key] = created_by
    if one_shot is not None:
        tt._env_one_shot[key] = one_shot
    tt._env_users[key] = set(users)


def _stub_run_agent():
    """``run_agent``'s two cleanup entry points; importing the real module needs a live install."""
    return SimpleNamespace(cleanup_vm=Mock(), cleanup_browser=Mock())


def _one_shot(monkeypatch, value: bool | None = True):
    with patch("gateway.session_finite.finite_turn_required", return_value=value):
        return lifecycle.release_run_env("turn-1")


class TestOneShotTurnRelease:
    def test_a_finished_one_shot_turn_stops_the_sandbox_it_created(self, session_key, monkeypatch):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))

        assert _one_shot(monkeypatch) is True
        assert env.cleanup_calls == 1
        assert session_key not in tt._active_environments

    def test_a_live_conversation_keeps_its_sandbox(self, session_key, monkeypatch):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))

        # finite=False: the gateway session will take another turn on this same sandbox.
        assert _one_shot(monkeypatch, value=False) is False
        assert env.cleanup_calls == 0
        assert session_key in tt._active_environments

    def test_a_goalloop_turn_without_a_finite_admission_keeps_its_sandbox(self, session_key, monkeypatch):
        """``None`` is a legacy CLI / kanban goal_mode turn: the same session keeps working."""
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))

        assert _one_shot(monkeypatch, value=None) is False
        assert env.cleanup_calls == 0

    def test_a_sandbox_another_run_still_uses_is_not_stopped(self, session_key, monkeypatch):
        """A delegate child shares its parent's session key; the child's turn end must not kill it."""
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1", "turn-child"))

        assert _one_shot(monkeypatch) is False
        assert env.cleanup_calls == 0

    def test_a_sandbox_an_earlier_turn_created_is_left_alone(self, session_key, monkeypatch):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-0", users=("turn-0", "turn-1"))

        assert _one_shot(monkeypatch) is False
        assert env.cleanup_calls == 0

    def test_a_recorded_one_shot_release_survives_a_lost_turn_context(self, session_key, monkeypatch):
        """The turn finalizer may run without the admission's context: the env recorded its origin."""
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",), one_shot=True)

        assert _one_shot(monkeypatch, value=None) is True
        assert env.cleanup_calls == 1

    def test_a_recorded_chat_env_is_not_released_by_a_stray_finite_context(self, session_key, monkeypatch):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",), one_shot=False)

        assert _one_shot(monkeypatch, value=True) is False
        assert env.cleanup_calls == 0

    def test_a_persistent_filesystem_is_never_released(self, session_key, monkeypatch):
        env = _FakeCloudEnv(persistent=True)
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))

        assert _one_shot(monkeypatch) is False
        assert env.cleanup_calls == 0

    def test_a_backend_that_does_not_declare_it_is_untouched(self, session_key, monkeypatch):
        env = _FakeOrdinaryEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))

        assert _one_shot(monkeypatch) is False
        assert env.cleanup_calls == 0
        assert session_key in tt._active_environments

    def test_an_in_flight_call_blocks_the_release(self, session_key, monkeypatch):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))
        tt._calls_in_flight[session_key] = 1

        assert _one_shot(monkeypatch) is False
        assert env.cleanup_calls == 0

    def test_nothing_cached_is_a_no_op(self, session_key, monkeypatch):
        assert _one_shot(monkeypatch) is False


class TestSessionEndRelease:
    def test_session_end_stops_the_sessions_sandbox_whatever_turn_created_it(self, session_key):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1", "turn-2"))

        assert lifecycle.release_run_env("session-id", session_ending=True) is True
        assert env.cleanup_calls == 1
        assert session_key not in tt._active_environments

    def test_session_end_still_respects_persistence(self, session_key):
        env = _FakeCloudEnv(persistent=True)
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))

        assert lifecycle.release_run_env("session-id", session_ending=True) is False
        assert env.cleanup_calls == 0

    def test_a_live_call_survives_session_end(self, session_key):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))
        tt._calls_in_flight[session_key] = 2

        assert lifecycle.release_run_env("session-id", session_ending=True) is False
        assert env.cleanup_calls == 0


class TestTurnFinalizerWiring:
    """``cleanup_task_resources`` is what actually runs at the end of every turn."""

    def test_a_finished_one_shot_turn_releases_through_the_per_turn_cleanup(self, session_key, monkeypatch):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))
        agent = SimpleNamespace(verbose_logging=False)
        with (
            patch("tools.browser_session_cloud.is_headed_mode", return_value=False),
            patch("agent.chat_completion_helpers._ra", return_value=_stub_run_agent()),
            patch("agent.chat_completion_helpers.is_persistent_env", return_value=False),
            patch("gateway.session_finite.finite_turn_required", return_value=True),
        ):
            cleanup_task_resources(agent, "turn-1")

        assert env.cleanup_calls == 1
        assert session_key not in tt._active_environments

    def test_a_chat_turn_leaves_the_sandbox_for_the_next_turn(self, session_key):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))
        agent = SimpleNamespace(verbose_logging=False)
        with (
            patch("tools.browser_session_cloud.is_headed_mode", return_value=False),
            patch("agent.chat_completion_helpers._ra", return_value=_stub_run_agent()),
            patch("agent.chat_completion_helpers.is_persistent_env", return_value=False),
            patch("gateway.session_finite.finite_turn_required", return_value=False),
        ):
            cleanup_task_resources(agent, "turn-1")

        assert env.cleanup_calls == 0
        assert session_key in tt._active_environments


class TestUserTracking:
    """The cache records WHO used/created each resolved key; without it the guards are blind."""

    def test_lookup_records_the_using_run(self, session_key):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by=None, users=())
        tt._last_activity.pop(session_key, None)
        with tt._env_lock:
            assert tt._lookup_active_env(session_key, "turn-9") is env
        assert tt._env_users[session_key] == {"turn-9"}

    def test_creation_records_whether_the_creating_run_was_a_one_shot(self, session_key):
        with patch("gateway.session_finite.finite_turn_required", return_value=True):
            with tt._env_lock:
                tt._note_env_user(session_key, "turn-1", created=True)
        assert tt._env_one_shot[session_key] is True

    def test_creation_records_a_conversation_turn_as_not_one_shot(self, session_key):
        with patch("gateway.session_finite.finite_turn_required", return_value=False):
            with tt._env_lock:
                tt._note_env_user(session_key, "turn-1", created=True)
        assert tt._env_one_shot[session_key] is False

    def test_unregister_forgets_the_tracking(self, session_key):
        env = _FakeCloudEnv()
        _cache(session_key, env, created_by="turn-1", users=("turn-1",))

        lifecycle._unregister_env(session_key)

        assert session_key not in tt._env_users
        assert session_key not in tt._env_created_by
        assert session_key not in tt._env_one_shot


class TestVercelDeclaresIt:
    def test_the_vercel_backend_opts_in(self):
        from tools.environments.vercel_sandbox import VercelSandboxEnvironment

        assert VercelSandboxEnvironment._release_when_run_finishes is True
