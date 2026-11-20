"""A queued gateway restart stops the embedded dispatcher from spawning new cards."""

import asyncio
from unittest.mock import MagicMock, patch

from gateway.run import GatewayRunner


def _run_one_tick(restart_requested):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner._restart_requested = restart_requested
    dispatcher = MagicMock()
    dispatcher.tick_once.return_value = []
    dispatcher.ready_nonempty.return_value = False
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def stop_after_one_tick(interval):
        runner._running = False

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    boot = (lambda: {"kanban": {}}, MagicMock(), {})
    with patch.object(runner, "_kanban_dispatcher_boot", return_value=boot), \
            patch("gateway.kanban_watchers._resolve_dispatcher_settings") as settings, \
            patch("gateway.kanban_watchers._KanbanDispatcher", return_value=dispatcher), \
            patch("gateway.kanban_watchers._kanban_dispatch_allowed", return_value=True), \
            patch("gateway.kanban_watchers._resolve_auto_decompose_settings", return_value=(False, 0)), \
            patch("gateway.kanban_watchers._to_thread_process_service", fake_to_thread), \
            patch("hermes_cli.kanban_db_dispatch.reap_worker_zombies", return_value=[]), \
            patch("asyncio.sleep", fake_sleep), \
            patch.object(runner, "_sleep_between_ticks", stop_after_one_tick), \
            patch.object(runner, "_release_kanban_dispatcher_lock", create=True):
        settings.return_value.interval = 60.0
        asyncio.run(runner._kanban_dispatcher_watcher())
    return dispatcher


def test_dispatcher_spawns_when_no_restart_is_queued():
    dispatcher = _run_one_tick(restart_requested=False)

    dispatcher.tick_once.assert_called_once()


def test_dispatcher_spawns_nothing_while_a_restart_is_queued():
    dispatcher = _run_one_tick(restart_requested=True)

    dispatcher.tick_once.assert_not_called()
