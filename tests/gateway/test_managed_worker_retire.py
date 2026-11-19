"""A managed worker's turn-final memory work drains before the provider shuts down."""
import hermes_bootstrap  # imported at collection: its dependency probe reads <checkout parent>/manifest.json, which the real-home guard refuses once a test is running


def test_retire_agent_flushes_pending_memory_before_shutdown():
    from agent.managed_worker import retire_agent

    calls = []

    class Manager:
        def flush_pending(self, timeout=None):
            calls.append(('flush', timeout))
            return True

    class Agent:
        _memory_manager = Manager()
        _session_messages = [{'role': 'user', 'content': 'hi'}]

        def shutdown_memory_provider(self, messages):
            calls.append(('shutdown', len(messages)))

        def release_clients(self):
            calls.append(('release',))

    retire_agent(Agent())
    assert calls == [('flush', 10), ('shutdown', 1), ('release',)]
