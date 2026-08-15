"""Tests for AgentExecutor single-tick execution."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.agents._stubs import AgentResult
from openjarvis.agents.errors import FatalError, RetryableError
from openjarvis.core.events import EventBus, EventType


class _PersonalMemorySpy:
    def __init__(self) -> None:
        self.archived: list[tuple[str, str, str]] = []

    def archive_exchange(self, *, user_text, assistant_text, source, **kwargs):
        self.archived.append((user_text, assistant_text, source))
        return SimpleNamespace(id="managed-agent-exchange")

    def enqueue_exchange(self, exchange_id: str) -> bool:
        return True


class _ReplyingAgent:
    accepts_tools = False
    supports_managed_tool_fallback = False

    def __init__(self, engine, model, *, bus=None) -> None:
        self._reply = "Start with 25 minutes."

    def run(self, input_text, context=None):
        return AgentResult(content=self._reply)


class _EmptyToolkit:
    instances = []
    by_name = {}
    mcp_clients = []

    def close(self) -> None:
        pass


@pytest.fixture
def manager():
    from openjarvis.agents.manager import AgentManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AgentManager(db_path=str(Path(tmpdir) / "agents.db"))
        yield mgr
        mgr.close()


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def executor(manager, event_bus):
    from openjarvis.agents.executor import AgentExecutor

    mock_system = MagicMock()
    ex = AgentExecutor(manager=manager, event_bus=event_bus)
    ex.set_system(mock_system)
    return ex


class TestExecutorBasic:
    def test_execute_tick_publishes_start_end_events(
        self, executor, manager, event_bus
    ):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")
        events = []
        event_bus.subscribe(EventType.AGENT_TICK_START, lambda e: events.append(e))
        event_bus.subscribe(EventType.AGENT_TICK_END, lambda e: events.append(e))

        rv = AgentResult(content="result text")
        with patch.object(executor, "_invoke_agent", return_value=rv):
            executor.execute_tick(agent["id"])

        assert len(events) == 2
        assert events[0].event_type == EventType.AGENT_TICK_START
        assert events[1].event_type == EventType.AGENT_TICK_END

    def test_execute_tick_updates_run_stats(self, executor, manager):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")

        rv = AgentResult(content="result text")
        with patch.object(executor, "_invoke_agent", return_value=rv):
            executor.execute_tick(agent["id"])

        updated = manager.get_agent(agent["id"])
        assert updated["total_runs"] == 1
        assert updated["status"] == "idle"

    def test_execute_tick_sets_running_then_idle(self, executor, manager):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")
        statuses = []

        original_start = manager.start_tick

        def track_start(aid):
            original_start(aid)
            statuses.append(manager.get_agent(aid)["status"])

        manager.start_tick = track_start

        rv = AgentResult(content="result")
        with patch.object(executor, "_invoke_agent", return_value=rv):
            executor.execute_tick(agent["id"])

        assert statuses == ["running"]
        assert manager.get_agent(agent["id"])["status"] == "idle"

    def test_execute_tick_handles_fatal_error(self, executor, manager, event_bus):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")
        errors = []
        event_bus.subscribe(EventType.AGENT_TICK_ERROR, lambda e: errors.append(e))

        with patch.object(
            executor, "_invoke_agent", side_effect=FatalError("bad config")
        ):
            executor.execute_tick(agent["id"])

        assert manager.get_agent(agent["id"])["status"] == "error"
        assert len(errors) == 1

    def test_execute_tick_retries_retryable_error(self, executor, manager):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")
        call_count = 0

        def flaky_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("rate limit")
            return AgentResult(content="success")

        with patch.object(executor, "_invoke_agent", side_effect=flaky_invoke):
            with patch("openjarvis.agents.executor.retry_delay", return_value=0):
                executor.execute_tick(agent["id"])

        assert call_count == 3
        assert manager.get_agent(agent["id"])["status"] == "idle"

    def test_execute_tick_gives_up_after_max_retries(self, executor, manager):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")

        with patch.object(
            executor, "_invoke_agent", side_effect=RetryableError("always fails")
        ):
            with patch("openjarvis.agents.executor.retry_delay", return_value=0):
                executor.execute_tick(agent["id"])

        assert manager.get_agent(agent["id"])["status"] == "error"

    def test_execute_tick_concurrency_guard(self, executor, manager):
        agent = manager.create_agent(name="test", agent_type="monitor_operative")
        manager.start_tick(agent["id"])  # Simulate already running

        # Second tick should handle the ValueError from start_tick
        rv = AgentResult(content="result")
        with patch.object(executor, "_invoke_agent", return_value=rv):
            executor.execute_tick(agent["id"])

        # Agent should still be running (first tick owns it)
        assert manager.get_agent(agent["id"])["status"] == "running"


def test_finalize_tick_reads_agent_result_metadata(tmp_path):
    """_finalize_tick() accumulates cost/tokens from AgentResult.metadata."""
    from openjarvis.agents.executor import AgentExecutor
    from openjarvis.agents.manager import AgentManager

    mgr = AgentManager(str(tmp_path / "test.db"))
    bus = EventBus()
    executor = AgentExecutor(mgr, bus)

    agent = mgr.create_agent("budget-agent")
    mgr.start_tick(agent["id"])

    result = AgentResult(
        content="done",
        metadata={"tokens_used": 500, "cost": 0.05},
    )
    executor._finalize_tick(agent["id"], result, error=None, duration=1.0)

    updated = mgr.get_agent(agent["id"])
    assert updated["total_tokens"] == 500
    assert updated["total_cost"] == 0.05
    assert updated["stall_retries"] == 0
    mgr.close()


def _managed_executor(manager, event_bus, personal_memory_service):
    from openjarvis.agents.executor import AgentExecutor

    system = SimpleNamespace(
        engine=object(),
        memory_backend=None,
        channel_backend=None,
        knowledge_db_path=None,
        config=SimpleNamespace(agent=SimpleNamespace(context_from_memory=False)),
    )
    return AgentExecutor(
        manager,
        event_bus,
        system=system,
        personal_memory_service=personal_memory_service,
    )


def test_successful_agent_turn_archives_its_pending_user_message(manager, event_bus):
    """A successful managed reply must keep the exact user request as evidence."""
    from openjarvis.agents import AgentRegistry

    AgentRegistry.register_value("personal_memory_test_agent", _ReplyingAgent)
    personal = _PersonalMemorySpy()
    executor = _managed_executor(manager, event_bus, personal)
    agent = manager.create_agent(
        name="companion",
        agent_type="personal_memory_test_agent",
        config={"model": "test-model"},
    )
    manager.send_message(agent["id"], "Help me make a study plan.", mode="immediate")

    with patch(
        "openjarvis.agents.executor.resolve_agent_tools",
        return_value=_EmptyToolkit(),
    ):
        executor.execute_tick(agent["id"])

    assert personal.archived == [
        ("Help me make a study plan.", "Start with 25 minutes.", "managed_agent")
    ]


def test_scheduled_tick_without_user_message_is_not_archived(manager, event_bus):
    """A scheduled maintenance tick has no user conversation to archive."""
    from openjarvis.agents import AgentRegistry

    AgentRegistry.register_value("personal_memory_test_agent", _ReplyingAgent)
    personal = _PersonalMemorySpy()
    executor = _managed_executor(manager, event_bus, personal)
    agent = manager.create_agent(
        name="scheduled",
        agent_type="personal_memory_test_agent",
        config={"model": "test-model"},
    )

    with patch(
        "openjarvis.agents.executor.resolve_agent_tools",
        return_value=_EmptyToolkit(),
    ):
        executor.execute_tick(agent["id"])

    assert personal.archived == []
