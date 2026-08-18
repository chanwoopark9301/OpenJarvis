"""Tests for ``jarvis chat`` interactive REPL command."""

from __future__ import annotations

import sys
from unittest import mock
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openjarvis.agents._stubs import (
    AgentContext,
    AgentResult,
    BaseAgent,
    ToolUsingAgent,
)
from openjarvis.cli.chat_cmd import (
    _ensure_requested_agent_registered,
    _personal_memory_status,
    _read_input,
    _supports_prompt_builder_injection,
    chat,
)
from openjarvis.core.config import JarvisConfig
from openjarvis.core.events import Event, EventBus, EventType
from openjarvis.core.registry import AgentRegistry, ToolRegistry
from openjarvis.core.types import Message, Role, ToolCall, ToolResult
from openjarvis.memory.store import LocalFactStore
from openjarvis.tools._stubs import BaseTool, ToolSpec


def test_personal_memory_status_reports_disabled_response_injection():
    config = JarvisConfig()
    config.agent.context_from_memory = False

    assert _personal_memory_status(config, "shadow") == (
        "collecting; response injection disabled"
    )


def test_prompt_builder_injection_requires_capability_and_explicit_parameter():
    class KeywordOnlyReceiver:
        accepts_prompt_builder = True

        def __init__(self, *, prompt_builder=None):
            pass

    class PositionalOrKeywordReceiver:
        accepts_prompt_builder = True

        def __init__(self, prompt_builder=None):
            pass

    class PositionalOnlyReceiver:
        accepts_prompt_builder = True

        def __init__(self, prompt_builder=None, /):
            pass

    class VariadicPositionalReceiver:
        accepts_prompt_builder = True

        def __init__(self, *prompt_builder):
            pass

    class SwallowingWrapper(KeywordOnlyReceiver):
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class NoCapability:
        def __init__(self, *, prompt_builder=None):
            pass

    assert _supports_prompt_builder_injection(KeywordOnlyReceiver)
    assert _supports_prompt_builder_injection(PositionalOrKeywordReceiver)
    assert not _supports_prompt_builder_injection(PositionalOnlyReceiver)
    assert not _supports_prompt_builder_injection(VariadicPositionalReceiver)
    assert not _supports_prompt_builder_injection(SwallowingWrapper)
    assert not _supports_prompt_builder_injection(NoCapability)


def test_requested_proactive_registration_is_lazy_and_compatible():
    import openjarvis.agents.proactive_agent as proactive_module
    import openjarvis.tools.proactive_tools as proactive_tools_module

    proactive_tools = {
        "check_permission": proactive_tools_module.CheckPermissionTool,
        "queue_action": proactive_tools_module.QueueActionTool,
        "get_pending_actions": proactive_tools_module.GetPendingActionsTool,
        "record_decision": proactive_tools_module.RecordDecisionTool,
        "execute_pending_actions": proactive_tools_module.ExecutePendingActionsTool,
    }
    assert "openjarvis.agents.proactive_agent" in sys.modules
    assert "openjarvis.tools.proactive_tools" in sys.modules
    AgentRegistry.clear()
    ToolRegistry.clear()

    _ensure_requested_agent_registered("proactive")
    _ensure_requested_agent_registered("proactive")

    assert AgentRegistry.get("proactive") is proactive_module.ProactiveAgent
    assert {key: ToolRegistry.get(key) for key in proactive_tools} == proactive_tools


class _SimpleChatAgent(BaseAgent):
    agent_id = "simple_chat_agent"

    def run(self, input, context: AgentContext | None = None, **kwargs):
        return AgentResult(content="simple ok", turns=1)


class _DangerousChatTool(BaseTool):
    tool_id = "dangerous_chat"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="dangerous_chat",
            description="Confirmation-gated chat tool.",
            requires_confirmation=True,
        )

    def execute(self, **params) -> ToolResult:
        return ToolResult(
            tool_name="dangerous_chat",
            content="chat executed!",
            success=True,
        )


class _ToolChatAgent(ToolUsingAgent):
    agent_id = "tool_chat_agent"

    def run(self, input, context: AgentContext | None = None, **kwargs):
        result = self._executor.execute(
            ToolCall(id="chat", name="dangerous_chat", arguments="{}")
        )
        return AgentResult(content=result.content, tool_results=[result], turns=1)


class _ContextSpyChatAgent(BaseAgent):
    agent_id = "context_spy_chat_agent"
    contexts: list[AgentContext | None] = []

    def run(self, input, context: AgentContext | None = None, **kwargs):
        type(self).contexts.append(context)
        return AgentResult(content="첫 답변", turns=1)


def run_chat_with_spy_agent(input_text: str):
    _ContextSpyChatAgent.contexts = []
    engine = MagicMock()
    engine.engine_id = "mock"
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    AgentRegistry.register_value("context_spy_chat_agent", _ContextSpyChatAgent)

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
    ):
        result = CliRunner().invoke(
            chat,
            ["--agent", "context_spy_chat_agent", "--model", "test-model"],
            input=input_text,
        )

    return result, _ContextSpyChatAgent


def test_agent_receives_prior_user_and_assistant_turns() -> None:
    result, spy = run_chat_with_spy_agent("내 이름은 찬우야.\n내 이름이 뭐야?\n/quit\n")

    assert result.exit_code == 0
    second_context = spy.contexts[1].conversation.messages
    assert [
        (message.role.value, message.content)
        for message in second_context
        if message.role.value != "system"
    ] == [
        ("user", "내 이름은 찬우야."),
        ("assistant", "첫 답변"),
    ]


def test_agent_receives_pending_memory_as_user_only_dialogue() -> None:
    from openjarvis.memory.context_composer import ComposedMemoryContext

    _ContextSpyChatAgent.contexts = []
    engine = MagicMock()
    engine.engine_id = "ollama"
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.agent.context_from_memory = True
    AgentRegistry.register_value("context_spy_chat_agent", _ContextSpyChatAgent)
    personal = ComposedMemoryContext(
        constraints=(),
        current_states=(),
        schemas=(),
        episodes=(),
        raw_evidence=(),
        unresolved=(),
        user_overlay="",
        sections=(),
        recent_pending_user_messages=(
            "older pending user message",
            "newer pending user message",
        ),
    )

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("ollama", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch("openjarvis.memory.build_memory_service", return_value=None),
        patch("openjarvis.memory.build_personal_memory_service", return_value=None),
        patch("openjarvis.cli.ask._get_memory_backend", return_value=None),
        patch(
            "openjarvis.memory.context_composer.compose_configured_personal_context",
            return_value=personal,
        ),
    ):
        result = CliRunner().invoke(
            chat,
            ["--agent", "context_spy_chat_agent", "--model", "test-model"],
            input="current request\n/quit\n",
        )

    assert result.exit_code == 0
    assert [
        (message.role, message.content)
        for message in _ContextSpyChatAgent.contexts[0].conversation.messages
    ] == [
        (Role.USER, "older pending user message"),
        (Role.USER, "newer pending user message"),
    ]


def test_agent_deduplicates_pending_against_live_history_by_multiplicity() -> None:
    from openjarvis.memory.context_composer import ComposedMemoryContext

    _ContextSpyChatAgent.contexts = []
    engine = MagicMock()
    engine.engine_id = "ollama"
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.agent.context_from_memory = True
    AgentRegistry.register_value("context_spy_chat_agent", _ContextSpyChatAgent)
    personal = ComposedMemoryContext(
        constraints=(),
        current_states=(),
        schemas=(),
        episodes=(),
        raw_evidence=(),
        unresolved=(),
        user_overlay="",
        sections=(),
        recent_pending_user_messages=("same message", "same message", "other pending"),
    )

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("ollama", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch("openjarvis.memory.build_memory_service", return_value=None),
        patch("openjarvis.memory.build_personal_memory_service", return_value=None),
        patch("openjarvis.cli.ask._get_memory_backend", return_value=None),
        patch(
            "openjarvis.memory.context_composer.compose_configured_personal_context",
            return_value=personal,
        ),
    ):
        result = CliRunner().invoke(
            chat,
            ["--agent", "context_spy_chat_agent", "--model", "test-model"],
            input="same message\nnext request\n/quit\n",
        )

    assert result.exit_code == 0
    second = _ContextSpyChatAgent.contexts[1].conversation.messages
    pending = [
        message.content
        for message in second
        if message.metadata.get("personal_memory_context") == "recent_dialogue"
    ]
    assert pending == ["same message", "other pending"]


def test_direct_chat_deduplicates_pending_against_live_history() -> None:
    from openjarvis.memory.context_composer import ComposedMemoryContext

    engine = MagicMock()
    engine.engine_id = "ollama"
    engine.generate.return_value = {"content": "reply"}
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.agent.context_from_memory = True
    personal = ComposedMemoryContext(
        constraints=(),
        current_states=(),
        schemas=(),
        episodes=(),
        raw_evidence=(),
        unresolved=(),
        user_overlay="",
        sections=(),
        recent_pending_user_messages=("same message", "other pending"),
    )
    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("ollama", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch("openjarvis.memory.build_memory_service", return_value=None),
        patch("openjarvis.memory.build_personal_memory_service", return_value=None),
        patch("openjarvis.cli.ask._get_memory_backend", return_value=None),
        patch(
            "openjarvis.tools.storage.context.inject_context",
            side_effect=lambda _q, messages, *_a, **_kw: messages,
        ),
        patch(
            "openjarvis.memory.context_composer.compose_configured_personal_context",
            return_value=personal,
        ),
    ):
        result = CliRunner().invoke(
            chat,
            ["--model", "test-model"],
            input="same message\nnext request\n/quit\n",
        )

    assert result.exit_code == 0
    second_messages = engine.generate.call_args_list[1].args[0]
    pending = [
        message.content
        for message in second_messages
        if message.metadata.get("personal_memory_context") == "recent_dialogue"
    ]
    assert pending == ["other pending"]


class TestChatCommand:
    """Test the Click command definition and help output."""

    def test_command_exists(self) -> None:
        result = CliRunner().invoke(chat, ["--help"])
        assert result.exit_code == 0
        assert "interactive" in result.output.lower() or "chat" in result.output.lower()

    def test_options(self) -> None:
        result = CliRunner().invoke(chat, ["--help"])
        assert result.exit_code == 0
        assert "--engine" in result.output
        assert "--model" in result.output
        assert "--agent" in result.output
        assert "--tools" in result.output
        assert "--system" in result.output

    def test_slash_commands_listed(self) -> None:
        result = CliRunner().invoke(chat, ["--help"])
        assert result.exit_code == 0
        assert "/quit" in result.output


class TestReadInput:
    """Test the _read_input helper function."""

    def test_read_input_eof(self) -> None:
        with mock.patch("builtins.input", side_effect=EOFError):
            assert _read_input() is None

    def test_read_input_keyboard_interrupt(self) -> None:
        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            assert _read_input() is None

    def test_read_input_normal(self) -> None:
        with mock.patch("builtins.input", return_value="hello"):
            assert _read_input() == "hello"


class TestChatAgents:
    def test_registered_proactive_agent_obeys_external_prompt_boundary(
        self,
        tmp_path,
    ) -> None:
        from openjarvis.agents.proactive_agent import ProactiveAgent
        from openjarvis.memory.archive import PersonalMemoryArchive

        soul = tmp_path / "SOUL.md"
        user = tmp_path / "USER.md"
        memory = tmp_path / "MEMORY.md"
        soul.write_text("SOUL MARKER", encoding="utf-8")
        user.write_text("PRIVATE USER MARKER", encoding="utf-8")
        memory.write_text("PRIVATE MEMORY MARKER", encoding="utf-8")
        archive = PersonalMemoryArchive(tmp_path / "personal.db")
        archive.set_metadata("canonical_profile_active", "1")
        engine = MagicMock()
        engine.engine_id = "openai"
        engine.generate.return_value = {"content": "[]"}
        config = JarvisConfig()
        config.intelligence.default_model = "test-model"
        config.personal_memory.enabled = False
        config.personal_memory.archive_path = str(archive.path)
        config.agent.context_from_memory = False
        config.memory_files.soul_path = str(soul)
        config.memory_files.user_path = str(user)
        config.memory_files.memory_path = str(memory)

        def run_with_real_prompt(self, input, context=None, **kwargs):
            self._engine.generate(
                [
                    Message(role=Role.SYSTEM, content=self._build_system_prompt()),
                    Message(role=Role.USER, content=input),
                ],
                model=self._model,
            )
            return AgentResult(content="safe reply", turns=1)

        with (
            patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
            patch("openjarvis.engine.get_engine", return_value=("openai", engine)),
            patch("openjarvis.intelligence.register_builtin_models"),
            patch.object(ProactiveAgent, "run", run_with_real_prompt),
            patch("openjarvis.memory.build_memory_service", return_value=None),
            patch("openjarvis.memory.build_personal_memory_service", return_value=None),
        ):
            result = CliRunner().invoke(
                chat,
                ["--agent", "proactive", "--model", "test-model"],
                input="hello\n/quit\n",
            )

        assert result.exit_code == 0
        payload = "\n".join(
            message.content for message in engine.generate.call_args.args[0]
        )
        assert "SOUL MARKER" in payload
        assert "PRIVATE USER MARKER" not in payload
        assert "PRIVATE MEMORY MARKER" not in payload

    def test_canonical_profile_keeps_only_soul_for_external_provider(
        self,
        tmp_path,
    ) -> None:
        from openjarvis.memory.archive import PersonalMemoryArchive

        soul = tmp_path / "SOUL.md"
        user = tmp_path / "USER.md"
        memory = tmp_path / "MEMORY.md"
        facts = tmp_path / "facts.jsonl"
        soul.write_text("SOUL MARKER", encoding="utf-8")
        user.write_text("PRIVATE USER MARKER", encoding="utf-8")
        memory.write_text("PRIVATE MEMORY MARKER", encoding="utf-8")
        LocalFactStore(facts).add("PRIVATE LEGACY FACT MARKER")
        archive = PersonalMemoryArchive(tmp_path / "personal.db")
        archive.set_metadata("canonical_profile_active", "1")

        engine = MagicMock()
        engine.engine_id = "openai"
        engine.generate.return_value = {"content": "safe reply"}
        config = JarvisConfig()
        config.intelligence.default_model = "test-model"
        config.personal_memory.archive_path = str(archive.path)
        config.memory.enabled = True
        config.memory.facts_path = str(facts)
        config.memory_files.soul_path = str(soul)
        config.memory_files.user_path = str(user)
        config.memory_files.memory_path = str(memory)

        with (
            patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
            patch("openjarvis.engine.get_engine", return_value=("openai", engine)),
            patch("openjarvis.intelligence.register_builtin_models"),
            patch("openjarvis.memory.build_memory_service", return_value=None),
            patch("openjarvis.memory.build_personal_memory_service", return_value=None),
            patch("openjarvis.cli.ask._get_memory_backend", return_value=None),
            patch(
                "openjarvis.memory.context_composer.compose_configured_personal_context"
            ) as compose_personal,
        ):
            result = CliRunner().invoke(
                chat,
                ["--model", "test-model"],
                input="hello\n/quit\n",
            )

        assert result.exit_code == 0
        messages = engine.generate.call_args.args[0]
        rendered = "\n".join(message.content for message in messages)
        assert "SOUL MARKER" in rendered
        assert "PRIVATE USER MARKER" not in rendered
        assert "PRIVATE MEMORY MARKER" not in rendered
        assert "PRIVATE LEGACY FACT MARKER" not in rendered
        compose_personal.assert_not_called()

    def test_local_direct_chat_injects_auto_memory_facts(self, tmp_path) -> None:
        facts_path = tmp_path / "facts.jsonl"
        LocalFactStore(facts_path).add(
            "The user's favorite color is blue",
            source="auto",
        )

        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {"content": "Blue."}
        config = JarvisConfig()
        config.intelligence.default_model = "test-model"
        config.memory.enabled = True
        config.memory.facts_path = str(facts_path)
        config.agent.context_from_memory = True

        with (
            patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
            patch("openjarvis.engine.get_engine", return_value=("ollama", engine)),
            patch("openjarvis.intelligence.register_builtin_models"),
            patch("openjarvis.memory.build_memory_service", return_value=None),
            patch("openjarvis.cli.ask._get_memory_backend", return_value=None),
        ):
            result = CliRunner().invoke(
                chat,
                ["--model", "test-model"],
                input="What is my favorite color?\n/quit\n",
            )

        assert result.exit_code == 0
        messages = engine.generate.call_args.args[0]
        assert messages[0].role.value == "system"
        assert "favorite color is blue" in messages[0].content

    def test_chat_generation_survives_fact_store_failure(self) -> None:
        class _FailingMemoryService:
            def start(self) -> None:
                pass

            def stop(self, timeout: float = 2.0) -> None:
                pass

            def list_facts(self):
                raise OSError("fact store unavailable")

        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {"content": "Still working."}
        config = JarvisConfig()
        config.intelligence.default_model = "test-model"
        config.memory.enabled = True
        config.agent.context_from_memory = True

        with (
            patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
            patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
            patch("openjarvis.intelligence.register_builtin_models"),
            patch(
                "openjarvis.memory.build_memory_service",
                return_value=_FailingMemoryService(),
            ),
            patch("openjarvis.cli.ask._get_memory_backend", return_value=None),
        ):
            result = CliRunner().invoke(
                chat,
                ["--model", "test-model"],
                input="hello\n/quit\n",
            )

        assert result.exit_code == 0
        assert "Still working." in result.output
        engine.generate.assert_called_once()

    def test_simple_agent_does_not_receive_tool_only_kwargs(self) -> None:
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {"content": "engine fallback"}
        config = JarvisConfig()
        config.intelligence.default_model = "test-model"

        AgentRegistry.register_value("simple_chat_agent", _SimpleChatAgent)

        with (
            patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
            patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
            patch("openjarvis.intelligence.register_builtin_models"),
        ):
            result = CliRunner().invoke(
                chat,
                ["--agent", "simple_chat_agent", "--model", "test-model"],
                input="hello\n/quit\n",
            )

        assert result.exit_code == 0
        assert "simple ok" in result.output
        assert "failed" not in result.output.lower()

    def test_memory_service_started_fed_and_stopped(self) -> None:
        """The REPL starts memory, publishes each turn, and stops it."""

        class _SpyMemoryService:
            def __init__(self, bus: EventBus) -> None:
                self.bus = bus
                self.started = False
                self.stopped = False
                self.submissions: list[tuple[str, str]] = []

            def start(self) -> None:
                self.started = True
                self.bus.subscribe(
                    EventType.CHAT_EXCHANGE_COMPLETED,
                    self._on_completed_exchange,
                )

            def _on_completed_exchange(self, event: Event) -> None:
                self.submissions.append(
                    (
                        event.data["user_text"],
                        event.data.get("assistant_text", ""),
                    )
                )

            def stop(self, timeout: float = 2.0) -> None:
                self.stopped = True
                self.bus.unsubscribe(
                    EventType.CHAT_EXCHANGE_COMPLETED,
                    self._on_completed_exchange,
                )

        spy: _SpyMemoryService | None = None

        def _build_memory_service(*args, event_bus: EventBus | None = None, **kwargs):
            nonlocal spy
            assert event_bus is not None
            spy = _SpyMemoryService(event_bus)
            return spy

        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {"content": "engine fallback"}
        config = JarvisConfig()
        config.intelligence.default_model = "test-model"

        AgentRegistry.register_value("simple_chat_agent", _SimpleChatAgent)

        with (
            patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
            patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
            patch("openjarvis.intelligence.register_builtin_models"),
            patch(
                "openjarvis.memory.build_memory_service",
                side_effect=_build_memory_service,
            ),
        ):
            result = CliRunner().invoke(
                chat,
                ["--agent", "simple_chat_agent", "--model", "test-model"],
                input="hello\n/quit\n",
            )

        assert result.exit_code == 0
        assert spy is not None
        assert spy.started is True
        assert spy.stopped is True
        assert spy.submissions == [("hello", "simple ok")]

    def test_chat_archives_completed_turn_with_personal_memory_service(self) -> None:
        """The REPL must archive a finished turn before it announces completion."""

        class _SpyPersonalMemoryService:
            def __init__(self) -> None:
                self.started = False
                self.stopped = False
                self.archived: list[tuple[str, str, str]] = []

            def start(self) -> None:
                self.started = True

            def stop(self, timeout: float = 2.0) -> None:
                self.stopped = True

            def archive_exchange(self, *, user_text, assistant_text, source, **kwargs):
                self.archived.append((user_text, assistant_text, source))
                return type("Exchange", (), {"id": "personal-exchange-1"})()

        spy: _SpyPersonalMemoryService | None = None

        def _build_personal_memory_service(*args, **kwargs):
            nonlocal spy
            spy = _SpyPersonalMemoryService()
            return spy

        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {"content": "saved reply"}
        config = JarvisConfig()
        config.intelligence.default_model = "test-model"
        config.personal_memory.enabled = True
        config.personal_memory.mode = "shadow"

        with (
            patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
            patch("openjarvis.engine.get_engine", return_value=("ollama", engine)),
            patch("openjarvis.intelligence.register_builtin_models"),
            patch("openjarvis.memory.build_memory_service", return_value=None),
            patch(
                "openjarvis.memory.build_personal_memory_service",
                side_effect=_build_personal_memory_service,
            ),
        ):
            result = CliRunner().invoke(
                chat,
                ["--model", "test-model"],
                input="remember this\n/quit\n",
            )

        assert result.exit_code == 0
        assert spy is not None
        assert spy.started is True
        assert spy.stopped is True
        assert spy.archived == [("remember this", "saved reply", "cli.chat")]
        assert (
            "Personal memory: collecting; direct rule injection enabled"
            in result.output
        )
        assert "Personal memory: active" not in result.output

    def test_tool_agent_uses_legacy_agent_tools_and_prompts_confirmation(self) -> None:
        engine = MagicMock()
        engine.engine_id = "mock"
        config = JarvisConfig()
        config.intelligence.default_model = "test-model"
        config.agent.tools = "dangerous_chat"
        config.agent.max_turns = 3

        AgentRegistry.register_value("tool_chat_agent", _ToolChatAgent)
        ToolRegistry.register_value("dangerous_chat", _DangerousChatTool)

        with (
            patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
            patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
            patch("openjarvis.intelligence.register_builtin_models"),
        ):
            result = CliRunner().invoke(
                chat,
                ["--agent", "tool_chat_agent", "--model", "test-model"],
                input="run tool\ny\n/quit\n",
            )

        assert result.exit_code == 0
        assert "Confirm:" in result.output
        assert "chat executed!" in result.output
