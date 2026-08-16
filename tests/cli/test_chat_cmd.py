"""Tests for ``jarvis chat`` interactive REPL command."""

from __future__ import annotations

from unittest import mock
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openjarvis.agents._stubs import (
    AgentContext,
    AgentResult,
    BaseAgent,
    ToolUsingAgent,
)
from openjarvis.cli._auto_search import AutoSearchResult
from openjarvis.cli._search_evidence import EvidenceSource
from openjarvis.cli.chat_cmd import _personal_memory_status, _read_input, chat
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


class _FixedAutoSearchPreflight:
    def __init__(self, result: AutoSearchResult):
        self.result = result

    def prepare(self, user_text: str) -> AutoSearchResult:
        return self.result


def _auto_search_chat_config() -> JarvisConfig:
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.agent.default_agent = "simple"
    config.agent.context_from_memory = False
    config.memory.enabled = False
    config.personal_memory.enabled = False
    return config


def test_auto_search_non_triggered_chat_keeps_single_generation_path():
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {"content": "안녕하세요."}
    config = _auto_search_chat_config()
    preflight = _FixedAutoSearchPreflight(
        AutoSearchResult(triggered=False, success=False)
    )

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch(
            "openjarvis.cli.chat_cmd._build_auto_search_preflight",
            return_value=preflight,
            create=True,
        ),
    ):
        result = CliRunner().invoke(
            chat,
            ["--model", "test-model"],
            input="안녕?\n/quit\n",
        )

    assert result.exit_code == 0
    assert "안녕하세요." in result.output
    assert engine.generate.call_count == 1


def test_auto_search_context_reaches_agent_and_sources_are_appended():
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {
        "content": (
            '{"items":[{"time":"10:30","title":"테미오래 둘러보기",'
            '"detail":"천천히 둘러봐.","venue":"테미오래",'
            '"address":"","hours":"","source_ids":["S1"],'
            '"status":"confirmed","check_before_visit":""}]}'
        )
    }
    config = _auto_search_chat_config()
    preflight = _FixedAutoSearchPreflight(
        AutoSearchResult(
            triggered=True,
            success=True,
            queries=("대전 테미오래 운영시간",),
            sources=("https://temiorae.com/",),
            context="UNTRUSTED WEB SEARCH RESULTS\nverified opening hours",
            evidence=(
                EvidenceSource(
                    "S1",
                    "R1",
                    "테미오래 공식 누리집",
                    "https://temiorae.com/",
                    "대전 테미오래 관람 안내",
                    "official",
                ),
            ),
        )
    )

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch(
            "openjarvis.cli.chat_cmd._build_auto_search_preflight",
            return_value=preflight,
            create=True,
        ),
    ):
        result = CliRunner().invoke(
            chat,
            ["--model", "test-model"],
            input="인터넷에서 테미오래 운영시간을 찾아줘\n/quit\n",
        )

    assert result.exit_code == 0
    messages = engine.generate.call_args.args[0]
    assert any("UNTRUSTED WEB SEARCH RESULTS" in msg.content for msg in messages)
    assert messages[-1].content == "인터넷에서 테미오래 운영시간을 찾아줘"
    assert "https://temiorae.com/" in result.output


def test_auto_search_keeps_supported_item_and_sanitizes_unsupported_item():
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {
        "content": (
            '{"items":['
            '{"time":"10:30","title":"테미오래 둘러보기",'
            '"detail":"천천히 둘러봐.","venue":"테미오래",'
            '"address":"","hours":"","source_ids":["S1"],'
            '"status":"confirmed","check_before_visit":""},'
            '{"time":"12:00","title":"없는식당 점심",'
            '"detail":"없는식당에서 점심을 먹어.","venue":"없는식당",'
            '"address":"없는 주소","hours":"오전 9시",'
            '"source_ids":["S2"],"status":"confirmed",'
            '"check_before_visit":""}]}'
        )
    }
    config = _auto_search_chat_config()
    preflight = _FixedAutoSearchPreflight(
        AutoSearchResult(
            triggered=True,
            success=True,
            queries=("대전 테미오래 운영시간",),
            sources=(
                "https://temiorae.com/",
                "https://example.com/food",
            ),
            context="UNTRUSTED WEB SEARCH RESULTS\nverified opening hours",
            evidence=(
                EvidenceSource(
                    "S1",
                    "R1",
                    "테미오래 공식 누리집",
                    "https://temiorae.com/",
                    "대전 테미오래 관람 안내",
                    "official",
                ),
                EvidenceSource(
                    "S2",
                    "R2",
                    "대전 한식 식당 모음",
                    "https://example.com/food",
                    "대전 한식 식당 안내",
                    "listing",
                ),
            ),
        )
    )

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch(
            "openjarvis.cli.chat_cmd._build_auto_search_preflight",
            return_value=preflight,
            create=True,
        ),
    ):
        result = CliRunner().invoke(
            chat,
            ["--model", "test-model"],
            input="인터넷에서 테미오래 운영시간을 찾아줘\n/quit\n",
        )

    normalized_output = " ".join(result.output.split())
    assert result.exit_code == 0
    assert "테미오래 둘러보기" in normalized_output
    assert "없는식당" not in normalized_output
    assert "방문 전 확인" in normalized_output
    assert "출처와 직접 연결해 확인하지 못했어" not in normalized_output
    assert "https://temiorae.com/" in result.output


def test_auto_search_repairs_malformed_itinerary_once_locally():
    repaired = (
        '{"items":[{"time":"10:30","title":"테미오래 둘러보기",'
        '"detail":"천천히 둘러봐.","venue":"테미오래","address":"",'
        '"hours":"","source_ids":["S1"],"status":"confirmed",'
        '"check_before_visit":""}]}'
    )
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [
        {"content": "일정은 다음과 같습니다."},
        {"content": repaired},
    ]
    config = _auto_search_chat_config()
    preflight = _FixedAutoSearchPreflight(
        AutoSearchResult(
            triggered=True,
            success=True,
            queries=("대전 테미오래 운영시간",),
            sources=("https://temiorae.com/",),
            context="[S1] 테미오래 공식 자료",
            evidence=(
                EvidenceSource(
                    "S1",
                    "R1",
                    "테미오래 공식 누리집",
                    "https://temiorae.com/",
                    "대전 테미오래 관람 안내",
                    "official",
                ),
            ),
        )
    )

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch(
            "openjarvis.cli.chat_cmd._build_auto_search_preflight",
            return_value=preflight,
        ),
    ):
        result = CliRunner().invoke(
            chat,
            ["--model", "test-model"],
            input="테미오래 운영시간을 인터넷에서 찾아줘\n/quit\n",
        )

    assert result.exit_code == 0
    assert "테미오래 둘러보기" in result.output
    assert engine.generate.call_count == 2
    repair_messages = engine.generate.call_args_list[1].args[0]
    repair_text = "\n".join(message.content for message in repair_messages)
    assert "일정은 다음과 같습니다." in repair_text
    assert "S1" in repair_text


def test_auto_search_uses_safe_notice_after_one_failed_repair():
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [
        {"content": "첫 번째 잘못된 형식"},
        {"content": "두 번째 잘못된 형식"},
    ]
    config = _auto_search_chat_config()
    preflight = _FixedAutoSearchPreflight(
        AutoSearchResult(
            triggered=True,
            success=True,
            queries=("대전 테미오래 운영시간",),
            sources=("https://temiorae.com/",),
            context="[S1] 테미오래 공식 자료",
            evidence=(
                EvidenceSource(
                    "S1",
                    "R1",
                    "테미오래 공식 누리집",
                    "https://temiorae.com/",
                    "대전 테미오래 관람 안내",
                    "official",
                ),
            ),
        )
    )

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch(
            "openjarvis.cli.chat_cmd._build_auto_search_preflight",
            return_value=preflight,
        ),
    ):
        result = CliRunner().invoke(
            chat,
            ["--model", "test-model"],
            input="테미오래 운영시간을 인터넷에서 찾아줘\n/quit\n",
        )

    normalized = " ".join(result.output.split())
    assert result.exit_code == 0
    assert engine.generate.call_count == 2
    assert "출처와 직접 연결해 확인하지 못했어" in normalized


def test_auto_search_failure_skips_final_model_and_archives_failure_reply():
    engine = MagicMock()
    engine.engine_id = "mock"
    config = _auto_search_chat_config()
    failure = (
        "인터넷에서 필요한 정보를 확인하지 못했어. 확인되지 않은 장소나 운영 "
        "정보를 지어내지는 않을게. 잠시 뒤 다시 검색해 줘."
    )
    preflight = _FixedAutoSearchPreflight(
        AutoSearchResult(
            triggered=True,
            success=False,
            error=failure,
        )
    )

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch(
            "openjarvis.cli.chat_cmd._build_auto_search_preflight",
            return_value=preflight,
            create=True,
        ),
        patch(
            "openjarvis.cli.chat_cmd.record_and_publish_completed_exchange"
        ) as record_exchange,
    ):
        result = CliRunner().invoke(
            chat,
            ["--model", "test-model"],
            input="인터넷에서 대전 맛집을 찾아줘\n/quit\n",
        )

    assert result.exit_code == 0
    assert " ".join(failure.split()) in " ".join(result.output.split())
    engine.generate.assert_not_called()
    archived = record_exchange.call_args.args
    assert archived[2:] == (
        "인터넷에서 대전 맛집을 찾아줘",
        failure,
    )


def test_auto_search_and_personal_memory_context_both_reach_agent():
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {
        "content": (
            '{"items":[{"time":"","title":"대전 장소 살펴보기",'
            '"detail":"확인된 자료부터 살펴봐.","venue":"",'
            '"address":"","hours":"","source_ids":["S1"],'
            '"status":"partial","check_before_visit":""}]}'
        )
    }
    config = _auto_search_chat_config()
    config.agent.context_from_memory = True
    preflight = _FixedAutoSearchPreflight(
        AutoSearchResult(
            triggered=True,
            success=True,
            sources=("https://example.com/source",),
            context="UNTRUSTED WEB SEARCH RESULTS\npublic evidence",
            evidence=(
                EvidenceSource(
                    "S1",
                    "R1",
                    "대전 장소 자료",
                    "https://example.com/source",
                    "대전 방문 정보",
                    "other",
                ),
            ),
        )
    )
    memory_message = Message(
        role=Role.SYSTEM,
        content="PERSONAL MEMORY CONTEXT\ndirect user rule",
        metadata={"memory_context": True},
    )

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch("openjarvis.memory.build_memory_service", return_value=None),
        patch("openjarvis.cli.ask._get_memory_backend", return_value=None),
        patch(
            "openjarvis.tools.storage.context.inject_context",
            return_value=[memory_message],
        ),
        patch(
            "openjarvis.cli.chat_cmd._build_auto_search_preflight",
            return_value=preflight,
            create=True,
        ),
    ):
        result = CliRunner().invoke(
            chat,
            ["--model", "test-model"],
            input="웹에서 대전 장소를 찾아줘\n/quit\n",
        )

    assert result.exit_code == 0
    messages = engine.generate.call_args.args[0]
    combined = "\n".join(message.content for message in messages)
    assert "PERSONAL MEMORY CONTEXT" in combined
    assert "UNTRUSTED WEB SEARCH RESULTS" in combined


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
    def test_direct_chat_injects_auto_memory_facts(self, tmp_path) -> None:
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
            patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
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
