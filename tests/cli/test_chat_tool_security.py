"""Security-boundary regressions for model-led chat tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from openjarvis.agents._stubs import (
    AgentContext,
    AgentResult,
    BaseAgent,
    ToolUsingAgent,
)
from openjarvis.cli.chat_cmd import chat
from openjarvis.core.config import CapabilitiesConfig, JarvisConfig, SecurityConfig
from openjarvis.core.registry import AgentRegistry, ToolRegistry
from openjarvis.core.types import Message, Role, ToolCall, ToolResult
from openjarvis.memory.context_composer import ComposedMemoryContext, ContextSection
from openjarvis.tools._stubs import BaseTool, ToolSpec


class _SecurityChatAgent(BaseAgent):
    """Plain chat agent upgraded to the managed function-calling loop."""

    agent_id = "security_chat_agent"
    supports_managed_tool_fallback = True

    def run(
        self,
        input: str,
        context: AgentContext | None = None,
        **kwargs,
    ) -> AgentResult:
        raise AssertionError("the managed tool runtime must own this turn")


class _DirectSecurityToolAgent(ToolUsingAgent):
    """Tool agent whose constructor intentionally has no security parameters."""

    agent_id = "direct_security_tool_agent"

    def __init__(
        self,
        engine,
        model,
        *,
        tools=None,
        bus=None,
        max_turns=None,
        interactive=False,
        confirm_callback=None,
    ) -> None:
        super().__init__(
            engine,
            model,
            tools=tools,
            bus=bus,
            max_turns=max_turns,
            interactive=interactive,
            confirm_callback=confirm_callback,
        )

    def run(
        self,
        input: str,
        context: AgentContext | None = None,
        **kwargs,
    ) -> AgentResult:
        messages = self._build_messages(input, context)
        first = self._generate(
            messages,
            tools=self._executor.get_openai_tools(),
        )
        raw_call = first["tool_calls"][0]
        tool_call = ToolCall(
            id=raw_call["id"],
            name=raw_call["name"],
            arguments=raw_call["arguments"],
        )
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=first.get("content", ""),
                tool_calls=[tool_call],
            )
        )
        tool_result = self._executor.execute(tool_call)
        messages.append(
            Message(
                role=Role.TOOL,
                content=tool_result.content,
                tool_call_id=tool_call.id,
                name=tool_call.name,
            )
        )
        final = self._generate(messages)
        return AgentResult(
            content=final.get("content", ""),
            tool_results=[tool_result],
            turns=2,
        )


class _ExternalSearchTool(BaseTool):
    tool_id = "security_external_search"
    is_local = False

    def __init__(self) -> None:
        self.queries: list[str] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description="Search public weather data.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
                "additionalProperties": False,
            },
            required_capabilities=["network:fetch"],
        )

    def execute(self, **params) -> ToolResult:
        self.queries.append(params["query"])
        return ToolResult(
            tool_name=self.tool_id,
            content="과천시 현재 기온 24도",
        )


@dataclass
class _ScanResult:
    findings: list[object] = field(default_factory=list)


class _LiteralPrivacyScanner:
    """Controlled scanner dependency; the real BoundaryGuard remains under test."""

    def __init__(self, sensitive_fragments: tuple[str, ...]) -> None:
        self._sensitive_fragments = sensitive_fragments

    def scan(self, text: str) -> _ScanResult:
        detected = any(fragment in text for fragment in self._sensitive_fragments)
        return _ScanResult([object()] if detected else [])

    def redact(self, text: str) -> str:
        for fragment in self._sensitive_fragments:
            text = text.replace(fragment, "[REDACTED]")
        return text


class _DenyNetworkPolicy:
    def check(self, agent_id: str, capability: str, resource: str = "") -> bool:
        return capability != "network:fetch"


def _run_secured_chat(
    tmp_path,
    monkeypatch,
    *,
    model_arguments: dict[str, object],
    sensitive_fragments: tuple[str, ...] = (),
    capabilities_enabled: bool = False,
    agent_cls: type[BaseAgent] = _SecurityChatAgent,
):
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "search-1",
                    "name": _ExternalSearchTool.tool_id,
                    "arguments": json.dumps(model_arguments, ensure_ascii=False),
                }
            ],
        },
        {"content": "도구 결과를 확인했어."},
    ]

    tool = _ExternalSearchTool()
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.agent.max_turns = 3
    config.agent.context_from_memory = False
    config.agent.tools = _ExternalSearchTool.tool_id
    config.tools.enabled = _ExternalSearchTool.tool_id
    config.memory.enabled = False
    config.personal_memory.enabled = False
    config.security = SecurityConfig(
        enabled=True,
        scan_input=False,
        scan_output=False,
        mode="block",
        secret_scanner=True,
        pii_scanner=True,
        audit_log_path=str(tmp_path / "audit.db"),
        capabilities=CapabilitiesConfig(enabled=capabilities_enabled),
    )

    scanner = _LiteralPrivacyScanner(sensitive_fragments)
    monkeypatch.setattr("openjarvis.security.SecretScanner", lambda: scanner)
    monkeypatch.setattr("openjarvis.security.PIIScanner", lambda: scanner)
    if capabilities_enabled:
        monkeypatch.setattr(
            "openjarvis.security.capabilities.CapabilityPolicy",
            lambda **kwargs: _DenyNetworkPolicy(),
        )

    AgentRegistry.register_value(agent_cls.agent_id, agent_cls)
    ToolRegistry.register_value(_ExternalSearchTool.tool_id, tool)

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
    ):
        result = CliRunner().invoke(
            chat,
            ["--agent", agent_cls.agent_id, "--model", "test-model"],
            input="오늘 과천시 날씨를 찾아줘.\n/quit\n",
        )

    second_messages = engine.generate.call_args_list[1].args[0]
    tool_feedback = [message.content for message in second_messages if message.name]
    return result, tool, tool_feedback


@pytest.mark.parametrize(
    ("leaking_query", "sensitive_fragment"),
    [
        ("과천 날씨 user@example.com", "user@example.com"),
        ("과천 날씨 AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    ],
)
def test_sensitive_model_arguments_never_execute_external_tool(
    tmp_path,
    monkeypatch,
    leaking_query,
    sensitive_fragment,
):
    result, tool, tool_feedback = _run_secured_chat(
        tmp_path,
        monkeypatch,
        model_arguments={"query": leaking_query},
        sensitive_fragments=(sensitive_fragment,),
    )

    assert result.exit_code == 0
    assert tool.queries == []
    assert any("Security block" in content for content in tool_feedback)


def test_capability_denial_prevents_external_tool_execution(tmp_path, monkeypatch):
    result, tool, tool_feedback = _run_secured_chat(
        tmp_path,
        monkeypatch,
        model_arguments={"query": "과천시 오늘 날씨"},
        capabilities_enabled=True,
    )

    assert result.exit_code == 0
    assert tool.queries == []
    assert any(
        "network:fetch" in content and "denied" in content for content in tool_feedback
    )


def test_valid_public_weather_query_executes_external_tool(tmp_path, monkeypatch):
    result, tool, tool_feedback = _run_secured_chat(
        tmp_path,
        monkeypatch,
        model_arguments={"query": "과천시 오늘 날씨"},
    )

    assert result.exit_code == 0
    assert tool.queries == ["과천시 오늘 날씨"]
    assert tool_feedback == ["과천시 현재 기온 24도"]


def test_schema_invalid_model_arguments_are_returned_without_execution(
    tmp_path,
    monkeypatch,
):
    result, tool, tool_feedback = _run_secured_chat(
        tmp_path,
        monkeypatch,
        model_arguments={"query": 24},
    )

    assert result.exit_code == 0
    assert tool.queries == []
    assert any(
        "Tool argument schema error" in content and "query" in content
        for content in tool_feedback
    )


def test_chat_secures_tool_agent_without_security_constructor_parameters(
    tmp_path,
    monkeypatch,
):
    result, tool, tool_feedback = _run_secured_chat(
        tmp_path,
        monkeypatch,
        model_arguments={"query": "과천 날씨 AKIAIOSFODNN7EXAMPLE"},
        sensitive_fragments=("AKIAIOSFODNN7EXAMPLE",),
        agent_cls=_DirectSecurityToolAgent,
    )

    assert result.exit_code == 0
    assert tool.queries == []
    assert any("Security block" in content for content in tool_feedback)


def _personal_context_with_relationship(
    *,
    recent_pending_user_messages: tuple[str, ...] = (),
) -> ComposedMemoryContext:
    relationship = "민지는 내 여자친구다"
    return ComposedMemoryContext(
        constraints=(),
        current_states=(),
        schemas=(relationship,),
        episodes=(),
        raw_evidence=(),
        unresolved=(),
        user_overlay="",
        sections=(ContextSection("schemas", (relationship,)),),
        recent_pending_user_messages=recent_pending_user_messages,
    )


def _run_chat_with_personal_context(
    tmp_path,
    *,
    model_query: str,
    profile_text: str = "",
    pending_messages: tuple[str, ...] = (),
):
    engine = MagicMock()
    engine.engine_id = "ollama"
    engine.generate.side_effect = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "search-1",
                    "name": _ExternalSearchTool.tool_id,
                    "arguments": json.dumps(
                        {"query": model_query},
                        ensure_ascii=False,
                    ),
                }
            ],
        },
        {"content": "도구 결과를 확인했어."},
    ]

    tool = _ExternalSearchTool()
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.agent.max_turns = 3
    config.agent.context_from_memory = True
    config.agent.tools = _ExternalSearchTool.tool_id
    config.tools.enabled = _ExternalSearchTool.tool_id
    config.memory.enabled = False
    config.memory_files.soul_path = ""
    config.memory_files.memory_path = ""
    if profile_text:
        profile_path = tmp_path / "USER.md"
        profile_path.write_text(profile_text, encoding="utf-8")
        config.memory_files.user_path = str(profile_path)
    else:
        config.memory_files.user_path = ""
    config.personal_memory.enabled = True
    config.personal_memory.mode = "shadow"
    config.personal_memory.archive_path = str(tmp_path / "personal-memory.db")
    config.security = SecurityConfig(
        enabled=True,
        scan_input=False,
        scan_output=False,
        mode="redact",
        secret_scanner=False,
        pii_scanner=False,
        audit_log_path=str(tmp_path / "audit.db"),
    )

    AgentRegistry.register_value("security_chat_agent", _SecurityChatAgent)
    ToolRegistry.register_value(_ExternalSearchTool.tool_id, tool)
    personal_context = _personal_context_with_relationship(
        recent_pending_user_messages=pending_messages,
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
            return_value=personal_context,
        ),
    ):
        result = CliRunner().invoke(
            chat,
            ["--agent", "security_chat_agent", "--model", "test-model"],
            input="오늘 과천시 날씨를 찾아줘.\n/quit\n",
        )

    first_messages = engine.generate.call_args_list[0].args[0]
    assert any("민지는 내 여자친구다" in message.text for message in first_messages)
    if profile_text:
        assert any(profile_text in message.text for message in first_messages)
    second_messages = engine.generate.call_args_list[1].args[0]
    tool_feedback = [message.content for message in second_messages if message.name]
    return result, tool, tool_feedback


def test_injected_private_memory_copied_to_external_args_is_blocked_by_default(
    tmp_path,
):
    result, tool, tool_feedback = _run_chat_with_personal_context(
        tmp_path,
        model_query="과천 날씨와 민지는 내 여자친구다 관련 정보를 검색",
    )

    assert result.exit_code == 0
    assert tool.queries == []
    assert any("Private context" in content for content in tool_feedback)


def test_public_weather_query_executes_with_private_memory_context(tmp_path):
    result, tool, tool_feedback = _run_chat_with_personal_context(
        tmp_path,
        model_query="과천시 오늘 날씨",
    )

    assert result.exit_code == 0
    assert tool.queries == ["과천시 오늘 날씨"]
    assert tool_feedback == ["과천시 현재 기온 24도"]


def test_private_profile_copy_is_blocked_without_custom_scanner(tmp_path):
    private_profile = "공박사"
    result, tool, tool_feedback = _run_chat_with_personal_context(
        tmp_path,
        model_query=f"과천 날씨와 {private_profile} 관련 정보를 검색",
        profile_text=private_profile,
    )

    assert result.exit_code == 0
    assert tool.queries == []
    assert any("Private context" in content for content in tool_feedback)


def test_current_public_query_matching_pending_context_executes(tmp_path):
    public_query = "오늘 과천시 날씨를 찾아줘."
    result, tool, tool_feedback = _run_chat_with_personal_context(
        tmp_path,
        model_query=public_query,
        pending_messages=(public_query,),
    )

    assert result.exit_code == 0
    assert tool.queries == [public_query]
    assert tool_feedback == ["과천시 현재 기온 24도"]
