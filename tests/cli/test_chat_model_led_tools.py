"""Model-led tool use regressions for ``jarvis chat``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.cli.chat_cmd import chat
from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import AgentRegistry, ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


class _ManagedFallbackChatAgent(BaseAgent):
    agent_id = "managed_fallback_chat_agent"
    supports_managed_tool_fallback = True

    def run(
        self,
        input: str,
        context: AgentContext | None = None,
        **kwargs,
    ) -> AgentResult:
        raise AssertionError("managed tool fallback should replace this agent")


class SearchTool(BaseTool):
    tool_id = "web_search"

    def __init__(self, result: str) -> None:
        self.result = result

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description="Search test data.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

    def execute(self, **params) -> ToolResult:
        return ToolResult(tool_name=self.tool_id, content=self.result)


def invoke_chat(input_text: str, engine: MagicMock, search_tool: SearchTool):
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.tools.enabled = "web_search"
    config.agent.max_turns = 3
    AgentRegistry.register_value(
        "managed_fallback_chat_agent", _ManagedFallbackChatAgent
    )
    ToolRegistry.register_value("web_search", search_tool)

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
    ):
        return CliRunner().invoke(
            chat,
            ["--agent", "managed_fallback_chat_agent", "--model", "test-model"],
            input=input_text,
        )


def test_simple_chat_uses_enabled_search_tool_without_preflight():
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "search-1",
                    "name": "web_search",
                    "arguments": '{"query":"과천시 날씨"}',
                }
            ],
        },
        {"content": "과천시 기온은 24도야."},
    ]

    result = invoke_chat(
        "오늘 과천시 날씨를 알려줘.\n/quit\n",
        engine,
        SearchTool("기온 24도"),
    )

    assert result.exit_code == 0
    assert "과천시 기온은 24도야." in result.output
    assert engine.generate.call_count == 2
