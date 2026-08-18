"""User-shaped behavioral regressions for the model-led chat path."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.cli.chat_cmd import chat
from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import AgentRegistry, ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec


class _ModelLedChatAgent(BaseAgent):
    """Plain chat agent deliberately upgraded by the managed tool runtime."""

    agent_id = "model_led_general_chat_agent"
    supports_managed_tool_fallback = True

    def run(
        self,
        input: str,
        context: AgentContext | None = None,
        **kwargs,
    ) -> AgentResult:
        raise AssertionError("the managed tool runtime must own this chat turn")


class _SearchTool(BaseTool):
    tool_id = "web_search"

    def __init__(self, content: str) -> None:
        self.content = content
        self.queries: list[str] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description="Search the public web for current information.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

    def execute(self, **params) -> ToolResult:
        self.queries.append(params["query"])
        return ToolResult(tool_name=self.tool_id, content=self.content)


def run_model_led_chat(
    request: str,
    *,
    tool_query: str | None = None,
    tool_content: str = "",
    final_content: str | None = None,
    direct_content: str | None = None,
):
    """Run one chat turn with scripted model function-calling output."""
    if tool_query is None:
        assert direct_content is not None
        responses = [{"content": direct_content}]
    else:
        assert final_content is not None
        responses = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "search-1",
                        "name": "web_search",
                        "arguments": json.dumps({"query": tool_query}),
                    }
                ],
            },
            {"content": final_content},
        ]

    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = responses
    search_tool = _SearchTool(tool_content)
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.tools.enabled = "web_search"
    config.agent.max_turns = 3
    config.agent.context_from_memory = False
    config.memory.enabled = False
    config.personal_memory.enabled = False
    AgentRegistry.register_value("model_led_general_chat_agent", _ModelLedChatAgent)
    ToolRegistry.register_value("web_search", search_tool)

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
    ):
        result = CliRunner().invoke(
            chat,
            ["--agent", "model_led_general_chat_agent", "--model", "test-model"],
            input=f"{request}\n/quit\n",
        )

    return result, engine, search_tool


@pytest.mark.parametrize(
    "user_request",
    [
        "오늘 과천시의 날씨는 어떤지 검색해볼래?",
        "오늘 과천시 날씨를 알려달라고.",
        "인터넷에서 과천시 날씨를 찾아줘.",
    ],
)
def test_current_information_requests_can_reach_web_search(user_request):
    result, engine, search_tool = run_model_led_chat(
        user_request,
        tool_query="과천시 날씨",
        tool_content="기온 24도",
        final_content="과천시 기온은 24도야.",
    )

    assert result.exit_code == 0
    assert search_tool.queries == ["과천시 날씨"]
    assert "24도" in result.output


def test_greeting_does_not_call_web_search():
    result, engine, search_tool = run_model_led_chat(
        "안녕? 좋은 하루야.",
        direct_content="안녕! 좋은 하루야.",
    )

    assert result.exit_code == 0
    assert search_tool.queries == []
    assert engine.generate.call_count == 1


def test_missing_tool_value_is_not_replaced_by_a_harness_fact():
    result, engine, search_tool = run_model_led_chat(
        "오늘 과천시 날씨를 알려줘.",
        tool_query="과천시 날씨",
        tool_content="검색 결과에 현재 기온 값이 없음",
        final_content="검색 결과에서 현재 기온을 확인하지 못했어.",
    )

    assert "확인하지 못했어" in result.output
    assert "3도" not in result.output


def test_private_relationship_context_is_not_added_to_public_query():
    request = "내 여자친구를 데리러 가기 전에 과천시 날씨를 검색해줘."
    result, engine, search_tool = run_model_led_chat(
        request,
        tool_query="과천시 날씨",
        tool_content="기온 24도",
        final_content="과천시 기온은 24도야.",
    )

    assert search_tool.queries == ["과천시 날씨"]
    assert "여자친구" not in search_tool.queries[0]
