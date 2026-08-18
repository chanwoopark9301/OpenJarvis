"""Cross-topic regressions for the generic model-led information path."""

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


class _GeneralInformationAgent(BaseAgent):
    agent_id = "general_information_model_led_agent"
    supports_managed_tool_fallback = True

    def run(
        self,
        input: str,
        context: AgentContext | None = None,
        **kwargs,
    ) -> AgentResult:
        raise AssertionError("model-led tool execution must replace this agent")


class _SearchTool(BaseTool):
    tool_id = "web_search"

    def __init__(self, content: str) -> None:
        self.content = content
        self.queries: list[str] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description="Search public information.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

    def execute(self, **params) -> ToolResult:
        self.queries.append(params["query"])
        return ToolResult(tool_name=self.tool_id, content=self.content)


@pytest.mark.parametrize(
    ("user_text", "query", "tool_content", "answer"),
    [
        (
            "초보 러닝 운동법을 인터넷에서 찾아줘.",
            "초보 러닝 운동법",
            "걷기와 달리기를 번갈아 시작",
            "처음에는 걷기와 달리기를 번갈아 시작해.",
        ),
        (
            "오늘 기술 뉴스를 찾아줘.",
            "오늘 기술 뉴스",
            "새 공개 모델 발표 소식",
            "새 공개 모델 발표 소식이 있어.",
        ),
    ],
)
def test_general_information_queries_can_be_model_selected(
    user_text, query, tool_content, answer
):
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "search-1",
                    "name": "web_search",
                    "arguments": json.dumps({"query": query}),
                }
            ],
        },
        {"content": answer},
    ]
    search_tool = _SearchTool(tool_content)
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.tools.enabled = "web_search"
    config.agent.max_turns = 3
    config.agent.context_from_memory = False
    config.memory.enabled = False
    config.personal_memory.enabled = False
    AgentRegistry.register_value(
        "general_information_model_led_agent", _GeneralInformationAgent
    )
    ToolRegistry.register_value("web_search", search_tool)

    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
    ):
        result = CliRunner().invoke(
            chat,
            ["--agent", "general_information_model_led_agent", "--model", "test-model"],
            input=f"{user_text}\n/quit\n",
        )

    assert result.exit_code == 0
    assert search_tool.queries == [query]
    assert answer in result.output
