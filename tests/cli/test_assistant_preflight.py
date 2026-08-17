"""Tests for generic assistant planning and search preflight."""

from __future__ import annotations

from dataclasses import replace

from openjarvis.cli._assistant_preflight import (
    AssistantPreflight,
    AssistantPreflightResult,
)
from openjarvis.cli._request_plan import PlanRequirement, RequestPlan
from openjarvis.core.types import Message, Role, ToolResult


class _FixedPlanner:
    def __init__(self, plan):
        self.fixed_plan = plan
        self.calls = []

    def plan(self, user_text, recent_messages):
        self.calls.append((user_text, recent_messages))
        return self.fixed_plan


class _ExplodingPlanner:
    def plan(self, user_text, recent_messages):
        raise AssertionError("ordinary chat must not invoke planner")


class _ExplodingSearchTool:
    def execute(self, **params):
        raise AssertionError("this mode must not invoke search")


class _RecordingSearchTool:
    def __init__(self, content_by_query=None):
        self.content_by_query = content_by_query or {}
        self.queries = []

    def execute(self, **params):
        query = params["query"]
        self.queries.append(query)
        content = self.content_by_query.get(query, "No results found.")
        return ToolResult(tool_name="web_search", content=content, success=True)


def _weather_plan():
    return RequestPlan(
        mode="search",
        goal="과천시 오늘 날씨 알려주기",
        clarifying_question="",
        response_style="direct",
        requirements=(
            PlanRequirement(
                id="R1",
                query="경기도 과천시 오늘 날씨",
                required_terms=("과천시", "날씨"),
            ),
        ),
    )


def test_clear_chat_does_not_call_planner_or_search():
    result = AssistantPreflight(
        _ExplodingPlanner(), _ExplodingSearchTool()
    ).prepare("안녕?", ())

    assert result == AssistantPreflightResult(triggered=False)


def test_planner_chat_mode_returns_to_normal_generation():
    plan = RequestPlan("chat", "사용자와 대화", "", "direct", ())

    result = AssistantPreflight(
        _FixedPlanner(plan), _ExplodingSearchTool()
    ).prepare("그건 어때?", (), pending_clarification=True)

    assert result.triggered is False
    assert result.mode == "chat"


def test_clarify_returns_question_without_search():
    plan = RequestPlan(
        "clarify", "오늘 날씨", "어느 지역의 날씨를 볼까?", "direct", ()
    )

    result = AssistantPreflight(
        _FixedPlanner(plan), _ExplodingSearchTool()
    ).prepare("오늘 날씨 검색해 줘", ())

    assert result.triggered is True
    assert result.mode == "clarify"
    assert result.clarifying_question == "어느 지역의 날씨를 볼까?"
    assert result.success is False


def test_action_returns_unavailable_without_claiming_completion():
    plan = RequestPlan("action", "음료수 주문", "", "direct", ())

    result = AssistantPreflight(
        _FixedPlanner(plan), _ExplodingSearchTool()
    ).prepare("쿠팡에서 음료수 주문해 줘", ())

    assert result.mode == "action"
    assert "직접 조작하는 기능은 아직 연결되지 않았어" in result.error
    assert "주문했어" not in result.error


def test_failed_plan_asks_for_one_rephrasing_without_search():
    result = AssistantPreflight(
        _FixedPlanner(None), _ExplodingSearchTool()
    ).prepare("인터넷에서 뭔가 찾아줘", ())

    assert result.mode == "clarify"
    assert result.clarifying_question == (
        "요청을 이해하지 못했어. 찾고 싶은 내용을 한 문장으로 다시 말해 줄래?"
    )


def test_search_executes_generic_requirements_without_domain_kinds():
    search = _RecordingSearchTool(
        {
            "경기도 과천시 오늘 날씨": (
                "### 과천시 오늘 날씨\n"
                "Source: https://weather.test/gwacheon\n"
                "Summary: 과천시 오늘 날씨 현재 기온 24도"
            )
        }
    )

    result = AssistantPreflight(_FixedPlanner(_weather_plan()), search).prepare(
        "과천시 오늘 날씨를 검색해 줘", ()
    )

    assert result.success is True
    assert result.mode == "search"
    assert result.response_style == "direct"
    assert result.queries == ("경기도 과천시 오늘 날씨",)
    assert result.evidence[0].requirement_id == "R1"
    assert "UNTRUSTED WEB SEARCH RESULTS" in result.context


def test_recent_messages_are_passed_only_to_local_planner():
    planner = _FixedPlanner(_weather_plan())
    search = _RecordingSearchTool(
        {
            "경기도 과천시 오늘 날씨": (
                "### 과천시 날씨\nSource: https://weather.test\n"
                "Summary: 과천시 날씨"
            )
        }
    )
    history = (
        Message(role=Role.USER, content="오늘 날씨를 검색해 줘"),
        Message(role=Role.ASSISTANT, content="어느 지역의 날씨를 볼까?"),
    )

    AssistantPreflight(planner, search).prepare(
        "과천시야", history, pending_clarification=True
    )

    assert planner.calls == [("과천시야", history)]
    assert search.queries == ["경기도 과천시 오늘 날씨"]


def test_private_query_is_rejected_again_before_external_search():
    private_plan = replace(
        _weather_plan(),
        requirements=(
            PlanRequirement("R1", "여친 과천시 오늘 날씨", ("과천시", "날씨")),
        ),
    )
    search = _RecordingSearchTool()

    result = AssistantPreflight(_FixedPlanner(private_plan), search).prepare(
        "날씨를 검색해 줘", ()
    )

    assert result.success is False
    assert result.queries == ()
    assert search.queries == []


def test_only_unmet_requirement_retries_once_with_public_terms():
    plan = RequestPlan(
        "search",
        "날씨와 행사 확인",
        "",
        "list",
        (
            PlanRequirement("R1", "과천시 오늘 날씨", ("과천시", "날씨")),
            PlanRequirement("R2", "과천시 오늘 행사", ("과천시", "행사")),
        ),
    )
    search = _RecordingSearchTool(
        {
            "과천시 오늘 날씨": (
                "### 과천시 날씨\nSource: https://weather.test\n"
                "Summary: 과천시 오늘 날씨"
            ),
            "과천시 오늘 행사": (
                "### 서울 행사\nSource: https://events.test/seoul\n"
                "Summary: 서울 행사"
            ),
            "과천시 행사 공식 최신 정보": (
                "### 과천시 행사 안내\nSource: https://events.test/gwacheon\n"
                "Summary: 과천시 오늘 행사"
            ),
        }
    )

    result = AssistantPreflight(_FixedPlanner(plan), search).prepare(
        "과천시 날씨와 행사를 찾아줘", ()
    )

    assert result.success is True
    assert search.queries == [
        "과천시 오늘 날씨",
        "과천시 오늘 행사",
        "과천시 행사 공식 최신 정보",
    ]


def test_all_irrelevant_searches_return_generic_failure():
    search = _RecordingSearchTool(
        {
            "경기도 과천시 오늘 날씨": (
                "### 시카고 날씨\nSource: https://weather.test/chicago\n"
                "Summary: 시카고 오늘 날씨"
            ),
            "과천시 날씨 공식 최신 정보": "No results found.",
        }
    )

    result = AssistantPreflight(_FixedPlanner(_weather_plan()), search).prepare(
        "과천시 날씨를 검색해 줘", ()
    )

    assert result.success is False
    assert result.evidence == ()
    assert "확인하지 못했어" in result.error
