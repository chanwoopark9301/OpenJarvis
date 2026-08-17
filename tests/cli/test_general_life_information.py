"""Cross-topic regression tests for one generic life-information path."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from openjarvis.cli._assistant_preflight import AssistantPreflight
from openjarvis.cli._grounded_response import (
    GroundedResponse,
    ResponseBlock,
    Support,
    render_grounded_response,
    validate_grounded_response,
)
from openjarvis.cli._request_plan import PlanRequirement, RequestPlan


@dataclass
class _ToolResult:
    success: bool
    content: str


class _Planner:
    def __init__(self, plan: RequestPlan):
        self.plan_value = plan

    def plan(self, user_text, recent_messages):
        return self.plan_value


class _SearchTool:
    def __init__(self, title: str, summary: str):
        self.title = title
        self.summary = summary
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return _ToolResult(
            True,
            f"### {self.title}\nSource: https://example.com/info\n"
            f"Summary: {self.summary}",
        )


@pytest.mark.parametrize(
    (
        "user_text",
        "goal",
        "query",
        "terms",
        "title",
        "summary",
        "fact",
        "excerpt",
    ),
    [
        (
            "과천시 오늘 날씨를 찾아줘",
            "과천시 오늘 날씨 안내",
            "과천시 오늘 날씨",
            ("과천시", "날씨"),
            "과천시 오늘 날씨",
            "현재 기온 24도",
            "현재 기온은 24도야.",
            "현재 기온 24도",
        ),
        (
            "초보 러닝 운동법을 인터넷에서 찾아줘",
            "초보 러닝 방법 안내",
            "초보 러닝 운동법",
            ("초보", "러닝"),
            "초보 러닝 안내",
            "걷기와 달리기를 번갈아 시작",
            "처음에는 걷기와 달리기를 번갈아 시작해.",
            "걷기와 달리기를 번갈아 시작",
        ),
        (
            "무설탕 음료를 비교해 줘",
            "무설탕 음료 비교",
            "무설탕 음료 비교",
            ("무설탕", "음료"),
            "무설탕 음료 비교",
            "라임맛 제품은 당류 0그램",
            "라임맛 제품은 당류가 0그램이야.",
            "라임맛 제품은 당류 0그램",
        ),
        (
            "오늘 기술 뉴스를 찾아줘",
            "오늘 기술 뉴스 요약",
            "오늘 기술 뉴스",
            ("기술", "뉴스"),
            "오늘 기술 뉴스",
            "새 공개 모델 발표 소식",
            "새 공개 모델 발표 소식이 있어.",
            "새 공개 모델 발표 소식",
        ),
        (
            "대전 당일 여행 계획을 구체화해 줘",
            "대전 당일 여행 정보 정리",
            "대전 당일 여행",
            ("대전", "여행"),
            "대전 당일 여행 안내",
            "도심 문화 공간 관람 정보",
            "도심 문화 공간 관람 정보가 있어.",
            "도심 문화 공간 관람 정보",
        ),
    ],
)
def test_everyday_topics_share_one_search_and_grounding_path(
    user_text,
    goal,
    query,
    terms,
    title,
    summary,
    fact,
    excerpt,
):
    plan = RequestPlan(
        mode="search",
        goal=goal,
        clarifying_question="",
        response_style="direct",
        requirements=(PlanRequirement("R1", query, terms),),
    )
    search_tool = _SearchTool(title, summary)

    prepared = AssistantPreflight(_Planner(plan), search_tool).prepare(user_text)

    assert prepared.success is True
    assert prepared.mode == "search"
    assert prepared.evidence[0].requirement_id == "R1"
    response = GroundedResponse(
        "확인했어.",
        (ResponseBlock("fact", fact, (Support("S1", excerpt),)),),
        "",
    )
    validated = validate_grounded_response(response, prepared.evidence)
    rendered = render_grounded_response(
        validated,
        prepared.evidence,
        style=prepared.response_style,
    )
    assert fact in rendered
    assert "https://example.com/info" in rendered


def test_external_action_uses_the_same_planner_but_never_claims_completion():
    plan = RequestPlan(
        mode="action",
        goal="음료 주문",
        clarifying_question="",
        response_style="steps",
        requirements=(),
    )
    prepared = AssistantPreflight(
        _Planner(plan),
        _SearchTool("사용 안 함", "사용 안 함"),
    ).prepare("쿠팡에서 음료수를 주문해 줘")

    assert prepared.mode == "action"
    assert prepared.success is False
    assert "아직 연결되지 않았어" in prepared.error
    assert "주문했어" not in prepared.error
