"""Tests for generic local request planning."""

from __future__ import annotations

import pytest

from openjarvis.cli._request_plan import (
    RequestPlan,
    RequestPlanner,
    needs_assistant_planning,
)
from openjarvis.core.types import Message, Role


class _FakeEngine:
    def __init__(self, *contents: str):
        self.contents = list(contents)
        self.calls: list[list[Message]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return {"content": self.contents.pop(0)}


@pytest.mark.parametrize(
    "text",
    ["안녕?", "오늘 좀 피곤했어", "그냥 이야기하자"],
)
def test_clear_companion_chat_skips_planner(text):
    assert needs_assistant_planning(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "오늘 날씨 검색해 볼래?",
        "운동법을 인터넷에서 찾아줘",
        "무설탕 음료 세 개를 비교해 줘",
        "대전 데이트 계획을 구체화해 줘",
        "쿠팡에서 음료수를 주문해 줘",
    ],
)
def test_information_and_action_requests_use_planner(text):
    assert needs_assistant_planning(text) is True


def test_pending_clarification_routes_short_answer_to_planner_once():
    assert needs_assistant_planning(
        "경기도 과천시야.", pending_clarification=True
    ) is True
    assert needs_assistant_planning("고마워.", pending_clarification=False) is False


def test_weather_without_location_returns_one_clarifying_question():
    engine = _FakeEngine(
        '{"mode":"clarify","goal":"오늘 날씨 알려주기",'
        '"clarifying_question":"어느 지역의 날씨를 볼까?",'
        '"response_style":"direct","requirements":[]}'
    )

    plan = RequestPlanner(engine, "test-model").plan(
        "오늘 날씨 검색해 볼래?", ()
    )

    assert plan == RequestPlan(
        mode="clarify",
        goal="오늘 날씨 알려주기",
        clarifying_question="어느 지역의 날씨를 볼까?",
        response_style="direct",
        requirements=(),
    )


def test_recent_public_context_resolves_follow_up_without_reasking():
    engine = _FakeEngine(
        '{"mode":"search","goal":"과천시 오늘 날씨",'
        '"clarifying_question":"","response_style":"direct",'
        '"requirements":[{"id":"R1","query":"경기도 과천시 오늘 날씨",'
        '"required_terms":["과천시","날씨"]}]}'
    )
    history = (
        Message(role=Role.USER, content="오늘 날씨를 검색해 줘."),
        Message(role=Role.ASSISTANT, content="어느 지역의 날씨를 볼까?"),
    )

    plan = RequestPlanner(engine, "test-model").plan(
        "경기도 과천시야.", history
    )

    assert plan is not None
    assert plan.mode == "search"
    assert plan.requirements[0].query == "경기도 과천시 오늘 날씨"
    assert "어느 지역의 날씨를 볼까?" in engine.calls[0][-1].content


def test_planner_passes_only_latest_six_non_system_messages():
    engine = _FakeEngine(
        '{"mode":"chat","goal":"대화","clarifying_question":"",'
        '"response_style":"direct","requirements":[]}'
    )
    history = tuple(
        Message(role=Role.USER, content=f"이전 대화 {index}")
        for index in range(8)
    ) + (Message(role=Role.SYSTEM, content="비밀 시스템 지시"),)

    RequestPlanner(engine, "test-model").plan("이어서 알려줘", history)

    planner_input = engine.calls[0][-1].content
    assert "이전 대화 1" not in planner_input
    assert "이전 대화 2" in planner_input
    assert "이전 대화 7" in planner_input
    assert "비밀 시스템 지시" not in planner_input


@pytest.mark.parametrize(
    "content",
    [
        '{"mode":"action","goal":"음료수 주문",'
        '"clarifying_question":"","response_style":"direct",'
        '"requirements":[{"id":"R1","query":"음료수",'
        '"required_terms":["음료수"]}]}',
        '{"mode":"search","goal":"날씨","clarifying_question":"",'
        '"response_style":"weather","requirements":[]}',
        '{"mode":"search","goal":"날씨","clarifying_question":"",'
        '"response_style":"direct","requirements":[]}',
        '{"mode":"clarify","goal":"날씨","clarifying_question":"",'
        '"response_style":"direct","requirements":[]}',
    ],
)
def test_invalid_mode_contracts_fail_closed_after_one_repair(content):
    engine = _FakeEngine(content, "still invalid")

    plan = RequestPlanner(engine, "test-model").plan("요청", ())

    assert plan is None
    assert len(engine.calls) == 2


@pytest.mark.parametrize(
    "query",
    [
        "여친 과천 날씨",
        "찬우 010-1234-5678 위치",
        "test@example.com 주문",
        "과천 今日 날씨",
    ],
)
def test_private_or_ideographic_search_query_fails_closed(query):
    content = (
        '{"mode":"search","goal":"공개 정보 찾기",'
        '"clarifying_question":"","response_style":"direct",'
        f'"requirements":[{{"id":"R1","query":"{query}",'
        '"required_terms":["과천"]}]}'
    )
    engine = _FakeEngine(content, "invalid repair")

    assert RequestPlanner(engine, "test-model").plan("검색해 줘", ()) is None
    assert len(engine.calls) == 2


def test_malformed_json_is_repaired_exactly_once():
    engine = _FakeEngine(
        "not json",
        '{"mode":"search","goal":"과천 날씨","clarifying_question":"",'
        '"response_style":"direct","requirements":[{"id":"R1",'
        '"query":"과천시 오늘 날씨","required_terms":["과천시","날씨"]}]}',
    )

    plan = RequestPlanner(engine, "test-model").plan("과천 날씨를 검색해 줘", ())

    assert plan is not None
    assert plan.mode == "search"
    assert len(engine.calls) == 2
    assert "not json" in engine.calls[1][-1].content


def test_duplicate_or_nonsequential_requirements_fail_closed():
    content = (
        '{"mode":"search","goal":"비교","clarifying_question":"",'
        '"response_style":"comparison","requirements":['
        '{"id":"R1","query":"제품 하나","required_terms":["제품"]},'
        '{"id":"R3","query":"제품 둘","required_terms":["제품"]}]}'
    )
    engine = _FakeEngine(content, "invalid repair")

    assert RequestPlanner(engine, "test-model").plan("제품 비교해 줘", ()) is None
