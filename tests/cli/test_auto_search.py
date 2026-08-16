"""Tests for privacy-safe automatic web-search preflight."""

from __future__ import annotations

import pytest

from openjarvis.cli._auto_search import (
    AutoSearchPreflight,
    AutoSearchResult,
    needs_web_search,
)
from openjarvis.cli._search_evidence import SearchRequirement
from openjarvis.core.types import ToolResult


class _FakeEngine:
    def __init__(self, content: str):
        self.content = content
        self.inputs: list[str] = []
        self.system_inputs: list[str] = []

    def generate(self, messages, **kwargs):
        self.system_inputs.append(messages[0].content)
        self.inputs.append(messages[-1].content)
        return {"content": self.content}


class _SequencedEngine:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.inputs: list[str] = []

    def generate(self, messages, **kwargs):
        self.inputs.append(messages[-1].content)
        return {"content": self.contents.pop(0)}


class _FakeSearchTool:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.queries: list[str] = []

    def execute(self, **params):
        query = params["query"]
        self.queries.append(query)
        if self.fail:
            return ToolResult(
                tool_name="web_search",
                content="Search error: unavailable",
                success=False,
            )
        return ToolResult(
            tool_name="web_search",
            content=(
                f"### {query}\n"
                f"Source: https://example.com/{len(self.queries)}\n"
                "Summary: verified public information"
            ),
            success=True,
            metadata={"engine": "test"},
        )


class _QuerySearchTool:
    def __init__(self, content_by_query: dict[str, str]):
        self.content_by_query = content_by_query
        self.queries: list[str] = []

    def execute(self, **params):
        query = params["query"]
        self.queries.append(query)
        return ToolResult(
            tool_name="web_search",
            content=self.content_by_query.get(query, "No results found."),
            success=True,
            metadata={"engine": "test"},
        )


@pytest.mark.parametrize(
    "user_text",
    [
        "안녕?",
        "요즘 내가 왜 이렇게 지치는지 모르겠어",
        "이 글을 요약해 줘",
    ],
)
def test_ordinary_or_personal_chat_does_not_trigger_web_search(user_text):
    """Broad routing must not send ordinary companion chat to the web."""
    assert needs_web_search(user_text) is False


@pytest.mark.parametrize(
    "user_text",
    [
        "내 상태와 관련된 심리 연구를 인터넷에서 찾아줘",
        "요즘 인기 있는 러닝화를 검색해 줘",
        "인터넷에서 대전 맛집을 찾아줘",
        "웹에서 테미오래 운영시간을 확인해 줘",
    ],
)
def test_explicit_search_request_always_triggers_web_search(user_text):
    """Removing explicit commands would ignore the user's search consent."""
    assert needs_web_search(user_text) is True


@pytest.mark.parametrize(
    "user_text",
    [
        "내일 대전 테미오래 운영시간을 확인해 줘",
        "대전 한식 맛집과 소품샵을 추천해 줘",
        "오늘 서울 날씨가 어때?",
    ],
)
def test_current_or_local_factual_request_triggers_web_search(user_text):
    """Current and real-place facts must not be answered from model memory."""
    assert needs_web_search(user_text) is True


def test_blank_input_does_not_trigger_web_search():
    assert needs_web_search("   ") is False


def test_non_search_chat_skips_query_planner_and_search_tool():
    engine = _FakeEngine('["unused"]')
    search_tool = _FakeSearchTool()
    result = AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare("안녕?")

    assert result == AutoSearchResult(triggered=False, success=False)
    assert engine.inputs == []
    assert search_tool.queries == []


def test_query_planner_requires_strict_json_before_searching():
    engine = _FakeEngine("['대전 테미오래 운영시간']")
    search_tool = _FakeSearchTool()

    result = AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare("인터넷에서 테미오래 운영시간을 찾아줘")

    assert result.triggered is True
    assert result.success is False
    assert search_tool.queries == []
    assert "확인하지 못했어" in result.error


def test_query_planner_limits_trims_and_deduplicates_queries():
    engine = _FakeEngine(
        '[{"id":"R1","kind":"general","query":" 대전 테미오래 운영시간 ",'
        '"required_terms":["테미오래"]},'
        '{"id":"R2","kind":"food","query":"대전 한식 맛집",'
        '"required_terms":["대전","한식"]},'
        '{"id":"R3","kind":"shops","query":"대전 소품샵",'
        '"required_terms":["대전","소품샵"]},'
        '{"id":"R4","kind":"general","query":"무시할 네 번째",'
        '"required_terms":["무시"]}]'
    )
    search_tool = _FakeSearchTool()

    result = AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare("인터넷에서 대전 여행지를 찾아줘")

    assert result.queries == (
        "대전 테미오래 운영시간",
        "대전 한식 맛집",
        "대전 소품샵",
    )
    assert search_tool.queries == list(result.queries)


@pytest.mark.parametrize(
    "private_query",
    [
        "여자친구와 대전 맛집",
        "여친 대전 데이트",
        "남자친구 서울 카페",
        "남친 부산 여행",
        "chanwoo@example.com 주변 식당",
        "010-1234-5678 근처 카페",
    ],
)
def test_private_query_is_rejected_before_external_search(private_query):
    engine = _FakeEngine(
        f'[{{"id":"R1","kind":"food","query":"{private_query}",'
        '"required_terms":["대전"]}]'
    )
    search_tool = _FakeSearchTool()

    result = AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare("인터넷에서 대전 일정을 찾아줘")

    assert result.success is False
    assert result.queries == ()
    assert search_tool.queries == []


def test_non_local_planner_never_receives_raw_user_text():
    engine = _FakeEngine('["대전 맛집"]')
    search_tool = _FakeSearchTool()

    result = AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=False,
    ).prepare("내 여자친구와 갈 대전 맛집을 검색해 줘")

    assert result.triggered is True
    assert result.success is False
    assert engine.inputs == []
    assert search_tool.queries == []


def test_external_search_receives_only_planned_public_queries():
    raw_request = "내 여자친구를 10시에 픽업해서 갈 대전 맛집을 검색해 줘"
    engine = _FakeEngine(
        '[{"id":"R1","kind":"food","query":"대전 한식 맛집 운영시간",'
        '"required_terms":["대전","한식"]}]'
    )
    search_tool = _FakeSearchTool()

    result = AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare(raw_request)

    assert result.success is True
    assert search_tool.queries == ["대전 한식 맛집 운영시간"]
    assert all(raw_request not in query for query in search_tool.queries)


def test_query_planner_is_told_to_preserve_places_and_ignore_background():
    engine = _FakeEngine(
        '[{"id":"R1","kind":"general","query":"대전 테미오래 운영시간",'
        '"required_terms":["테미오래"]}]'
    )
    search_tool = _FakeSearchTool()

    AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare("꽃다발을 준비했어. 테미오래 운영시간을 검색해 줘")

    prompt = engine.system_inputs[0]
    assert "Preserve exact public place" in prompt
    assert "Ignore unrelated background" in prompt
    assert "official hours/access" in prompt
    assert engine.inputs == ["테미오래 운영시간을 검색해 줘"]


def test_query_planner_keeps_immediately_preceding_trip_context():
    engine = _FakeEngine(
        '[{"id":"R1","kind":"general",'
        '"query":"대전 테미오래 근처 한식 맛집",'
        '"required_terms":["대전","한식"]}]'
    )
    search_tool = _FakeSearchTool()

    AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare(
        "꽃다발을 준비했어. 내일 대전 테미오래를 방문할 거야. "
        "그 후 한식 맛집을 찾아줘."
    )

    assert engine.inputs == [
        "내일 대전 테미오래를 방문할 거야. 그 후 한식 맛집을 찾아줘"
    ]


def test_query_planner_keeps_multi_sentence_trip_request_without_small_talk():
    engine = _FakeEngine(
        '[{"id":"R1","kind":"general",'
        '"query":"대전 테미오래 근처 한식 맛집",'
        '"required_terms":["대전","한식"]}]'
    )
    search_tool = _FakeSearchTool()

    AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare(
        "분홍색 꽃다발이야. 내일 대전에서 출발해. 테미오래를 방문할 거야. "
        "그 후 맛집과 소품샵을 둘러볼 거야. 계획을 구체화해 줘. "
        "한식과 모여 있는 소품샵을 인터넷에서 찾아줘."
    )

    planning_input = engine.inputs[0]
    assert "대전" in planning_input
    assert "테미오래" in planning_input
    assert "한식" in planning_input
    assert "소품샵" in planning_input
    assert "꽃다발" not in planning_input


def test_successful_search_builds_untrusted_context_and_unique_sources():
    engine = _FakeEngine(
        '[{"id":"R1","kind":"general","query":"대전 테미오래",'
        '"required_terms":["대전","테미오래"]},'
        '{"id":"R2","kind":"food","query":"대전 한식 맛집",'
        '"required_terms":["대전","한식"]}]'
    )
    search_tool = _FakeSearchTool()

    result = AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare("인터넷에서 대전 여행 정보를 찾아줘")

    assert result.success is True
    assert "UNTRUSTED WEB SEARCH RESULTS" in result.context
    assert "Include useful source links" in result.context
    assert "Never invent venue names, addresses, phone numbers, or hours" in (
        result.context
    )
    assert "Preserve the exact spelling" in result.context
    assert result.sources == (
        "https://example.com/1",
        "https://example.com/2",
    )


def test_all_failed_searches_return_fixed_error_without_context():
    engine = _FakeEngine(
        '[{"id":"R1","kind":"general","query":"대전 테미오래 운영시간",'
        '"required_terms":["테미오래"]}]'
    )
    search_tool = _FakeSearchTool(fail=True)

    result = AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare("웹에서 테미오래 운영시간을 확인해 줘")

    assert result.triggered is True
    assert result.success is False
    assert result.context == ""
    assert result.error == (
        "인터넷에서 필요한 정보를 확인하지 못했어. 확인되지 않은 장소나 운영 "
        "정보를 지어내지는 않을게. 잠시 뒤 다시 검색해 줘."
    )


def test_search_content_without_source_url_fails_closed():
    class _NoSourceSearchTool:
        def execute(self, **params):
            return ToolResult(
                tool_name="web_search",
                content="No results found.",
                success=True,
            )

    result = AutoSearchPreflight(
        _FakeEngine(
            '[{"id":"R1","kind":"general",'
            '"query":"대전 테미오래 운영시간",'
            '"required_terms":["테미오래"]}]'
        ),
        "test-model",
        _NoSourceSearchTool(),
        local_planning_allowed=True,
    ).prepare("테미오래 운영시간을 인터넷에서 찾아줘")

    assert result.success is False
    assert result.sources == ()
    assert "확인하지 못했어" in result.error


def test_query_planner_creates_structured_requirements():
    engine = _FakeEngine(
        '[{"id":"R1","kind":"official",'
        '"query":"대전 테미오래 공식 운영시간 접근",'
        '"required_terms":["대전","테미오래"]},'
        '{"id":"R2","kind":"food",'
        '"query":"대전 테미오래 근처 한식 맛집",'
        '"required_terms":["대전","한식"]},'
        '{"id":"R3","kind":"shops",'
        '"query":"대전 테미오래 주변 소품샵 여러 곳",'
        '"required_terms":["대전","소품샵"]}]'
    )

    result = AutoSearchPreflight(
        engine,
        "test-model",
        _FakeSearchTool(),
        local_planning_allowed=True,
    ).prepare("인터넷에서 대전 테미오래 일정을 찾아줘")

    assert result.requirements == (
        SearchRequirement(
            "R1",
            "official",
            "대전 테미오래 공식 운영시간 접근",
            ("대전", "테미오래"),
        ),
        SearchRequirement(
            "R2",
            "food",
            "대전 테미오래 근처 한식 맛집",
            ("대전", "한식"),
        ),
        SearchRequirement(
            "R3",
            "shops",
            "대전 테미오래 주변 소품샵 여러 곳",
            ("대전", "소품샵"),
        ),
    )


def test_only_unmet_requirement_is_retried_with_public_context():
    raw_request = (
        "여친을 픽업해서 대전 테미오래와 한식 맛집, 소품샵을 찾아줘"
    )
    engine = _SequencedEngine(
        [
            '[{"id":"R1","kind":"official",'
            '"query":"대전 테미오래 공식 운영시간 접근",'
            '"required_terms":["대전","테미오래"]},'
            '{"id":"R2","kind":"food",'
            '"query":"대전 테미오래 근처 한식 맛집",'
            '"required_terms":["대전","한식"]},'
            '{"id":"R3","kind":"shops",'
            '"query":"대전 테미오래 주변 소품샵 여러 곳",'
            '"required_terms":["대전","소품샵"]}]',
            '{"query":"대전 테미오래 인근 소품샵 모음 대흥동 은행동"}',
        ]
    )
    search_tool = _QuerySearchTool(
        {
            "대전 테미오래 공식 운영시간 접근": (
                "### 테미오래 공식 누리집\n"
                "Source: https://temiorae.com/\n"
                "Summary: 대전 테미오래 운영시간과 관람 안내"
            ),
            "대전 테미오래 근처 한식 맛집": (
                "### 대전 한식 맛집\n"
                "Source: https://www.diningcode.com/daejeon\n"
                "Summary: 대전 지역 한식 식당 목록"
            ),
            "대전 테미오래 주변 소품샵 여러 곳": (
                "### 서울 소품샵\n"
                "Source: https://example.com/seoul\n"
                "Summary: 서울 상점 목록"
            ),
            "대전 테미오래 인근 소품샵 모음 대흥동 은행동": (
                "### 대전 소품샵 모음\n"
                "Source: https://example.com/daejeon-shops\n"
                "Summary: 대전 대흥동 소품샵 여러 곳"
            ),
        }
    )

    result = AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare(raw_request)

    assert result.success is True
    assert search_tool.queries == [
        "대전 테미오래 공식 운영시간 접근",
        "대전 테미오래 근처 한식 맛집",
        "대전 테미오래 주변 소품샵 여러 곳",
        "대전 테미오래 인근 소품샵 모음 대흥동 은행동",
    ]
    assert raw_request not in engine.inputs[1]
    assert "여친" not in engine.inputs[1]


def test_unmet_requirement_retries_only_once_when_retry_is_still_unrelated():
    engine = _SequencedEngine(
        [
            '[{"id":"R1","kind":"shops","query":"대전 소품샵",'
            '"required_terms":["대전","소품샵"]}]',
            '{"query":"대전 소품 상점 모음"}',
        ]
    )
    search_tool = _QuerySearchTool(
        {
            "대전 소품샵": (
                "### 부산 소품샵\nSource: https://example.com/one\n"
                "Summary: 부산 소품샵"
            ),
            "대전 소품 상점 모음": (
                "### 서울 상점\nSource: https://example.com/two\n"
                "Summary: 서울 상점"
            ),
        }
    )

    result = AutoSearchPreflight(
        engine,
        "test-model",
        search_tool,
        local_planning_allowed=True,
    ).prepare("대전 소품샵을 찾아줘")

    assert result.success is True
    assert search_tool.queries == ["대전 소품샵", "대전 소품 상점 모음"]
    assert len(engine.inputs) == 2
