"""Generic local planning for conversation, search, and future actions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from openjarvis.core.types import Message, Role

_MODES = frozenset({"chat", "clarify", "search", "action"})
_STYLES = frozenset({"direct", "list", "comparison", "steps"})
_WORK_SIGNALS = (
    "검색",
    "찾아",
    "확인",
    "추천",
    "비교",
    "계획",
    "구체화",
    "주문",
    "구매",
    "예약",
    "결제",
    "보내줘",
)
_TEMPORAL_SIGNALS = ("오늘", "내일", "지금", "현재", "최신", "이번주", "이번 주")
_CHANGING_INFO_SIGNALS = (
    "날씨",
    "가격",
    "요금",
    "시간",
    "운영",
    "영업",
    "휴무",
    "재고",
    "교통",
    "뉴스",
)
_PRIVATE_QUERY_SIGNALS = (
    "여자친구",
    "여자 친구",
    "여친",
    "남자친구",
    "남자 친구",
    "남친",
    "내 마음",
    "내 심리",
    "내 감정",
)
_IDEOGRAPH_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<!\d)01[016789][ -]?\d{3,4}[ -]?\d{4}(?!\d)")
_PLACEHOLDER_TERMS = frozenset(
    {
        "place",
        "location",
        "city",
        "region",
        "product",
        "item",
        "topic",
        "subject",
    }
)
_KOREAN_PLACE_PATTERN = re.compile(
    r"(?:서울|부산|대구|인천|광주|대전|울산|세종|제주|"
    r"[가-힣]{2,}(?:시|군|구|도))"
)
_WEATHER_PREFIX_PATTERN = re.compile(r"([가-힣]{2,})\s*날씨")
_NON_PLACE_WEATHER_PREFIXES = frozenset(
    {"오늘", "내일", "현재", "지금", "요즘", "이번주"}
)

_PLAN_SYSTEM_PROMPT = """Plan one everyday assistant request locally. Return only
one JSON object with exactly: mode, goal, clarifying_question, response_style,
requirements. mode is chat, clarify, search, or action. response_style is direct,
list, comparison, or steps. Search has one to three requirements, each with exactly
id, query, required_terms; IDs are R1, R2, R3 in order. All other modes have no
requirements. All text fields are strings, never null. Clarify asks one concise
question in the current request's language only when essential information is
missing. Use recent conversation to resolve follow-ups and never ask for information
already present. If a search needs an unknown place, item, date range, or other
essential selector, use clarify instead of a broad search. Never use placeholders
such as place, location, city, region, product, item, topic, or subject. Every
required term must be a concrete word that appears in its query. Action means
changing an external site or account. Queries contain only public subject, place,
product, and time terms. Keep the exact spelling and language of public names from
the request and recent conversation. Phrase each query to retrieve the concrete
answer value, not merely a page about the topic. Never include people, relations,
feelings, home details, phones, emails, gifts, or private context. Never answer the
request and never create a domain-specific mode. You cannot access device location.
For example, current weather without a place must ask which place."""

_REPAIR_SYSTEM_PROMPT = """Repair one malformed assistant plan. Return only one JSON
object with exactly mode, goal, clarifying_question, response_style, requirements.
mode is chat, clarify, search, or action. response_style is direct, list, comparison,
or steps. Search contains one to three sequential R1-R3 requirements with exactly
id, query, required_terms. All text fields are strings, never null. Other modes
contain no requirements. Do not add facts or
private information. If the original request lacks an essential selector, return
clarify with one concise question in the original request's language. Never use
placeholder terms, and make every
required term appear in its query. You cannot access device location; current
weather without a place must ask which place. Preserve the exact spelling and
language of public names, and ask search for concrete answer values."""


@dataclass(frozen=True)
class PlanRequirement:
    """One bounded public-information search requirement."""

    id: str
    query: str
    required_terms: tuple[str, ...]


@dataclass(frozen=True)
class RequestPlan:
    """One generic local plan for the current assistant turn."""

    mode: str
    goal: str
    clarifying_question: str
    response_style: str
    requirements: tuple[PlanRequirement, ...]


def needs_assistant_planning(
    user_text: str,
    pending_clarification: bool = False,
) -> bool:
    """Return whether a turn needs planning beyond ordinary conversation."""
    if pending_clarification:
        return True
    compact = re.sub(r"\s+", "", user_text.casefold())
    if not compact:
        return False
    if any(signal in compact for signal in _WORK_SIGNALS):
        return True
    if "인터넷" in compact or "웹에서" in compact:
        return True
    return any(signal in compact for signal in _TEMPORAL_SIGNALS) and any(
        signal in compact for signal in _CHANGING_INFO_SIGNALS
    )


def _role_value(message: Message) -> str:
    return message.role.value if isinstance(message.role, Role) else str(message.role)


def _planner_input(
    user_text: str,
    recent_messages: tuple[Message, ...],
) -> str:
    bounded = [
        {"role": _role_value(message), "content": message.content}
        for message in recent_messages
        if _role_value(message) != Role.SYSTEM.value
    ][-6:]
    return json.dumps(
        {"recent_conversation": bounded, "current_request": user_text},
        ensure_ascii=False,
    )


def _content(response: Any) -> str:
    return response.get("content", "") if isinstance(response, dict) else str(response)


def is_safe_public_query(value: str) -> bool:
    """Return whether text is safe to send to a public search service."""
    normalized = value.casefold()
    return not (
        any(signal in normalized for signal in _PRIVATE_QUERY_SIGNALS)
        or bool(_EMAIL_PATTERN.search(value))
        or bool(_PHONE_PATTERN.search(value))
        or bool(_IDEOGRAPH_PATTERN.search(value))
    )


def _normalize_for_match(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def _parse_plan(content: str) -> RequestPlan | None:
    try:
        value = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        isinstance(value, dict)
        and value.get("mode") in {"chat", "clarify", "action"}
        and "requirements" not in value
    ):
        value = {**value, "requirements": []}
    if not isinstance(value, dict) or set(value) != {
        "mode",
        "goal",
        "clarifying_question",
        "response_style",
        "requirements",
    }:
        return None

    mode = value["mode"]
    goal = value["goal"]
    question = value["clarifying_question"]
    style = value["response_style"]
    raw_requirements = value["requirements"]
    if (
        mode not in _MODES
        or style not in _STYLES
        or not isinstance(goal, str)
        or not goal.strip()
        or not isinstance(question, str)
        or not isinstance(raw_requirements, list)
    ):
        return None

    if mode == "search":
        if question.strip() or not 1 <= len(raw_requirements) <= 3:
            return None
    elif raw_requirements:
        return None
    elif mode == "clarify" and not question.strip():
        return None
    elif mode != "clarify" and question.strip():
        return None

    requirements: list[PlanRequirement] = []
    seen_queries: set[str] = set()
    for index, raw in enumerate(raw_requirements, start=1):
        if not isinstance(raw, dict) or set(raw) != {
            "id",
            "query",
            "required_terms",
        }:
            return None
        query = raw["query"]
        terms = raw["required_terms"]
        expected_id = f"R{index}"
        if (
            raw["id"] != expected_id
            or not isinstance(query, str)
            or not query.strip()
            or query.strip() in seen_queries
            or not is_safe_public_query(query)
            or not isinstance(terms, list)
            or not 1 <= len(terms) <= 3
            or any(
                not isinstance(term, str)
                or not term.strip()
                or not is_safe_public_query(term)
                or term.strip().casefold() in _PLACEHOLDER_TERMS
                or _normalize_for_match(term) not in _normalize_for_match(query)
                for term in terms
            )
        ):
            return None
        clean_query = query.strip()
        seen_queries.add(clean_query)
        requirements.append(
            PlanRequirement(
                id=expected_id,
                query=clean_query,
                required_terms=tuple(term.strip() for term in terms),
            )
        )

    return RequestPlan(
        mode=mode,
        goal=goal.strip(),
        clarifying_question=question.strip(),
        response_style=style,
        requirements=tuple(requirements),
    )


def _weather_location_is_missing(
    user_text: str,
    recent_messages: tuple[Message, ...],
) -> bool:
    """Catch one essential local selector the model cannot infer from a device."""
    if "날씨" not in user_text:
        return False
    public_context = " ".join(
        [
            *(
                message.content
                for message in recent_messages[-6:]
                if _role_value(message) != Role.SYSTEM.value
            ),
            user_text,
        ]
    )
    if _KOREAN_PLACE_PATTERN.search(public_context) is not None:
        return False
    return not any(
        match.group(1) not in _NON_PLACE_WEATHER_PREFIXES
        for match in _WEATHER_PREFIX_PATTERN.finditer(public_context)
    )


def _preserve_public_search_language(
    plan: RequestPlan,
    user_text: str,
    recent_messages: tuple[Message, ...],
) -> RequestPlan:
    """Keep explicit Korean public places in local current-weather searches."""
    if plan.mode != "search":
        return plan
    public_context = " ".join(
        [
            *(
                message.content
                for message in recent_messages[-6:]
                if _role_value(message) != Role.SYSTEM.value
            ),
            user_text,
        ]
    )
    if "날씨" not in public_context:
        return plan
    places = tuple(dict.fromkeys(_KOREAN_PLACE_PATTERN.findall(public_context)))[-2:]
    if not places or not plan.requirements:
        return plan
    query = f"{' '.join(places)} 현재 날씨 기온 강수"
    terms = (*places, "날씨")[-3:]
    requirement = replace(
        plan.requirements[0],
        query=query,
        required_terms=terms,
    )
    return replace(plan, requirements=(requirement, *plan.requirements[1:]))


class RequestPlanner:
    """Use the configured local model to make a strict generic request plan."""

    def __init__(self, engine: Any, model: str) -> None:
        self._engine = engine
        self._model = model

    def plan(
        self,
        user_text: str,
        recent_messages: tuple[Message, ...],
    ) -> RequestPlan | None:
        if _weather_location_is_missing(user_text, recent_messages):
            return RequestPlan(
                mode="clarify",
                goal="오늘 날씨 알려주기",
                clarifying_question="어느 지역의 날씨를 볼까?",
                response_style="direct",
                requirements=(),
            )
        planner_input = _planner_input(user_text, recent_messages)
        try:
            response = self._engine.generate(
                [
                    Message(role=Role.SYSTEM, content=_PLAN_SYSTEM_PROMPT),
                    Message(role=Role.USER, content=planner_input),
                ],
                model=self._model,
                temperature=0.0,
                max_tokens=512,
            )
        except Exception:  # noqa: BLE001 - local planning fails closed
            return None
        raw_content = _content(response)
        parsed = _parse_plan(raw_content)
        if parsed is not None:
            return _preserve_public_search_language(
                parsed,
                user_text,
                recent_messages,
            )
        repaired = self._repair(raw_content, planner_input)
        if repaired is None:
            return None
        return _preserve_public_search_language(
            repaired,
            user_text,
            recent_messages,
        )

    def _repair(self, malformed: str, planner_input: str) -> RequestPlan | None:
        try:
            response = self._engine.generate(
                [
                    Message(role=Role.SYSTEM, content=_REPAIR_SYSTEM_PROMPT),
                    Message(
                        role=Role.USER,
                        content=(
                            f"Original request and recent context:\n{planner_input}"
                            f"\n\nMalformed plan:\n{malformed}"
                        ),
                    ),
                ],
                model=self._model,
                temperature=0.0,
                max_tokens=512,
            )
        except Exception:  # noqa: BLE001 - repair is attempted once
            return None
        return _parse_plan(_content(response))


__all__ = [
    "PlanRequirement",
    "RequestPlan",
    "RequestPlanner",
    "is_safe_public_query",
    "needs_assistant_planning",
]
