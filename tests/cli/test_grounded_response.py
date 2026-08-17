"""Tests for generic, block-level grounded search answers."""

from __future__ import annotations

import pytest

from openjarvis.cli._grounded_response import (
    GroundedResponse,
    ResponseBlock,
    Support,
    build_grounded_prompt,
    parse_grounded_json,
    render_grounded_response,
    repair_grounded_json,
    validate_grounded_response,
)
from openjarvis.cli._search_evidence import EvidenceSource
from openjarvis.core.types import Message


class _FakeEngine:
    def __init__(self, *contents: str):
        self.contents = list(contents)
        self.calls: list[list[Message]] = []

    def generate(self, messages, **kwargs):
        self.calls.append(list(messages))
        return {"content": self.contents.pop(0)}


def _evidence() -> tuple[EvidenceSource, ...]:
    return (
        EvidenceSource(
            "S1",
            "R1",
            "과천시 오늘 날씨",
            "https://weather.example/gwacheon",
            "현재 기온 24도, 저녁에는 흐림",
            "official",
        ),
        EvidenceSource(
            "S2",
            "R2",
            "무설탕 탄산음료 비교",
            "https://example.com/drinks",
            "라임맛 제품은 한 병에 1500원",
            "listing",
        ),
    )


def test_parse_grounded_json_accepts_strict_shape_and_deduplicates_supports():
    content = (
        '{"lead":"과천시 날씨를 확인했어.","blocks":['
        '{"kind":"fact","text":"현재 기온은 24도야.","supports":['
        '{"source_id":"S1","excerpt":"현재 기온 24도"},'
        '{"source_id":"S1","excerpt":"현재 기온 24도"}]},'
        '{"kind":"advice","text":"얇은 겉옷을 챙겨.","supports":[]}],'
        '"follow_up":""}'
    )

    assert parse_grounded_json(content) == GroundedResponse(
        lead="과천시 날씨를 확인했어.",
        blocks=(
            ResponseBlock(
                kind="fact",
                text="현재 기온은 24도야.",
                supports=(Support("S1", "현재 기온 24도"),),
            ),
            ResponseBlock(kind="advice", text="얇은 겉옷을 챙겨.", supports=()),
        ),
        follow_up="",
    )


@pytest.mark.parametrize(
    "content",
    [
        "```json\n{\"lead\":\"\",\"blocks\":[],\"follow_up\":\"\"}\n```",
        "{'lead':'','blocks':[],'follow_up':''}",
        '{"items":[]}',
        '{"lead":"답","blocks":[],"follow_up":"","extra":true}',
        '{"lead":"답","blocks":[{"kind":"claim","text":"내용",'
        '"supports":[]}],"follow_up":""}',
        '{"lead":"답","blocks":[{"kind":"fact","text":"내용",'
        '"supports":[{"source_id":"S1","excerpt":"a"},'
        '{"source_id":"S1","excerpt":"b"},'
        '{"source_id":"S1","excerpt":"c"},'
        '{"source_id":"S1","excerpt":"d"}]}],"follow_up":""}',
    ],
)
def test_parse_grounded_json_rejects_non_strict_documents(content):
    with pytest.raises(ValueError):
        parse_grounded_json(content)


def test_parse_grounded_json_rejects_more_than_eight_blocks():
    block = '{"kind":"advice","text":"한 걸음씩 해.","supports":[]}'

    with pytest.raises(ValueError):
        parse_grounded_json(
            '{"lead":"답","blocks":[' + ",".join([block] * 9) + '],"follow_up":""}'
        )


def test_parser_removes_ideographs_from_visible_text():
    response = parse_grounded_json(
        '{"lead":"확인 後 알려줄게","blocks":[{"kind":"advice",'
        '"text":"운동 後 물을 마셔.","supports":[]}],"follow_up":"再 확인할까?"}'
    )

    visible = response.lead + response.blocks[0].text + response.follow_up
    assert "後" not in visible
    assert "再" not in visible


def test_validation_keeps_exactly_supported_fact_and_matching_number():
    response = GroundedResponse(
        lead="확인했어.",
        blocks=(
            ResponseBlock(
                "fact", "현재 기온은 24도야.", (Support("S1", "현재 기온 24도"),)
            ),
        ),
        follow_up="",
    )

    assert validate_grounded_response(response, _evidence()).blocks == response.blocks


def test_validation_removes_only_bad_fact_and_keeps_good_fact_and_advice():
    response = GroundedResponse(
        lead="확인했어.",
        blocks=(
            ResponseBlock(
                "fact", "현재 기온은 24도야.", (Support("S1", "현재 기온 24도"),)
            ),
            ResponseBlock(
                "fact", "현재 기온은 29도야.", (Support("S1", "현재 기온 24도"),)
            ),
            ResponseBlock(
                "fact", "비가 와.", (Support("S404", "비가 와"),)
            ),
            ResponseBlock("advice", "얇은 겉옷을 챙겨.", ()),
        ),
        follow_up="",
    )

    validated = validate_grounded_response(response, _evidence())

    assert [block.text for block in validated.blocks] == [
        "현재 기온은 24도야.",
        "얇은 겉옷을 챙겨.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "https://bad.example에서 확인해.",
        "010-1234-5678로 전화해.",
        "가격은 1500원이야.",
        "오후 8시 24분에 나가.",
        "중앙로 10번길로 가.",
        "주문했어.",
        "예약을 마쳤어.",
    ],
)
def test_validation_removes_risky_uncited_advice(text):
    response = GroundedResponse(
        lead="정리했어.",
        blocks=(ResponseBlock("advice", text, ()),),
        follow_up="",
    )

    assert validate_grounded_response(response, _evidence()).blocks == ()


def test_validation_replaces_risky_lead_and_follow_up():
    response = GroundedResponse(
        lead="현재 29도고 주문도 끝냈어.",
        blocks=(ResponseBlock("advice", "물을 챙겨.", ()),),
        follow_up="1500원짜리도 살까?",
    )

    validated = validate_grounded_response(response, _evidence())

    assert validated.lead == "확인한 내용을 정리했어."
    assert validated.follow_up == ""


@pytest.mark.parametrize("style", ["direct", "list", "comparison", "steps"])
def test_renderer_uses_requested_generic_style_and_only_cited_sources(style):
    response = GroundedResponse(
        lead="확인했어.",
        blocks=(
            ResponseBlock(
                "fact", "현재 기온은 24도야.", (Support("S1", "현재 기온 24도"),)
            ),
            ResponseBlock("advice", "얇은 겉옷을 챙겨.", ()),
        ),
        follow_up="",
    )

    rendered = render_grounded_response(response, _evidence(), style=style)

    assert "현재 기온은 24도야." in rendered
    assert "https://weather.example/gwacheon" in rendered
    assert "https://example.com/drinks" not in rendered
    assert "방문 전 확인" not in rendered
    assert "일정으로 정리" not in rendered
    if style == "steps":
        assert "1. 현재 기온은 24도야." in rendered
    elif style in {"list", "comparison"}:
        assert "- 현재 기온은 24도야." in rendered


def test_prompt_contains_generic_schema_request_goal_style_and_evidence():
    prompt = build_grounded_prompt(
        user_text="과천시 날씨를 알려줘",
        goal="과천시 현재 날씨 안내",
        response_style="direct",
        context="[S1] 현재 기온 24도",
    )

    assert "과천시 날씨를 알려줘" in prompt
    assert "과천시 현재 날씨 안내" in prompt
    assert "direct" in prompt
    assert "[S1] 현재 기온 24도" in prompt
    assert '"lead"' in prompt and '"blocks"' in prompt
    assert '"excerpt"' in prompt
    assert "itinerary" not in prompt.casefold()
    assert "venue" not in prompt.casefold()


def test_repair_is_attempted_once_and_limits_valid_source_ids():
    engine = _FakeEngine(
        '{"lead":"고쳤어.","blocks":[{"kind":"fact",'
        '"text":"현재 기온은 24도야.","supports":['
        '{"source_id":"S1","excerpt":"현재 기온 24도"}]}],"follow_up":""}'
    )

    repaired = repair_grounded_json(
        engine,
        "test-model",
        "not json",
        valid_source_ids=("S1",),
    )

    assert repaired is not None
    assert len(engine.calls) == 1
    prompt = engine.calls[0][-1].content
    assert "S1" in prompt
    assert "S404" not in prompt
