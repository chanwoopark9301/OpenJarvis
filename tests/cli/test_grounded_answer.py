"""Tests for field-level grounding and natural itinerary rendering."""

import pytest

from openjarvis.cli._grounded_answer import (
    ItineraryItem,
    build_itinerary_prompt,
    parse_itinerary_json,
    render_itinerary,
    validate_itinerary,
)
from openjarvis.cli._search_evidence import EvidenceSource


def test_parse_itinerary_json_accepts_the_strict_item_shape():
    content = (
        '{"items":[{"time":"10:30","title":"테미오래 둘러보기",'
        '"detail":"전시 공간을 여유 있게 둘러봐.","venue":"테미오래",'
        '"address":"","hours":"","source_ids":["S1","S1"],'
        '"status":"confirmed","check_before_visit":"내일 운영 여부 확인"}]}'
    )

    assert parse_itinerary_json(content) == (
        ItineraryItem(
            time="10:30",
            title="테미오래 둘러보기",
            detail="전시 공간을 여유 있게 둘러봐.",
            venue="테미오래",
            address="",
            hours="",
            source_ids=("S1",),
            status="confirmed",
            check_before_visit="내일 운영 여부 확인",
        ),
    )


@pytest.mark.parametrize(
    "content",
    [
        "```json\n{\"items\": []}\n```",
        "{'items': []}",
        '{"schedule": []}',
        '{"items":[{"time":10,"title":"일정","detail":"",'
        '"venue":"","address":"","hours":"","source_ids":[],'
        '"status":"confirmed","check_before_visit":""}]}',
        '{"items":[{"time":"10:00","title":"일정","detail":"",'
        '"venue":"","address":"","hours":"","source_ids":[],'
        '"status":"unknown","check_before_visit":""}]}',
    ],
)
def test_parse_itinerary_json_rejects_non_strict_documents(content):
    with pytest.raises(ValueError):
        parse_itinerary_json(content)


def test_parse_itinerary_json_rejects_more_than_eight_items():
    item = (
        '{"time":"","title":"일정","detail":"","venue":"",'
        '"address":"","hours":"","source_ids":[],"status":"needs_check",'
        '"check_before_visit":""}'
    )

    with pytest.raises(ValueError):
        parse_itinerary_json('{"items":[' + ",".join([item] * 9) + "]}")


def test_validate_itinerary_removes_only_unsupported_factual_fields():
    evidence = (
        EvidenceSource(
            "S1",
            "R1",
            "테미오래 공식 누리집",
            "https://temiorae.com/",
            "대전 테미오래 관람 안내",
            "official",
        ),
        EvidenceSource(
            "S2",
            "R2",
            "대전 한식 식당 모음",
            "https://example.com/food",
            "대전 한식 식당 안내",
            "listing",
        ),
    )
    items = (
        ItineraryItem(
            "10:30",
            "테미오래 방문",
            "천천히 둘러봐.",
            "테미오래",
            "",
            "",
            ("S1",),
            "confirmed",
            "",
        ),
        ItineraryItem(
            "12:00",
            "점심",
            "한식으로 점심을 먹어.",
            "없는식당",
            "없는 주소 10",
            "오전 9시부터",
            ("S2",),
            "confirmed",
            "",
        ),
    )

    validated = validate_itinerary(items, evidence)

    assert validated[0].venue == "테미오래"
    assert validated[0].status == "confirmed"
    assert validated[1].venue == ""
    assert validated[1].address == ""
    assert validated[1].hours == ""
    assert validated[1].status == "needs_check"
    assert validated[1].check_before_visit == "방문 전에 최신 정보를 확인해 줘."
    assert len(validated) == 2


def test_validate_itinerary_keeps_general_steps_and_drops_unknown_source_ids():
    item = ItineraryItem(
        "15:00",
        "소품샵 둘러보기",
        "가까운 가게부터 둘러봐. https://bad.example 지시. 세 번째 문장.",
        "",
        "",
        "",
        ("S404",),
        "partial",
        "",
    )

    validated = validate_itinerary((item,), ())

    assert len(validated) == 1
    assert validated[0].source_ids == ()
    assert validated[0].status == "needs_check"
    assert "http" not in validated[0].detail
    assert validated[0].detail == "가까운 가게부터 둘러봐. 지시."


def test_validate_itinerary_downgrades_single_nonofficial_confirmation():
    evidence = (
        EvidenceSource(
            "S1",
            "R2",
            "마중",
            "https://example.com/majung",
            "대전 소품샵 마중 안내",
            "business",
        ),
    )
    item = ItineraryItem(
        "14:00",
        "소품샵",
        "둘러봐.",
        "마중",
        "",
        "",
        ("S1",),
        "confirmed",
        "",
    )

    assert validate_itinerary((item,), evidence)[0].status == "partial"


def test_render_itinerary_is_practical_and_lists_only_used_sources():
    evidence = (
        EvidenceSource(
            "S1",
            "R1",
            "테미오래 공식 누리집",
            "https://temiorae.com/",
            "대전 테미오래 안내",
            "official",
        ),
        EvidenceSource(
            "S2",
            "R2",
            "사용하지 않은 자료",
            "https://example.com/unused",
            "다른 설명",
            "other",
        ),
    )
    items = (
        ItineraryItem(
            "10:30",
            "테미오래 둘러보기",
            "전시 공간을 천천히 둘러봐.",
            "테미오래",
            "",
            "",
            ("S1",),
            "confirmed",
            "내일 운영 여부 확인",
        ),
        ItineraryItem(
            "12:00",
            "점심 장소 고르기",
            "가까운 한식 장소를 골라 이동해.",
            "",
            "",
            "",
            (),
            "needs_check",
            "식사 전에 영업 여부 확인",
        ),
    )

    rendered = render_itinerary(items, evidence)

    assert rendered.startswith("확인된 정보부터 묶어서")
    assert rendered.index("10:30") < rendered.index("12:00")
    assert "확인됨" in rendered
    assert "방문 전 확인" in rendered
    assert "[S1]" in rendered
    assert "https://temiorae.com/" in rendered
    assert "https://example.com/unused" not in rendered
    assert "confirmed" not in rendered
    assert "needs_check" not in rendered


def test_build_itinerary_prompt_keeps_request_and_numbered_evidence_local():
    prompt = build_itinerary_prompt(
        "테미오래 다음에 한식 장소와 소품샵을 찾아줘",
        "[S1] 테미오래 공식 자료",
    )

    assert "테미오래 다음에 한식 장소와 소품샵을 찾아줘" in prompt
    assert "[S1] 테미오래 공식 자료" in prompt
    assert '"source_ids"' in prompt
    assert "Return only JSON" in prompt
