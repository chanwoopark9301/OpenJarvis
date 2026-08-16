"""Tests for structured evidence parsed from web-search output."""

from openjarvis.cli._search_evidence import (
    EvidenceSource,
    SearchRequirement,
    parse_search_content,
    record_is_relevant,
    requirement_is_met,
)


def test_parse_search_content_assigns_stable_ids_and_fields():
    requirement = SearchRequirement(
        id="R1",
        kind="official",
        query="대전 테미오래 공식 운영시간",
        required_terms=("대전", "테미오래"),
    )
    content = (
        "### 테미오래 공식 누리집\n"
        "Source: https://temiorae.com/\n"
        "Summary: 대전 테미오래 관람 안내와 운영시간\n\n---\n\n"
        "### 대전 관광 안내\n"
        "Source: https://example.com/tour\n"
        "Summary: 테미오래 방문 정보"
    )

    evidence = parse_search_content(requirement, content, start_index=4)

    assert evidence == (
        EvidenceSource(
            id="S4",
            requirement_id="R1",
            title="테미오래 공식 누리집",
            url="https://temiorae.com/",
            summary="대전 테미오래 관람 안내와 운영시간",
            source_kind="official",
        ),
        EvidenceSource(
            id="S5",
            requirement_id="R1",
            title="대전 관광 안내",
            url="https://example.com/tour",
            summary="테미오래 방문 정보",
            source_kind="business",
        ),
    )


def test_parse_search_content_ignores_malformed_url_less_and_duplicate_blocks():
    requirement = SearchRequirement(
        id="R2",
        kind="food",
        query="대전 한식 맛집",
        required_terms=("대전", "한식"),
    )
    content = (
        "### 주소 없음\nSummary: 대전 한식\n\n---\n\n"
        "### 첫 결과\nSource: https://example.com/food\nSummary: 대전 한식\n\n"
        "---\n\n"
        "### 중복 결과\nSource: https://example.com/food\nSummary: 다른 설명\n\n"
        "---\n\n"
        "broken block"
    )

    evidence = parse_search_content(requirement, content, start_index=1)

    assert len(evidence) == 1
    assert evidence[0].id == "S1"
    assert evidence[0].title == "첫 결과"


def test_parse_search_content_classifies_listing_review_and_other_sources():
    requirement = SearchRequirement(
        id="R3",
        kind="shops",
        query="대전 소품샵",
        required_terms=("대전", "소품샵"),
    )
    content = (
        "### 대전 소품샵 목록\n"
        "Source: https://www.tripadvisor.com/shops\n"
        "Summary: 대전 소품샵\n\n---\n\n"
        "### 대전 소품샵 후기\n"
        "Source: https://writer.tistory.com/post\n"
        "Summary: 대전 소품샵\n\n---\n\n"
        "### 다른 지역 뉴스\n"
        "Source: https://news.example.org/article\n"
        "Summary: 부산 행사"
    )

    evidence = parse_search_content(requirement, content, start_index=1)

    assert [item.source_kind for item in evidence] == [
        "listing",
        "review",
        "other",
    ]


def test_requirement_is_met_requires_terms_in_one_record_and_official_source():
    requirement = SearchRequirement(
        id="R1",
        kind="official",
        query="대전 테미오래 공식 운영시간",
        required_terms=("대전", "테미오래"),
    )
    split_terms = (
        EvidenceSource("S1", "R1", "대전 관광", "https://a.test", "", "official"),
        EvidenceSource("S2", "R1", "테미오래", "https://b.test", "", "official"),
    )
    unofficial_match = (
        EvidenceSource(
            "S3",
            "R1",
            "대전 테미오래",
            "https://blog.test",
            "운영시간",
            "review",
        ),
    )
    official_match = (
        EvidenceSource(
            "S4",
            "R1",
            "테미오래 공식 누리집",
            "https://temiorae.com",
            "대전 관람 안내",
            "official",
        ),
    )

    assert requirement_is_met(requirement, split_terms) is False
    assert requirement_is_met(requirement, unofficial_match) is False
    assert requirement_is_met(requirement, official_match) is True


def test_nonofficial_requirement_accepts_one_matching_record():
    requirement = SearchRequirement(
        id="R2",
        kind="food",
        query="대전 한식 맛집",
        required_terms=("대전", "한식"),
    )
    evidence = (
        EvidenceSource(
            "S1",
            "R2",
            "대전 맛집 모음",
            "https://example.com",
            "한식 식당 안내",
            "listing",
        ),
    )

    assert requirement_is_met(requirement, evidence) is True


def test_generic_social_profile_is_not_relevant_from_snippet_only():
    requirement = SearchRequirement(
        id="R2",
        kind="food",
        query="대전 테미오래 근처 한식 맛집",
        required_terms=("한식", "맛집"),
    )
    generic_profile = EvidenceSource(
        "S1",
        "R2",
        "늘 하 (@example) - Instagram photos and videos",
        "https://www.instagram.com/example/",
        "대전 테미오래 근처 한식 맛집 검색 결과",
        "review",
    )

    assert record_is_relevant(requirement, generic_profile) is False
