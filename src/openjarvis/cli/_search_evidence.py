"""Structured evidence parsed from the web-search tool's text output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from openjarvis.cli._request_plan import PlanRequirement

_LISTING_DOMAINS = (
    "diningcode.com",
    "siksinhot.com",
    "trip.com",
    "tripadvisor.com",
)
_REVIEW_DOMAINS = (
    "blog.naver.com",
    "m.blog.naver.com",
    "instagram.com",
    "tiktok.com",
    "tistory.com",
    "youtube.com",
)
_KOREAN_REGION_NAMES = frozenset(
    {
        "서울",
        "부산",
        "대구",
        "인천",
        "광주",
        "대전",
        "울산",
        "세종",
        "경기",
        "강원",
        "충북",
        "충남",
        "전북",
        "전남",
        "경북",
        "경남",
        "제주",
        "서울시",
        "부산시",
        "대구시",
        "인천시",
        "광주시",
        "대전시",
        "울산시",
        "세종시",
        "경기도",
        "강원도",
        "충청북도",
        "충청남도",
        "전라북도",
        "전라남도",
        "경상북도",
        "경상남도",
        "제주도",
    }
)
_NEARBY_WORDS = frozenset({"근처", "인근", "주변"})


@dataclass(frozen=True)
class EvidenceSource:
    """One numbered search result that can support an everyday factual claim."""

    id: str
    requirement_id: str
    title: str
    url: str
    summary: str
    source_kind: str


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def _record_matches(requirement: PlanRequirement, record: EvidenceSource) -> bool:
    searchable = _normalize(f"{record.title} {record.summary} {record.url}")
    return all(_normalize(term) in searchable for term in requirement.required_terms)


def _query_place_anchors(query: str) -> tuple[str, ...]:
    tokens = re.findall(r"[가-힣]{2,}", query)
    anchors: list[str] = [token for token in tokens if token in _KOREAN_REGION_NAMES]
    anchors.extend(
        token
        for token in tokens
        if len(token) >= 3 and token.endswith(("시", "군", "구", "도"))
    )
    for index, token in enumerate(tokens):
        if token in _NEARBY_WORDS and index:
            anchors.append(tokens[index - 1])
    return tuple(dict.fromkeys(anchors))


def record_is_relevant(
    requirement: PlanRequirement,
    record: EvidenceSource,
) -> bool:
    """Return whether a result mentions at least one requested public term."""
    searchable = _normalize(f"{record.title} {record.summary} {record.url}")
    term_matches = all(
        normalized_term in searchable
        for term in requirement.required_terms
        if (normalized_term := _normalize(term))
    )
    anchors = _query_place_anchors(requirement.query)
    anchor_matches = not anchors or any(
        _normalize(anchor) in searchable for anchor in anchors
    )
    hostname = (urlparse(record.url).hostname or "").casefold()
    social_domains = ("instagram.com", "tiktok.com", "youtube.com")
    if any(domain in hostname for domain in social_domains):
        title_text = _normalize(record.title)
        title_matches = any(
            _normalize(value) in title_text
            for value in (*requirement.required_terms, *anchors)
            if _normalize(value)
        )
        return term_matches and anchor_matches and title_matches
    return term_matches and anchor_matches


def _classify_source(
    requirement: PlanRequirement,
    *,
    title: str,
    url: str,
    summary: str,
) -> str:
    hostname = (urlparse(url).hostname or "").casefold()
    combined = f"{title} {summary}"
    if hostname.endswith(".go.kr") or "공식" in combined or "누리집" in combined:
        return "official"
    if any(domain in hostname for domain in _LISTING_DOMAINS):
        return "listing"
    if any(domain in hostname for domain in _REVIEW_DOMAINS):
        return "review"
    searchable = _normalize(f"{combined} {url}")
    if all(_normalize(term) in searchable for term in requirement.required_terms):
        return "business"
    return "other"


def parse_search_content(
    requirement: PlanRequirement,
    content: str,
    *,
    start_index: int,
) -> tuple[EvidenceSource, ...]:
    """Parse valid, unique search result blocks into stable evidence IDs."""
    evidence: list[EvidenceSource] = []
    seen_urls: set[str] = set()
    for block in re.split(r"\n\s*---\s*\n", content):
        title_match = re.search(r"^###\s+(.+)$", block, flags=re.MULTILINE)
        url_match = re.search(r"^Source:\s*(https?://\S+)$", block, re.MULTILINE)
        summary_match = re.search(r"^Summary:\s*(.*)$", block, re.MULTILINE)
        if title_match is None or url_match is None or summary_match is None:
            continue
        title = " ".join(title_match.group(1).split())
        url = url_match.group(1).rstrip(".,;:!?'\"")
        summary = " ".join(summary_match.group(1).split())
        if url in seen_urls:
            continue
        seen_urls.add(url)
        evidence.append(
            EvidenceSource(
                id=f"S{start_index + len(evidence)}",
                requirement_id=requirement.id,
                title=title,
                url=url,
                summary=summary,
                source_kind=_classify_source(
                    requirement,
                    title=title,
                    url=url,
                    summary=summary,
                ),
            )
        )
    return tuple(evidence)


def requirement_is_met(
    requirement: PlanRequirement,
    evidence: tuple[EvidenceSource, ...],
) -> bool:
    """Return whether one record satisfies the requirement's public terms."""
    for record in evidence:
        if record.requirement_id != requirement.id or not _record_matches(
            requirement,
            record,
        ):
            continue
        return True
    return False


__all__ = [
    "EvidenceSource",
    "parse_search_content",
    "record_is_relevant",
    "requirement_is_met",
]
