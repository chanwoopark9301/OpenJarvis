"""Structured evidence parsed from the web-search tool's text output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

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


@dataclass(frozen=True)
class SearchRequirement:
    """One public-information need and the query used to satisfy it."""

    id: str
    kind: str
    query: str
    required_terms: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceSource:
    """One numbered search result that can support itinerary facts."""

    id: str
    requirement_id: str
    title: str
    url: str
    summary: str
    source_kind: str


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def _record_matches(requirement: SearchRequirement, record: EvidenceSource) -> bool:
    searchable = _normalize(f"{record.title} {record.summary} {record.url}")
    return all(_normalize(term) in searchable for term in requirement.required_terms)


def _classify_source(
    requirement: SearchRequirement,
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
    requirement: SearchRequirement,
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
    requirement: SearchRequirement,
    evidence: tuple[EvidenceSource, ...],
) -> bool:
    """Return whether one record satisfies the requirement's public terms."""
    for record in evidence:
        if record.requirement_id != requirement.id or not _record_matches(
            requirement,
            record,
        ):
            continue
        if requirement.kind != "official" or record.source_kind == "official":
            return True
    return False


__all__ = [
    "EvidenceSource",
    "SearchRequirement",
    "parse_search_content",
    "requirement_is_met",
]
