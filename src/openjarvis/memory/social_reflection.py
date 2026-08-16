"""Consent and deidentification boundary for optional outside knowledge."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Protocol, Sequence

from openjarvis.memory.archive import PersonalMemoryArchive

_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_PHONE_RE = re.compile(r"(?:\+?\d[\d -]{7,}\d)")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_SAFE_CAPITALIZED_QUERY_WORDS = frozenset({"General", "How", "What", "Why"})
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z가-힣0-9-]+\s+(?:street|st|road|rd|avenue|ave|길|로)\b",
    re.IGNORECASE,
)
_RAW_PROFILE_MARKERS = (
    "user:",
    "assistant:",
    "conversation_exchanges",
    "personal_schemas",
    "personal_claims",
)


@dataclass(frozen=True, slots=True)
class KnowledgeResult:
    """One general source result returned by an optional provider."""

    source_url: str
    summary: str


class ExternalKnowledgeProvider(Protocol):
    """Provider boundary implemented by a later network integration."""

    provider_id: str

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> Sequence[KnowledgeResult]: ...


@dataclass(frozen=True, slots=True)
class ExternalAction:
    """Safe state returned before any optional provider call."""

    action: str
    reason_code: str
    query: str = ""
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class StoredExternalKnowledge:
    """Reference to knowledge that remains outside the personal model."""

    id: str
    kind: str = "external_knowledge"


class SocialReflectionCoordinator:
    """Keep outside learning consented, deidentified, and non-personal."""

    def __init__(
        self,
        archive: PersonalMemoryArchive,
        *,
        mode: str = "ask",
        provider: ExternalKnowledgeProvider | None = None,
    ) -> None:
        if mode not in {"off", "ask", "allowed"}:
            raise ValueError("external mode must be off, ask, or allowed")
        self._archive = archive
        self._mode = mode
        self._provider = provider

    def prepare_request(
        self,
        query: str,
        *,
        user_requested_analysis: bool,
        consent_granted: bool,
        sensitive: bool = False,
    ) -> ExternalAction:
        clean_query = " ".join(query.split())
        if not user_requested_analysis:
            return ExternalAction("hold_local", "analysis_not_requested")
        if self._mode == "off":
            return ExternalAction("hold_local", "external_reflection_disabled")
        if not self._is_safe_query(clean_query):
            return ExternalAction("reject_unsafe_query", "query_not_deidentified")
        if sensitive and not consent_granted:
            return ExternalAction("ask_consent", "sensitive_consent_required")
        if self._mode == "ask" and not consent_granted:
            return ExternalAction("ask_consent", "external_consent_required")
        return ExternalAction(
            "ready",
            "deidentified_request_ready",
            clean_query,
            str(uuid.uuid4()),
        )

    def explore(
        self,
        query: str,
        *,
        user_requested_analysis: bool,
        consent_granted: bool,
        sensitive: bool = False,
        consent_id: str = "",
    ) -> ExternalAction:
        action = self.prepare_request(
            query,
            user_requested_analysis=user_requested_analysis,
            consent_granted=consent_granted,
            sensitive=sensitive,
        )
        if action.action != "ready":
            return action
        if self._provider is None:
            return ExternalAction(
                "provider_unavailable",
                "external_provider_not_configured",
                action.query,
                action.request_id,
            )
        for result in self._provider.search(action.query, max_results=5):
            self.store_external_result(
                request_id=action.request_id,
                provider_id=self._provider.provider_id,
                query=action.query,
                consent_id=consent_id,
                result=result,
            )
        return ExternalAction(
            "complete",
            "external_knowledge_stored",
            action.query,
            action.request_id,
        )

    def store_external_result(
        self,
        *,
        request_id: str,
        provider_id: str,
        query: str,
        consent_id: str,
        result: KnowledgeResult,
    ) -> StoredExternalKnowledge:
        if not self._is_safe_query(query):
            raise ValueError("external query is not deidentified")
        item_id = self._archive.store_external_knowledge(
            request_id=request_id,
            provider_id=provider_id,
            source_url=result.source_url,
            query_text=query,
            summary=result.summary,
            consent_id=consent_id,
            deidentified=True,
        )
        return StoredExternalKnowledge(item_id)

    @staticmethod
    def _is_safe_query(query: str) -> bool:
        normalized = query.casefold()
        return not (
            not query.strip()
            or _NON_ASCII_RE.search(query)
            or any(
                token not in _SAFE_CAPITALIZED_QUERY_WORDS
                for token in _PROPER_NAME_RE.findall(query)
            )
            or _EMAIL_RE.search(query)
            or _PHONE_RE.search(query)
            or _ADDRESS_RE.search(query)
            or any(marker in normalized for marker in _RAW_PROFILE_MARKERS)
        )


__all__ = [
    "ExternalAction",
    "ExternalKnowledgeProvider",
    "KnowledgeResult",
    "SocialReflectionCoordinator",
    "StoredExternalKnowledge",
]
