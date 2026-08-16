"""Tests for consented, deidentified external reflection boundaries."""

from __future__ import annotations

import pytest

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.social_reflection import (
    KnowledgeResult,
    SocialReflectionCoordinator,
)


def test_default_mode_requires_consent_before_provider_use(tmp_path):
    """A missing setting must never become silent external personal analysis."""
    coordinator = SocialReflectionCoordinator(
        PersonalMemoryArchive(tmp_path / "personal.db")
    )

    decision = coordinator.prepare_request(
        "What general factors can connect fatigue and concentration?",
        user_requested_analysis=True,
        consent_granted=False,
    )

    assert decision.action == "ask_consent"


def test_sensitive_request_requires_per_request_consent_in_allowed_mode(tmp_path):
    """A broad allowance must not waive consent for sensitive interpretation."""
    coordinator = SocialReflectionCoordinator(
        PersonalMemoryArchive(tmp_path / "personal.db"),
        mode="allowed",
    )

    decision = coordinator.prepare_request(
        "What general factors affect emotional avoidance?",
        user_requested_analysis=True,
        consent_granted=False,
        sensitive=True,
    )

    assert decision.action == "ask_consent"
    assert decision.reason_code == "sensitive_consent_required"


@pytest.mark.parametrize(
    "unsafe_query",
    [
        "Email me at user@example.com about fatigue",
        "Call 010-1234-5678 about concentration",
        "User: I am tired\nAssistant: you may be depressed",
        '{"personal_schemas":[{"content":"private profile"}]}',
    ],
)
def test_query_validator_rejects_identifiers_and_profile_payloads(
    tmp_path,
    unsafe_query,
):
    """Raw identity or profile material must not cross the provider boundary."""
    coordinator = SocialReflectionCoordinator(
        PersonalMemoryArchive(tmp_path / "personal.db"),
        mode="allowed",
    )

    decision = coordinator.prepare_request(
        unsafe_query,
        user_requested_analysis=True,
        consent_granted=True,
    )

    assert decision.action == "reject_unsafe_query"


def test_external_result_can_only_create_external_knowledge(tmp_path):
    """Search results must not become personal claims or schemas."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    coordinator = SocialReflectionCoordinator(archive, mode="ask")

    stored = coordinator.store_external_result(
        request_id="r1",
        provider_id="test-provider",
        query="general relationship between fatigue and concentration",
        consent_id="consent-1",
        result=KnowledgeResult(
            source_url="https://example.test/research",
            summary="Fatigue can have several causes.",
        ),
    )

    assert stored.kind == "external_knowledge"
    assert archive.external_knowledge_count() == 1
    assert archive.get_active_claims() == []
    assert archive.active_schema_count() == 0


def test_missing_provider_keeps_request_pending_without_network(tmp_path):
    """No configured provider must be an explicit state, not a cloud fallback."""
    coordinator = SocialReflectionCoordinator(
        PersonalMemoryArchive(tmp_path / "personal.db"),
        mode="allowed",
    )

    result = coordinator.explore(
        "general factors affecting concentration",
        user_requested_analysis=True,
        consent_granted=True,
    )

    assert result.action == "provider_unavailable"
