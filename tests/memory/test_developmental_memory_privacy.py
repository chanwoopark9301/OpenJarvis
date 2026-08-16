"""Privacy-boundary acceptance checks for personal memory."""

from __future__ import annotations

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.social_reflection import (
    KnowledgeResult,
    SocialReflectionCoordinator,
)


def test_external_summary_stays_outside_personal_claims_and_context(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    coordinator = SocialReflectionCoordinator(archive, mode="allowed")

    result = coordinator.store_external_result(
        request_id="r1",
        provider_id="intercepted-provider",
        query="general factors affecting fatigue",
        consent_id="consent-1",
        result=KnowledgeResult(
            source_url="https://example.test/research",
            summary="Fatigue may have several general causes.",
        ),
    )

    assert result.kind == "external_knowledge"
    assert archive.get_active_claims() == []
    assert archive.get_active_schemas() == []
