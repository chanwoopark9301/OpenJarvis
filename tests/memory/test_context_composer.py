"""Tests for safe, bounded developmental memory context composition."""

from __future__ import annotations

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.context_composer import (
    ContextComposer,
    compose_configured_personal_context,
)
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import (
    AdaptationOperation,
    CandidateDraft,
    CandidateKind,
)


def _accept_claim(
    archive: PersonalMemoryArchive,
    *,
    index: int,
    kind: CandidateKind,
    content: str,
    temporal_scope: str = "persistent",
    subject: str = "user",
):
    exchange_id = f"exchange-{index}"
    archive.record_exchange(
        exchange_id=exchange_id,
        user_text=content,
        assistant_text="Acknowledged.",
        source="test",
        session_id=f"session-{index}",
    )
    assert archive.claim_candidate_job(exchange_id) is not None
    candidate = archive.complete_candidate_job(
        exchange_id,
        [
            CandidateDraft(
                kind,
                content,
                1.0,
                1.0,
                temporal_scope=temporal_scope,
                subject=subject,
            )
        ],
        engine_id="ollama",
        extractor_version="v2",
    )[0]
    assert MemoryEvaluator(archive).evaluate(candidate.id).applied is True
    return archive.get_active_claims()[0]


def test_context_budget_and_direct_rule_order(tmp_path):
    """Archive size must not enlarge the prompt or bury direct constraints."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    for index in range(7):
        _accept_claim(
            archive,
            index=index,
            kind=CandidateKind.CONSTRAINT,
            content=f"Do not proactively mention topic {index}.",
            temporal_scope="until_changed",
            subject=f"topic:{index}",
        )

    context = ContextComposer(archive, max_constraints=5).compose("How are you?")

    assert len(context.constraints) == 5
    assert context.sections[0].name == "direct_constraints"
    assert context.sections[0].items == context.constraints


def test_current_state_is_labeled_and_not_generalized(tmp_path):
    """A temporary state must be kept out of confirmed personal patterns."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    _accept_claim(
        archive,
        index=1,
        kind=CandidateKind.FACT,
        content="I feel tired lately.",
        temporal_scope="current",
        subject="energy",
    )

    context = ContextComposer(archive).compose("Why can I not focus?")

    assert context.current_states == ("I feel tired lately.",)
    assert context.schemas == ()
    assert "CURRENT USER STATE" in context.render()
    assert "do not generalize" in context.render()


def test_external_knowledge_never_enters_personal_context(tmp_path):
    """General search material must not be rendered as a claim about the user."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.store_external_knowledge(
        request_id="r1",
        provider_id="test",
        source_url="https://example.test",
        query_text="general fatigue factors",
        summary="Fatigue can affect concentration.",
        consent_id="c1",
        deidentified=True,
    )

    context = ContextComposer(archive).compose("I cannot focus")

    assert "Fatigue can affect concentration" not in context.render()


def test_latest_unevaluated_text_is_a_user_overlay_not_a_system_rule(tmp_path):
    """Immediate continuity must not elevate raw text into authoritative memory."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="pending",
        user_text="I might want to run tomorrow.",
        assistant_text="You are definitely a runner.",
        source="test",
    )

    context = ContextComposer(archive).compose("What did I just say?")

    assert context.user_overlay == "I might want to run tomorrow."
    assert "definitely a runner" not in context.render()
    assert "definitely a runner" not in context.user_overlay


def test_configured_context_refuses_non_local_response_engine(tmp_path):
    from openjarvis.core.config import JarvisConfig

    config = JarvisConfig()
    config.personal_memory.enabled = True
    config.personal_memory.archive_path = str(tmp_path / "personal.db")

    assert (
        compose_configured_personal_context(
            config,
            "hello",
            engine_key="openai",
        )
        is None
    )


def test_configured_context_allows_loopback_response_engine(tmp_path):
    from openjarvis.core.config import JarvisConfig

    config = JarvisConfig()
    config.personal_memory.enabled = True
    config.personal_memory.mode = "active"
    config.personal_memory.archive_path = str(tmp_path / "personal.db")

    context = compose_configured_personal_context(
        config,
        "hello",
        engine_key="ollama",
    )

    assert context is not None


def test_shadow_mode_never_injects_personal_context(tmp_path):
    from openjarvis.core.config import JarvisConfig

    config = JarvisConfig()
    config.personal_memory.enabled = True
    config.personal_memory.mode = "shadow"
    config.personal_memory.archive_path = str(tmp_path / "personal.db")

    assert (
        compose_configured_personal_context(
            config,
            "hello",
            engine_key="ollama",
        )
        is None
    )
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    assert archive.shadow_metrics()["compositions"] == 1


def test_confirmed_schema_is_rendered_with_conditions(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    for index in range(3):
        _accept_claim(
            archive,
            index=index,
            kind=CandidateKind.FACT,
            content=f"Running experience {index} improved my mood.",
        )
    evidence_ids = tuple(
        evidence_id
        for claim in archive.get_active_claims()
        for evidence_id in claim.evidence_ids
    )
    schema = archive.apply_schema_accommodation(
        content="Running often improves my mood.",
        operation=AdaptationOperation.ACCOMMODATE_CREATE,
        support_evidence_ids=evidence_ids,
        conditions=("when fatigue is manageable",),
        subject_scope="running_and_mood",
        user_confirmed=True,
    )
    assert schema is not None

    context = ContextComposer(archive).compose("How has running felt?")

    assert context.schemas == (
        "Running often improves my mood. (conditions: when fatigue is manageable)",
    )
