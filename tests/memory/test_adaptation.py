"""Tests for deterministic assimilation and accommodation gates."""

from __future__ import annotations

from openjarvis.memory.adaptation import (
    RelationProposal,
    SchemaAdaptationEngine,
    source_priority,
)
from openjarvis.memory.personal_models import (
    AdaptationOperation,
    CandidateDraft,
    CandidateKind,
    ClaimState,
    EvidenceMass,
    EvidenceSource,
    PersonalClaim,
    PersonalSchema,
    SchemaMaturity,
)


def _claim(
    claim_id: str,
    content: str,
    *,
    kind: CandidateKind = CandidateKind.PREFERENCE,
    subject_scope: str = "assistant_behavior",
) -> PersonalClaim:
    return PersonalClaim(
        id=claim_id,
        kind=kind,
        content=content,
        state=ClaimState.ACTIVE,
        source=EvidenceSource.USER_DIRECT,
        evidence_ids=("e-old",),
        created_at=1.0,
        updated_at=1.0,
        temporal_scope="persistent",
        subject_scope=subject_scope,
    )


def _schema(*, broad: bool = False) -> PersonalSchema:
    return PersonalSchema(
        id="schema-1",
        content="Exercise improves the user's mood.",
        maturity=SchemaMaturity.EMERGING,
        state=ClaimState.ACTIVE,
        current_version_id="schema-version-1",
        evidence_mass=EvidenceMass(support_mass=0.6, scope_fit=0.8),
        created_at=1.0,
        updated_at=1.0,
        subject_scope="exercise_and_mood",
        broad_interpretation=broad,
    )


def test_new_user_correction_supersedes_its_explicit_target():
    """Removing the hard gate would let an old preference outvote a correction."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.CORRECTION,
            "Do not offer study coaching unless asked.",
            1.0,
            1.0,
            temporal_scope="until_changed",
            subject="",
            target_claim_id="old-preference",
        ),
        active_claims=[_claim("old-preference", "Offer study coaching proactively.")],
        related_schemas=[],
    )

    assert result.operation is AdaptationOperation.ACCOMMODATE_SUPERSEDE
    assert result.superseded_claim_ids == ("old-preference",)
    assert result.requires_user_confirmation is False


def test_correction_without_target_supersedes_active_direct_claims_in_subject():
    """Removing the kind gate could make any same-subject statement destructive."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.CORRECTION,
            "The assistant name is 공박사.",
            1.0,
            1.0,
            subject="assistant.name",
        ),
        active_claims=[
            _claim(
                "old-name",
                "The assistant name is 조비서.",
                kind=CandidateKind.ROLE_PREFERENCE,
                subject_scope="assistant.name",
            ),
            _claim(
                "other-subject",
                "Keep replies concise.",
                kind=CandidateKind.ROLE_PREFERENCE,
                subject_scope="response_style",
            ),
            _claim(
                "same-subject-fact",
                "The user coined an assistant nickname.",
                kind=CandidateKind.FACT,
                subject_scope="assistant.name",
            ),
        ],
        related_schemas=[],
    )

    assert result.operation is AdaptationOperation.ACCOMMODATE_SUPERSEDE
    assert result.superseded_claim_ids == ("old-name",)


def test_correction_with_generic_subject_requires_explicit_target():
    """The fallback subject sentinel must never select unrelated direct rules."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.CORRECTION,
            "The assistant name is 공박사.",
            1.0,
            1.0,
            subject="",
        ),
        active_claims=[
            _claim(
                "generic-name",
                "The assistant name is 조비서.",
                kind=CandidateKind.ROLE_PREFERENCE,
                subject_scope="user",
            ),
            _claim(
                "generic-style",
                "Keep replies concise.",
                kind=CandidateKind.ROLE_PREFERENCE,
                subject_scope="user",
            ),
        ],
        related_schemas=[],
    )

    assert result.operation is AdaptationOperation.ACCOMMODATE_CREATE
    assert result.superseded_claim_ids == ()


def test_invalid_explicit_target_does_not_fall_through_to_subject_match():
    """A supplied target remains authoritative even when it cannot be selected."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.CORRECTION,
            "The assistant name is 공박사.",
            1.0,
            1.0,
            subject="assistant.name",
            target_claim_id="missing-target",
        ),
        active_claims=[
            _claim(
                "old-name",
                "The assistant name is 조비서.",
                kind=CandidateKind.ROLE_PREFERENCE,
                subject_scope="assistant.name",
            )
        ],
        related_schemas=[],
    )

    assert result.operation is AdaptationOperation.ACCOMMODATE_CREATE
    assert result.superseded_claim_ids == ()


def test_non_correction_with_same_subject_does_not_implicitly_supersede():
    """Correction-like wording cannot replace the model's typed correction label."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.ROLE_PREFERENCE,
            "Correction: the assistant name is 공박사.",
            1.0,
            1.0,
            subject="assistant.name",
        ),
        active_claims=[
            _claim(
                "old-name",
                "The assistant name is 조비서.",
                kind=CandidateKind.ROLE_PREFERENCE,
                subject_scope="assistant.name",
            )
        ],
        related_schemas=[],
    )

    assert result.operation is AdaptationOperation.ACCOMMODATE_CREATE
    assert result.superseded_claim_ids == ()


def test_unrelated_generic_direct_rules_do_not_supersede_each_other():
    """A generic extractor subject must not erase a different direct rule."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.CONSTRAINT,
            "Do not mention timers.",
            1.0,
            1.0,
            subject="user",
        ),
        active_claims=[
            _claim(
                "exam-rule",
                "Do not mention exams.",
                kind=CandidateKind.CONSTRAINT,
                subject_scope="user",
            )
        ],
        related_schemas=[],
    )

    assert result.operation is AdaptationOperation.ACCOMMODATE_CREATE
    assert result.superseded_claim_ids == ()


def test_shared_broad_subject_does_not_supersede_without_explicit_target():
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.ROLE_PREFERENCE,
            "Use a warm tone.",
            1.0,
            1.0,
            subject="assistant_behavior",
        ),
        active_claims=[
            _claim(
                "length-rule",
                "Keep replies concise.",
                kind=CandidateKind.ROLE_PREFERENCE,
            )
        ],
        related_schemas=[],
    )

    assert result.operation is AdaptationOperation.ACCOMMODATE_CREATE
    assert result.superseded_claim_ids == ()


def test_external_candidate_cannot_change_a_personal_schema():
    """External knowledge must remain a hypothesis, never personal evidence."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.HYPOTHESIS,
            "Fatigue can reduce concentration.",
            0.7,
            0.7,
            source=EvidenceSource.EXTERNAL,
        ),
        active_claims=[],
        related_schemas=[_schema()],
        relation=RelationProposal.SUPPORTS,
    )

    assert result.operation is AdaptationOperation.NO_OP
    assert result.reason_code == "source_not_personal_evidence"


def test_supporting_evidence_is_assimilated_without_rewriting_the_schema():
    """Support should reinforce an existing version instead of inventing a new one."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.EPISODE,
            "Running improved my mood today.",
            0.7,
            0.9,
            temporal_scope="current",
            subject="exercise_and_mood",
        ),
        active_claims=[],
        related_schemas=[_schema()],
        relation=RelationProposal.SUPPORTS,
    )

    assert result.operation is AdaptationOperation.ASSIMILATE_REINFORCE
    assert result.target_schema_ids == ("schema-1",)


def test_conditional_exception_does_not_destroy_a_supported_schema():
    """A tired day should be represented as an exception, not a personality reversal."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.EPISODE,
            "I was too tired to enjoy running today.",
            0.7,
            0.9,
            temporal_scope="current",
            subject="exercise_and_mood",
        ),
        active_claims=[],
        related_schemas=[_schema()],
        relation=RelationProposal.CONDITIONAL_EXCEPTION,
    )

    assert result.operation is AdaptationOperation.ASSIMILATE_AS_EXCEPTION


def test_ambiguous_or_contradictory_evidence_is_held_as_disequilibrium():
    """Unclear conflict must not silently mutate the current schema."""
    for relation in (RelationProposal.AMBIGUOUS, RelationProposal.CONTRADICTS):
        result = SchemaAdaptationEngine().decide(
            candidate=CandidateDraft(
                CandidateKind.EPISODE,
                "I did not feel better after running.",
                0.6,
                0.8,
                subject="exercise_and_mood",
            ),
            active_claims=[],
            related_schemas=[_schema()],
            relation=relation,
        )

        assert result.operation is AdaptationOperation.HOLD_DISEQUILIBRIUM
        assert result.target_schema_ids == ("schema-1",)


def test_new_narrow_pattern_starts_tentative():
    """One experience must not become a stable personal pattern."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.EPISODE,
            "I enjoyed a run today.",
            0.5,
            0.8,
            subject="exercise_and_mood",
        ),
        active_claims=[],
        related_schemas=[],
    )

    assert result.operation is AdaptationOperation.ACCOMMODATE_CREATE
    assert result.initial_maturity is SchemaMaturity.TENTATIVE


def test_different_subject_scope_cannot_reinforce_an_unrelated_schema():
    """A classifier support label must not override deterministic scope mismatch."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.FACT,
            "I concentrate better after sleeping well.",
            0.7,
            0.9,
            subject="sleep_and_concentration",
        ),
        active_claims=[],
        related_schemas=[_schema()],
        relation=RelationProposal.SUPPORTS,
    )

    assert result.operation is AdaptationOperation.ACCOMMODATE_CREATE
    assert result.reason_code == "scope_mismatch"


def test_source_priority_preserves_provenance_order_without_a_sum():
    """Changing provenance order could let legacy imports outweigh confirmation."""
    assert source_priority(EvidenceSource.USER_CONFIRMED) > source_priority(
        EvidenceSource.USER_DIRECT
    )
    assert source_priority(EvidenceSource.USER_DIRECT) > source_priority(
        EvidenceSource.SOCIAL_TESTIMONY
    )
    assert source_priority(EvidenceSource.SOCIAL_TESTIMONY) > source_priority(
        EvidenceSource.LEGACY_IMPORT
    )
    assert source_priority(EvidenceSource.EXTERNAL) == 0


def test_broad_personality_interpretation_requires_user_confirmation():
    """Broad identity claims must not become stable through recurrence alone."""
    result = SchemaAdaptationEngine().decide(
        candidate=CandidateDraft(
            CandidateKind.HYPOTHESIS,
            "The user is fundamentally achievement-driven.",
            0.8,
            0.6,
            subject="broad_personality",
        ),
        active_claims=[],
        related_schemas=[],
    )

    assert result.operation is AdaptationOperation.ACCOMMODATE_CREATE
    assert result.requires_user_confirmation is True
    assert result.initial_maturity is SchemaMaturity.TENTATIVE
