"""Behavior tests for developmental personal-memory value objects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import openjarvis.memory.personal_models as models


def test_candidate_kinds_cover_direct_rules_and_provisional_hypotheses():
    """Dropping a candidate kind would make a required memory class unrepresentable."""
    assert {kind.value for kind in models.CandidateKind} == {
        "fact",
        "episode",
        "preference",
        "constraint",
        "correction",
        "role_preference",
        "capability_boundary",
        "hypothesis",
    }


@pytest.mark.parametrize(
    "kind",
    [
        "constraint",
        "correction",
        "role_preference",
        "capability_boundary",
    ],
)
def test_direct_rule_candidate_requires_direct_or_confirmed_user_evidence(kind):
    """Assistant or external text must never become an authoritative user rule."""
    with pytest.raises(ValueError, match="direct user evidence"):
        models.CandidateDraft(
            kind=models.CandidateKind(kind),
            content="Do not mention timers.",
            importance=1.0,
            confidence=1.0,
            source=models.EvidenceSource.ASSISTANT,
        )


def test_existing_string_candidate_api_remains_compatible():
    """The existing extractor may keep passing string kinds during migration."""
    draft = models.CandidateDraft("fact", "User prefers short answers.", 0.8, 0.9)

    assert draft.kind is models.CandidateKind.FACT
    assert draft.kind == "fact"
    assert draft.source is models.EvidenceSource.USER_DIRECT


def test_candidate_keeps_exact_user_evidence_excerpt():
    """Dropping the source wording would make proposal review unauditable."""
    draft = models.CandidateDraft(
        "correction",
        "The assistant name is 공박사.",
        1,
        1,
        subject="assistant.name",
        evidence_excerpt="공박사라고.",
    )

    assert draft.evidence_excerpt == "공박사라고."


def test_evidence_mass_keeps_axes_separate_and_immutable():
    """A future summed score must not replace independent support and counter mass."""
    mass = models.EvidenceMass(
        support_mass=0.8,
        counter_mass=0.3,
        stability=0.6,
        plasticity=0.4,
        scope_fit=0.9,
        source_quality=1.0,
        temporal_validity=0.7,
    )

    assert mass.support_mass == 0.8
    assert mass.counter_mass == 0.3
    assert not hasattr(mass, "total_score")
    with pytest.raises(FrozenInstanceError):
        mass.support_mass = 0.1  # type: ignore[misc]


def test_schema_maturity_keeps_challenged_records_out_of_stable_state():
    """A challenged schema must remain a distinct state instead of silently decaying."""
    assert {state.value for state in models.SchemaMaturity} == {
        "tentative",
        "emerging",
        "stable_candidate",
        "stable",
        "challenged",
        "retired",
    }


def test_adaptation_operations_form_a_closed_decision_contract():
    """Unknown model-proposed actions must not reach persistence code."""
    assert {operation.value for operation in models.AdaptationOperation} == {
        "assimilate_reinforce",
        "assimilate_link_only",
        "assimilate_as_exception",
        "hold_disequilibrium",
        "accommodate_refine",
        "accommodate_split",
        "accommodate_supersede",
        "accommodate_create",
        "no_op",
    }


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_evidence_mass_rejects_values_outside_its_documented_range(value):
    """Out-of-range model output must not distort later adaptation comparisons."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        models.EvidenceMass(support_mass=value)
