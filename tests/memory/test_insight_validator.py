"""Tests for multi-axis insight validation."""

from __future__ import annotations

from openjarvis.memory.insight_validator import InsightValidator
from openjarvis.memory.personal_models import AdaptationOperation
from openjarvis.memory.reflection import ReflectionProposal


def _proposal(*, confirmation: bool = False) -> ReflectionProposal:
    return ReflectionProposal(
        content="Exercise helps mood when fatigue is manageable.",
        operation=AdaptationOperation.ACCOMMODATE_REFINE,
        scope="exercise_and_mood",
        support_evidence_ids=("e1", "e2", "e3"),
        counter_evidence_ids=("e4",),
        uncertainties=("sleep",),
        requires_user_confirmation=confirmation,
    )


def test_narrow_grounded_insight_can_enter_probation_without_a_sum():
    """A narrow reversible pattern may advance while keeping each quality axis."""
    result = InsightValidator().validate(
        _proposal(),
        available_evidence_ids={"e1", "e2", "e3", "e4"},
        conflict_evidence_ids={"e4"},
        user_relevant=True,
        broad_interpretation=False,
    )

    assert result.provenance_fidelity is True
    assert result.conflict_coverage is True
    assert result.compression_gain is True
    assert result.overgeneralized is False
    assert result.may_enter_probation is True
    assert not hasattr(result, "total_score")


def test_broad_interpretation_waits_for_user_confirmation():
    """Recurrence alone must not confirm a broad personality interpretation."""
    pending = InsightValidator().validate(
        _proposal(confirmation=True),
        available_evidence_ids={"e1", "e2", "e3", "e4"},
        conflict_evidence_ids={"e4"},
        user_relevant=True,
        broad_interpretation=True,
        user_confirmation="pending",
    )
    confirmed = InsightValidator().validate(
        _proposal(confirmation=True),
        available_evidence_ids={"e1", "e2", "e3", "e4"},
        conflict_evidence_ids={"e4"},
        user_relevant=True,
        broad_interpretation=True,
        user_confirmation="confirmed",
    )

    assert pending.may_enter_probation is False
    assert confirmed.may_enter_probation is True


def test_missing_counter_or_invented_provenance_blocks_insight():
    """A proposal must cover conflicts and stay grounded before any accommodation."""
    proposal = ReflectionProposal(
        content="Broad claim",
        operation=AdaptationOperation.ACCOMMODATE_CREATE,
        scope="personality",
        support_evidence_ids=("e1", "invented"),
        counter_evidence_ids=(),
        uncertainties=(),
        requires_user_confirmation=True,
    )

    result = InsightValidator().validate(
        proposal,
        available_evidence_ids={"e1", "e4"},
        conflict_evidence_ids={"e4"},
        user_relevant=True,
        broad_interpretation=True,
    )

    assert result.provenance_fidelity is False
    assert result.conflict_coverage is False
    assert result.may_enter_probation is False
