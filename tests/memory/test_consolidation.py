"""Tests for evidence-gated schema consolidation."""

from __future__ import annotations

from openjarvis.memory.consolidation import (
    ConsolidationEvidence,
    SchemaConsolidationGate,
)
from openjarvis.memory.personal_models import CandidateKind, EvidenceSource


def _evidence(
    index: int,
    session_id: str,
    *,
    raw: bool = True,
    related: bool = True,
) -> ConsolidationEvidence:
    return ConsolidationEvidence(
        id=f"e{index}",
        session_id=session_id,
        source=EvidenceSource.USER_DIRECT,
        source_quality=0.9,
        scope_fit=0.8,
        conflict_value=0.2,
        created_at=float(index),
        is_raw_user_evidence=raw,
        semantically_related=related,
    )


def test_three_related_raw_items_across_two_sessions_are_eligible():
    """A derived pattern needs recurrence beyond one conversation."""
    result = SchemaConsolidationGate().evaluate(
        [_evidence(1, "a"), _evidence(2, "a"), _evidence(3, "b")]
    )

    assert result.eligible is True
    assert result.reason_code == "sufficient_cross_session_evidence"


def test_same_session_or_unrelated_items_do_not_consolidate():
    """Volume inside one chat must not masquerade as a stable pattern."""
    gate = SchemaConsolidationGate()

    same_session = gate.evaluate(
        [_evidence(1, "a"), _evidence(2, "a"), _evidence(3, "a")]
    )
    unrelated = gate.evaluate(
        [
            _evidence(1, "a"),
            _evidence(2, "b"),
            _evidence(3, "c", related=False),
        ]
    )

    assert same_session.eligible is False
    assert unrelated.eligible is False


def test_schema_summary_cannot_become_evidence_for_another_schema():
    """Recursive summaries would amplify prior model errors without raw grounding."""
    result = SchemaConsolidationGate().evaluate(
        [_evidence(1, "a"), _evidence(2, "b"), _evidence(3, "c", raw=False)]
    )

    assert result.eligible is False
    assert result.reason_code == "insufficient_raw_evidence"


def test_direct_rule_bypasses_recurrence_gate():
    """One explicit ban must be usable immediately."""
    result = SchemaConsolidationGate().evaluate(
        [_evidence(1, "a")],
        candidate_kind=CandidateKind.CONSTRAINT,
    )

    assert result.eligible is True
    assert result.reason_code == "direct_rule_bypass"


def test_selection_is_capped_at_twelve_without_an_aggregate_score():
    """A reflection job must never grow with the complete archive."""
    evidence = [_evidence(index, f"s{index % 3}") for index in range(20)]

    selected = SchemaConsolidationGate(max_evidence=12).select(evidence)

    assert len(selected) == 12
    assert [item.id for item in selected[:3]] == ["e19", "e18", "e17"]
