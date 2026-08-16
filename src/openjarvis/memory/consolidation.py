"""Evidence limits and eligibility gates for personal schema consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openjarvis.memory.personal_models import (
    DIRECT_RULE_KINDS,
    CandidateKind,
    EvidenceSource,
)


@dataclass(frozen=True, slots=True)
class ConsolidationEvidence:
    """Raw evidence ranking fields kept separate rather than summed."""

    id: str
    session_id: str
    source: EvidenceSource
    source_quality: float
    scope_fit: float
    conflict_value: float
    created_at: float
    is_raw_user_evidence: bool
    semantically_related: bool


@dataclass(frozen=True, slots=True)
class ConsolidationEligibility:
    """Whether bounded evidence may enter a schema proposal job."""

    eligible: bool
    reason_code: str


class SchemaConsolidationGate:
    """Require recurrence and raw grounding before derived schema creation."""

    def __init__(self, *, max_evidence: int = 12) -> None:
        self._max_evidence = max(1, int(max_evidence))

    def evaluate(
        self,
        evidence: Sequence[ConsolidationEvidence],
        *,
        candidate_kind: CandidateKind | str | None = None,
    ) -> ConsolidationEligibility:
        if (
            candidate_kind is not None
            and CandidateKind(candidate_kind) in DIRECT_RULE_KINDS
        ):
            return ConsolidationEligibility(True, "direct_rule_bypass")
        if len([item for item in evidence if item.is_raw_user_evidence]) < 3:
            return ConsolidationEligibility(False, "insufficient_raw_evidence")
        related = [
            item
            for item in evidence
            if item.is_raw_user_evidence and item.semantically_related
        ]
        if len(related) < 3:
            return ConsolidationEligibility(False, "insufficient_related_evidence")
        if len({item.session_id for item in related if item.session_id}) < 2:
            return ConsolidationEligibility(False, "insufficient_session_recurrence")
        return ConsolidationEligibility(True, "sufficient_cross_session_evidence")

    def select(
        self,
        evidence: Sequence[ConsolidationEvidence],
    ) -> tuple[ConsolidationEvidence, ...]:
        """Apply lexicographic evidence ordering with a fixed payload cap."""
        raw = [item for item in evidence if item.is_raw_user_evidence]
        ordered = sorted(
            raw,
            key=lambda item: (
                item.source_quality,
                item.scope_fit,
                item.conflict_value,
                item.created_at,
                item.id,
            ),
            reverse=True,
        )
        return tuple(ordered[: self._max_evidence])


__all__ = [
    "ConsolidationEligibility",
    "ConsolidationEvidence",
    "SchemaConsolidationGate",
]
