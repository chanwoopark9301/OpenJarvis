"""Independent validation axes for reflection insight proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

from openjarvis.memory.reflection import ReflectionProposal


@dataclass(frozen=True, slots=True)
class InsightValidationResult:
    """Separate quality judgments that are never collapsed into one score."""

    provenance_fidelity: bool
    conflict_coverage: bool
    predictive_utility: str
    user_relevance: bool
    compression_gain: bool
    overgeneralized: bool
    user_confirmation: str
    may_enter_probation: bool


class InsightValidator:
    """Gate schema accommodation using provenance, conflict, scope, and consent."""

    def validate(
        self,
        proposal: ReflectionProposal,
        *,
        available_evidence_ids: Set[str] | set[str],
        conflict_evidence_ids: Set[str] | set[str],
        user_relevant: bool,
        broad_interpretation: bool,
        user_confirmation: str = "pending",
    ) -> InsightValidationResult:
        cited = set(proposal.support_evidence_ids) | set(
            proposal.counter_evidence_ids
        )
        available = set(available_evidence_ids)
        conflicts = set(conflict_evidence_ids)
        provenance_fidelity = bool(cited) and cited <= available
        conflict_coverage = not conflicts or conflicts <= set(
            proposal.counter_evidence_ids
        )
        compression_gain = len(set(proposal.support_evidence_ids)) >= 3
        overgeneralized = broad_interpretation and not (
            proposal.requires_user_confirmation
        )
        confirmation_ready = (
            not broad_interpretation
            and not proposal.requires_user_confirmation
        ) or user_confirmation == "confirmed"
        may_enter_probation = all(
            (
                provenance_fidelity,
                conflict_coverage,
                bool(user_relevant),
                compression_gain,
                not overgeneralized,
                confirmation_ready,
            )
        )
        return InsightValidationResult(
            provenance_fidelity=provenance_fidelity,
            conflict_coverage=conflict_coverage,
            predictive_utility="unknown",
            user_relevance=bool(user_relevant),
            compression_gain=compression_gain,
            overgeneralized=overgeneralized,
            user_confirmation=user_confirmation,
            may_enter_probation=may_enter_probation,
        )


__all__ = ["InsightValidationResult", "InsightValidator"]
