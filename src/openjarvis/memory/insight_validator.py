"""Independent validation axes for reflection insight proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Set

from openjarvis.memory.reflection import ReflectionProposal

if TYPE_CHECKING:
    from openjarvis.memory.archive import PersonalMemoryArchive
    from openjarvis.memory.personal_models import PersonalSchema


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

    def accommodate(
        self,
        archive: PersonalMemoryArchive,
        proposal: ReflectionProposal,
        *,
        available_evidence_ids: Set[str] | set[str],
        conflict_evidence_ids: Set[str] | set[str],
        user_relevant: bool,
        broad_interpretation: bool,
        user_confirmation: str = "pending",
        target_schema_id: str = "",
    ) -> PersonalSchema | None:
        """Persist one validated accommodation or leave all schema state unchanged."""
        result = self.validate(
            proposal,
            available_evidence_ids=available_evidence_ids,
            conflict_evidence_ids=conflict_evidence_ids,
            user_relevant=user_relevant,
            broad_interpretation=broad_interpretation,
            user_confirmation=user_confirmation,
        )
        if not result.may_enter_probation:
            return None
        return archive.apply_schema_accommodation(
            content=proposal.content,
            operation=proposal.operation,
            support_evidence_ids=proposal.support_evidence_ids,
            counter_evidence_ids=proposal.counter_evidence_ids,
            conditions=proposal.uncertainties,
            subject_scope=proposal.scope,
            broad_interpretation=broad_interpretation,
            user_confirmed=user_confirmation == "confirmed",
            target_schema_id=target_schema_id,
        )


__all__ = ["InsightValidationResult", "InsightValidator"]
