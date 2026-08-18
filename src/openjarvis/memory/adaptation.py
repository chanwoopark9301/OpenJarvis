"""Deterministic gates for developmental personal-memory adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from openjarvis.memory.personal_models import (
    DIRECT_RULE_KINDS,
    AdaptationOperation,
    CandidateDraft,
    CandidateKind,
    EvidenceSource,
    PersonalClaim,
    PersonalSchema,
    SchemaMaturity,
)


class RelationProposal(str, Enum):
    """Bounded relation labels a local classifier may propose."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONDITIONAL_EXCEPTION = "conditional_exception"
    UNRELATED = "unrelated"
    AMBIGUOUS = "ambiguous"


_PERSONAL_EVIDENCE_SOURCES = frozenset(
    {
        EvidenceSource.USER_DIRECT,
        EvidenceSource.USER_CONFIRMED,
        EvidenceSource.SOCIAL_TESTIMONY,
        EvidenceSource.LEGACY_IMPORT,
    }
)
_BROAD_SUBJECT_MARKERS = frozenset(
    {
        "broad_personality",
        "identity",
        "mental_health",
        "health",
        "relationship_motive",
    }
)
_SOURCE_PRIORITY = {
    EvidenceSource.USER_CONFIRMED: 4,
    EvidenceSource.USER_DIRECT: 3,
    EvidenceSource.SOCIAL_TESTIMONY: 2,
    EvidenceSource.LEGACY_IMPORT: 1,
    EvidenceSource.ASSISTANT: 0,
    EvidenceSource.EXTERNAL: 0,
}


def source_priority(source: EvidenceSource | str) -> int:
    """Return provenance rank for deterministic ordering, never aggregation."""
    try:
        resolved = EvidenceSource(source)
    except ValueError:
        return 0
    return _SOURCE_PRIORITY[resolved]


@dataclass(frozen=True, slots=True)
class AdaptationDecision:
    """Auditable instruction returned to persistence code."""

    operation: AdaptationOperation
    reason_code: str
    superseded_claim_ids: tuple[str, ...] = ()
    target_schema_ids: tuple[str, ...] = ()
    initial_maturity: SchemaMaturity = SchemaMaturity.TENTATIVE
    requires_user_confirmation: bool = False

    @classmethod
    def no_op(cls, reason_code: str) -> AdaptationDecision:
        return cls(AdaptationOperation.NO_OP, reason_code)


class SchemaAdaptationEngine:
    """Apply provenance and direct-rule gates before relation proposals."""

    def decide(
        self,
        *,
        candidate: CandidateDraft,
        active_claims: Sequence[PersonalClaim],
        related_schemas: Sequence[PersonalSchema],
        relation: RelationProposal | str | None = None,
    ) -> AdaptationDecision:
        """Return one closed operation without mutating persistence state."""
        if candidate.source not in _PERSONAL_EVIDENCE_SOURCES:
            return AdaptationDecision.no_op("source_not_personal_evidence")
        if candidate.kind in DIRECT_RULE_KINDS:
            return self._apply_direct_rule(candidate, active_claims)

        if not related_schemas:
            return AdaptationDecision(
                AdaptationOperation.ACCOMMODATE_CREATE,
                "no_related_schema",
                requires_user_confirmation=self._requires_confirmation(candidate),
            )

        scoped_schemas = tuple(
            schema
            for schema in related_schemas
            if schema.subject_scope in {"", "user", candidate.subject}
        )
        if not scoped_schemas:
            return AdaptationDecision(
                AdaptationOperation.ACCOMMODATE_CREATE,
                "scope_mismatch",
                requires_user_confirmation=self._requires_confirmation(candidate),
            )

        target_ids = tuple(schema.id for schema in scoped_schemas)
        try:
            proposed = RelationProposal(relation or RelationProposal.AMBIGUOUS)
        except ValueError:
            proposed = RelationProposal.AMBIGUOUS

        if proposed is RelationProposal.SUPPORTS:
            return AdaptationDecision(
                AdaptationOperation.ASSIMILATE_REINFORCE,
                "supporting_evidence",
                target_schema_ids=target_ids,
            )
        if proposed is RelationProposal.CONDITIONAL_EXCEPTION:
            return AdaptationDecision(
                AdaptationOperation.ASSIMILATE_AS_EXCEPTION,
                "conditional_exception",
                target_schema_ids=target_ids,
            )
        if proposed in {
            RelationProposal.CONTRADICTS,
            RelationProposal.AMBIGUOUS,
        }:
            return AdaptationDecision(
                AdaptationOperation.HOLD_DISEQUILIBRIUM,
                "contradictory_evidence"
                if proposed is RelationProposal.CONTRADICTS
                else "ambiguous_relation",
                target_schema_ids=target_ids,
            )
        return AdaptationDecision(
            AdaptationOperation.ACCOMMODATE_CREATE,
            "unrelated_to_existing_schemas",
            requires_user_confirmation=self._requires_confirmation(candidate),
        )

    @staticmethod
    def _apply_direct_rule(
        candidate: CandidateDraft,
        active_claims: Sequence[PersonalClaim],
    ) -> AdaptationDecision:
        explicit_target = candidate.target_claim_id
        if explicit_target:
            superseded = tuple(
                claim.id
                for claim in active_claims
                if claim.state.value == "active" and claim.id == explicit_target
            )
        elif candidate.kind is CandidateKind.CORRECTION:
            superseded = tuple(
                claim.id
                for claim in active_claims
                if claim.state.value == "active"
                and claim.kind in DIRECT_RULE_KINDS
                and claim.subject_scope == candidate.subject
            )
        else:
            superseded = ()
        if superseded:
            return AdaptationDecision(
                AdaptationOperation.ACCOMMODATE_SUPERSEDE,
                "direct_user_rule_supersedes_active_claim",
                superseded_claim_ids=superseded,
            )
        return AdaptationDecision(
            AdaptationOperation.ACCOMMODATE_CREATE,
            "new_direct_user_rule",
        )

    @staticmethod
    def _requires_confirmation(candidate: CandidateDraft) -> bool:
        return (
            candidate.kind is CandidateKind.HYPOTHESIS
            and candidate.subject in _BROAD_SUBJECT_MARKERS
        )


__all__ = [
    "AdaptationDecision",
    "RelationProposal",
    "SchemaAdaptationEngine",
    "source_priority",
]
