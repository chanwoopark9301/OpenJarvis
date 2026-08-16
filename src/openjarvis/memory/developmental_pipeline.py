"""One-shot consolidation, reflection, and validation job handlers."""

from __future__ import annotations

import hashlib
import json
from typing import Callable

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.consolidation import (
    ConsolidationEvidence,
    SchemaConsolidationGate,
)
from openjarvis.memory.insight_validator import InsightValidator
from openjarvis.memory.personal_models import (
    DIRECT_RULE_KINDS,
    EvidenceSource,
    MemoryJob,
)
from openjarvis.memory.reflection import ReflectionEngine, ReflectionProposal

CHECK_CONSOLIDATION = "check_consolidation"
REFLECT_CONFLICTS = "reflect_conflicts"
VALIDATE_INSIGHT = "validate_insight"


class DevelopmentalMemoryPipeline:
    """Turn recurring raw evidence into validated schema proposals sequentially."""

    def __init__(
        self,
        archive: PersonalMemoryArchive,
        reflection_engine: ReflectionEngine,
        *,
        max_evidence: int = 12,
    ) -> None:
        self.archive = archive
        self.reflection_engine = reflection_engine
        self.gate = SchemaConsolidationGate(max_evidence=max_evidence)
        self.validator = InsightValidator()

    def handlers(self) -> dict[str, Callable[[MemoryJob], None]]:
        """Return the closed durable job dispatch table."""
        return {
            CHECK_CONSOLIDATION: self.check_consolidation,
            REFLECT_CONFLICTS: self.reflect,
            VALIDATE_INSIGHT: self.validate_insight,
        }

    def check_consolidation(self, job: MemoryJob) -> None:
        """Schedule reflection only after recurrent cross-session evidence."""
        candidate = self.archive.get_candidate(job.subject_id)
        if candidate is None or candidate.kind in DIRECT_RULE_KINDS:
            return
        evidence = self.archive.evidence_items_for_subject(candidate.subject, limit=100)
        ranked = [
            ConsolidationEvidence(
                id=item.id,
                session_id=item.session_id,
                source=item.source,
                source_quality=self._source_quality(item.source),
                scope_fit=1.0,
                conflict_value=0.0,
                created_at=item.created_at,
                is_raw_user_evidence=True,
                semantically_related=True,
            )
            for item in evidence
        ]
        if not self.gate.evaluate(ranked, candidate_kind=candidate.kind).eligible:
            return
        selected = self.gate.select(ranked)
        evidence_ids = [item.id for item in selected]
        fingerprint = hashlib.sha256("\x00".join(evidence_ids).encode()).hexdigest()
        self.archive.enqueue_job(
            job_type=REFLECT_CONFLICTS,
            subject_id=candidate.subject,
            idempotency_key=f"{REFLECT_CONFLICTS}:{candidate.subject}:{fingerprint}",
            payload={"evidence_ids": evidence_ids, "subject": candidate.subject},
            priority=25,
        )

    def reflect(self, job: MemoryJob) -> None:
        """Persist at most one evidence-bounded proposal, then schedule validation."""
        payload = json.loads(job.payload_json)
        evidence_ids = tuple(payload.get("evidence_ids", ()))[:12]
        evidence = self.archive.evidence_items_by_id(evidence_ids)
        proposal = self.reflection_engine.reflect(
            evidence={item.id: item.content for item in evidence},
            conflict_evidence_ids=set(),
        )
        if proposal is None:
            return
        insight = self.archive.store_insight_candidate(
            content=proposal.content,
            operation=proposal.operation,
            support_evidence_ids=proposal.support_evidence_ids,
            counter_evidence_ids=proposal.counter_evidence_ids,
            uncertainties=proposal.uncertainties,
            requires_user_confirmation=proposal.requires_user_confirmation,
        )
        subject = str(payload.get("subject", "user"))
        self.archive.enqueue_job(
            job_type=VALIDATE_INSIGHT,
            subject_id=insight.id,
            idempotency_key=f"{VALIDATE_INSIGHT}:{insight.id}",
            payload={"subject": subject},
            priority=20,
        )

    def validate_insight(self, job: MemoryJob) -> None:
        """Apply narrow grounded proposals; leave confirmation-required ones pending."""
        insight = self.archive.get_insight_candidate(job.subject_id)
        if insight is None:
            return
        payload = json.loads(job.payload_json)
        subject = str(payload.get("subject", "user"))
        proposal = ReflectionProposal(
            content=insight.content,
            operation=insight.operation,
            scope=subject,
            support_evidence_ids=insight.support_evidence_ids,
            counter_evidence_ids=insight.counter_evidence_ids,
            uncertainties=insight.uncertainties,
            requires_user_confirmation=insight.requires_user_confirmation,
        )
        available_ids = {
            item.id
            for item in self.archive.evidence_items_by_id(
                (*insight.support_evidence_ids, *insight.counter_evidence_ids)
            )
        }
        broad = subject in {
            "broad_personality",
            "identity",
            "mental_health",
            "health",
            "relationship_motive",
        }
        schema = self.validator.accommodate(
            self.archive,
            proposal,
            available_evidence_ids=available_ids,
            conflict_evidence_ids=set(insight.counter_evidence_ids),
            user_relevant=True,
            broad_interpretation=broad,
        )
        if schema is not None:
            self.archive.set_insight_state(insight.id, "probation")
        elif not insight.requires_user_confirmation:
            self.archive.set_insight_state(insight.id, "rejected")

    @staticmethod
    def _source_quality(source: EvidenceSource) -> float:
        return {
            EvidenceSource.USER_CONFIRMED: 1.0,
            EvidenceSource.USER_DIRECT: 0.9,
            EvidenceSource.SOCIAL_TESTIMONY: 0.6,
        }.get(source, 0.0)


__all__ = ["DevelopmentalMemoryPipeline"]
