"""One-shot consolidation, reflection, and validation job handlers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.conflicts import ConflictMonitor, ConflictObservation
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
from openjarvis.memory.understanding_gap import QuestionCoordinator

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
        self.conflict_monitor = ConflictMonitor()
        self.validator = InsightValidator()
        self.questions = QuestionCoordinator(archive)
        self.max_evidence = max(1, int(max_evidence))

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
        conflict_rows = self.archive.unresolved_conflicts_for_subject(
            candidate.subject
        )
        conflict_ids = {str(row["evidence_id"]) for row in conflict_rows}
        ranked = [
            ConsolidationEvidence(
                id=item.id,
                session_id=item.session_id,
                source=item.source,
                source_quality=self._source_quality(item.source),
                scope_fit=1.0,
                conflict_value=1.0 if item.id in conflict_ids else 0.0,
                created_at=item.created_at,
                is_raw_user_evidence=True,
                semantically_related=self._lexically_related(
                    candidate.content,
                    item.content,
                ),
            )
            for item in evidence
        ]
        recurrence = self.gate.evaluate(
            ranked,
            candidate_kind=candidate.kind,
        ).eligible
        conflict_trigger = self.conflict_monitor.should_reflect(
            [
                ConflictObservation(
                    schema_id=str(row["schema_id"]),
                    session_id=str(row["session_id"]),
                    prediction_error=float(row["prediction_error"]),
                    created_at=float(row["created_at"]),
                )
                for row in conflict_rows
            ]
        ).triggered
        if not recurrence and not conflict_trigger:
            return
        selected = self.gate.select(ranked)
        evidence_ids = list(
            dict.fromkeys(
                [
                    *(item.id for item in selected if item.id in conflict_ids),
                    *(item.id for item in selected),
                ]
            )
        )[: self.max_evidence]
        fingerprint = hashlib.sha256("\x00".join(evidence_ids).encode()).hexdigest()
        self.archive.enqueue_job(
            job_type=REFLECT_CONFLICTS,
            subject_id=candidate.subject,
            idempotency_key=f"{REFLECT_CONFLICTS}:{candidate.subject}:{fingerprint}",
            payload={
                "evidence_ids": evidence_ids,
                "conflict_evidence_ids": [
                    value for value in evidence_ids if value in conflict_ids
                ],
                "subject": candidate.subject,
            },
            priority=25,
        )

    def reflect(self, job: MemoryJob) -> None:
        """Persist at most one evidence-bounded proposal, then schedule validation."""
        payload = json.loads(job.payload_json)
        evidence_ids = tuple(payload.get("evidence_ids", ()))[:12]
        conflict_evidence_ids = set(payload.get("conflict_evidence_ids", ()))
        evidence = self.archive.evidence_items_by_id(evidence_ids)
        proposal = self.reflection_engine.reflect(
            evidence={item.id: item.content for item in evidence},
            conflict_evidence_ids=conflict_evidence_ids,
        )
        if proposal is None:
            return
        insight = self.archive.store_insight_candidate(
            content=proposal.content,
            scope=proposal.scope,
            operation=proposal.operation,
            support_evidence_ids=proposal.support_evidence_ids,
            counter_evidence_ids=proposal.counter_evidence_ids,
            uncertainties=proposal.uncertainties,
            requires_user_confirmation=proposal.requires_user_confirmation,
        )
        subject = insight.scope or str(payload.get("subject", "user"))
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
        payload_subject = str(payload.get("subject", "user"))
        subject = (
            payload_subject
            if insight.scope == "user" and payload_subject != "user"
            else insight.scope
        )
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
        broad = self._is_broad_scope(subject)
        needs_confirmation = broad or insight.requires_user_confirmation
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
        elif needs_confirmation:
            self.questions.create(
                subject_id=insight.id,
                question="Does this proposed personal pattern fit your experience?",
                decision_effect="confirm or reject the proposed personal pattern",
            )
        else:
            self.archive.set_insight_state(insight.id, "rejected")

    def answer_question(
        self,
        question_id: str,
        answer_text: str,
        *,
        confirms: bool,
    ):
        """Apply an explicit answer to the insight that created a question."""
        question = self.archive.get_pending_question(question_id)
        if question is None:
            raise KeyError(f"unknown question: {question_id}")
        if str(question["state"]) not in {"pending", "unresolved"}:
            raise KeyError(f"completed question: {question_id}")
        insight = self.archive.get_insight_candidate(str(question["subject_id"]))
        if insight is None:
            raise KeyError("question is not linked to an insight")
        evidence = self.archive.record_user_confirmed_evidence(
            user_text=answer_text,
            content=answer_text,
            subject=insight.scope,
        )
        answered = self.questions.answer(
            question_id,
            answer_text,
            answer_evidence_id=evidence.id,
        )
        if answered.state != "answered":
            return answered
        if confirms is False:
            self.archive.set_insight_state(insight.id, "rejected")
            return answered
        proposal = ReflectionProposal(
            content=insight.content,
            operation=insight.operation,
            scope=insight.scope,
            support_evidence_ids=tuple(
                dict.fromkeys((*insight.support_evidence_ids, evidence.id))
            ),
            counter_evidence_ids=insight.counter_evidence_ids,
            uncertainties=insight.uncertainties,
            requires_user_confirmation=True,
        )
        available_ids = {
            item.id
            for item in self.archive.evidence_items_by_id(
                (*proposal.support_evidence_ids, *proposal.counter_evidence_ids)
            )
        }
        schema = self.validator.accommodate(
            self.archive,
            proposal,
            available_evidence_ids=available_ids,
            conflict_evidence_ids=set(proposal.counter_evidence_ids),
            user_relevant=True,
            broad_interpretation=self._is_broad_scope(insight.scope),
            user_confirmation="confirmed",
        )
        if schema is not None:
            self.archive.set_insight_state(insight.id, "probation")
        else:
            self.archive.set_insight_state(insight.id, "rejected")
        return answered

    @staticmethod
    def _is_broad_scope(subject: str) -> bool:
        return subject in {
            "broad_personality",
            "identity",
            "mental_health",
            "health",
            "relationship_motive",
        }

    @staticmethod
    def _lexically_related(left: str, right: str) -> bool:
        """Require shared content words instead of trusting only a broad subject tag."""
        def tokens(value: str) -> set[str]:
            return {
                token
                for token in re.findall(r"[\w가-힣]+", value.casefold())
                if len(token) >= 3
            }

        left_tokens = tokens(left)
        right_tokens = tokens(right)
        if not left_tokens or not right_tokens:
            return False
        return len(left_tokens & right_tokens) / min(
            len(left_tokens), len(right_tokens)
        ) >= 0.25

    @staticmethod
    def _source_quality(source: EvidenceSource) -> float:
        return {
            EvidenceSource.USER_CONFIRMED: 1.0,
            EvidenceSource.USER_DIRECT: 0.9,
            EvidenceSource.SOCIAL_TESTIMONY: 0.6,
        }.get(source, 0.0)


__all__ = ["DevelopmentalMemoryPipeline"]
