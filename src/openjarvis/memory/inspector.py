"""User-facing inspection and control service for personal memory."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import (
    AdaptationOperation,
    CandidateDraft,
    CandidateKind,
    EvidenceSource,
    PersonalClaim,
    PersonalSchema,
)

DELETE_ALL_CONFIRMATION_TOKEN = "DELETE ALL PERSONAL MEMORY"


@dataclass(frozen=True, slots=True)
class MemorySubjectView:
    """A safe read model containing one subject and its own evidence only."""

    id: str
    subject_type: str
    content: str
    state: str
    maturity: str
    conditions: tuple[str, ...]
    created_at: float
    updated_at: float
    supporting_evidence: tuple[str, ...]
    opposing_evidence: tuple[str, ...] = ()
    previous_versions: tuple[str, ...] = ()
    last_use_reason: str = ""


class PersonalMemoryInspector:
    """Expose bounded reads and explicit, immediately effective mutations."""

    def __init__(self, archive: PersonalMemoryArchive) -> None:
        self.archive = archive

    def list_subjects(self) -> tuple[MemorySubjectView, ...]:
        """List claims and schemas without unrelated archive records."""
        claims = tuple(
            self._claim_view(claim, include_evidence=False)
            for claim in self.archive.list_claims()
        )
        schemas = tuple(
            self._schema_view(schema, include_evidence=False)
            for schema in self.archive.list_schemas()
        )
        return tuple(
            sorted((*claims, *schemas), key=lambda item: item.updated_at, reverse=True)
        )

    def explain(self, subject_id: str) -> MemorySubjectView | None:
        """Explain one claim using only evidence linked to that claim."""
        claim = self.archive.get_claim(subject_id)
        if claim is not None:
            return self._claim_view(claim, include_evidence=True)
        schema = self.archive.get_schema(subject_id)
        return self._schema_view(schema, include_evidence=True) if schema else None

    def confirm(self, subject_id: str, user_text: str) -> PersonalClaim | None:
        """Confirm a claim by superseding it with user-confirmed evidence."""
        claim = self.archive.get_claim(subject_id)
        if claim is not None:
            return self.correct(subject_id, claim.content, user_text)
        schema = self.archive.get_schema(subject_id)
        if schema is not None:
            return self.correct(subject_id, schema.content, user_text)
        return None

    def correct(
        self,
        subject_id: str,
        replacement_text: str,
        user_text: str,
    ) -> PersonalClaim | PersonalSchema | None:
        """Create a new confirmed correction; never rewrite prior evidence."""
        original = self.archive.get_claim(subject_id)
        replacement = replacement_text.strip()
        if not replacement or not user_text.strip():
            return None
        if original is None:
            schema = self.archive.get_schema(subject_id)
            if schema is None:
                return None
            evidence = self.archive.record_user_confirmed_evidence(
                user_text=user_text,
                content=replacement,
                subject=schema.subject_scope,
            )
            support_ids = tuple(
                dict.fromkeys(
                    (*self.archive.schema_evidence_ids(schema.id), evidence.id)
                )
            )
            return self.archive.apply_schema_accommodation(
                content=replacement,
                operation=AdaptationOperation.ACCOMMODATE_REFINE,
                support_evidence_ids=support_ids,
                conditions=schema.conditions,
                subject_scope=schema.subject_scope,
                broad_interpretation=schema.broad_interpretation,
                user_confirmed=True,
                target_schema_id=schema.id,
            )
        exchange_id = str(uuid.uuid4())
        self.archive.record_exchange(
            exchange_id=exchange_id,
            user_text=user_text,
            assistant_text="",
            source="user_memory_control",
        )
        if self.archive.claim_candidate_job(exchange_id) is None:
            return None
        candidate = self.archive.complete_candidate_job(
            exchange_id,
            [
                CandidateDraft(
                    CandidateKind.CORRECTION,
                    replacement,
                    1.0,
                    1.0,
                    source=EvidenceSource.USER_CONFIRMED,
                    temporal_scope=original.temporal_scope,
                    subject=original.subject_scope,
                    target_claim_id=original.id,
                    evidence_excerpt=user_text,
                )
            ],
            engine_id="deterministic-user-control",
            extractor_version="inspector-v1",
        )[0]
        result = MemoryEvaluator(self.archive).evaluate(candidate.id)
        if not result.applied:
            return None
        return next(
            (
                claim
                for claim in self.archive.get_active_claims()
                if claim.content == replacement and claim.supersedes_id == original.id
            ),
            None,
        )

    def suppress(self, subject_id: str, reason: str) -> bool:
        """Remove one claim from the next composed context without deletion."""
        if self.archive.get_claim(subject_id) is not None:
            return self.archive.set_claim_state(
                subject_id,
                "suppressed",
                reason=reason,
            )
        return self.archive.set_schema_state(
            subject_id,
            "suppressed",
            reason=reason,
        )

    def restore(self, subject_id: str) -> bool:
        """Restore a previously suppressed claim to response context."""
        if self.archive.get_claim(subject_id) is not None:
            return self.archive.set_claim_state(
                subject_id,
                "active",
                reason="user restored",
            )
        return self.archive.set_schema_state(
            subject_id,
            "active",
            reason="user restored",
        )

    def delete_subject(
        self,
        subject_id: str,
        *,
        include_raw_evidence: bool = False,
    ) -> dict[str, int]:
        """Delete one subject, retaining immutable raw evidence by default."""
        if self.archive.get_claim(subject_id) is not None:
            return self.archive.delete_claim_subject(
                subject_id,
                include_raw_evidence=include_raw_evidence,
            )
        return self.archive.delete_schema_subject(subject_id)

    def deletion_preview(self) -> dict[str, int]:
        """Return affected row counts before a bulk delete is authorized."""
        return self.archive.personal_table_counts()

    def delete_all_personal_memory(self, confirm_token: str) -> dict[str, int]:
        """Delete all personal memory only with the exact confirmation token."""
        if confirm_token != DELETE_ALL_CONFIRMATION_TOKEN:
            raise ValueError("exact confirmation token required")
        return self.archive.delete_all_personal_data()

    def _claim_view(
        self,
        claim: PersonalClaim,
        *,
        include_evidence: bool,
    ) -> MemorySubjectView:
        evidence = (
            self.archive.evidence_texts(claim.evidence_ids)
            if include_evidence
            else ()
        )
        return MemorySubjectView(
            id=claim.id,
            subject_type="claim",
            content=claim.content,
            state=claim.state.value,
            maturity="atomic",
            conditions=(),
            created_at=claim.created_at,
            updated_at=claim.updated_at,
            supporting_evidence=evidence,
        )

    def _schema_view(
        self,
        schema: PersonalSchema,
        *,
        include_evidence: bool,
    ) -> MemorySubjectView:
        support = (
            self.archive.schema_evidence_texts(schema.id, relation="support")
            if include_evidence
            else ()
        )
        counter = (
            self.archive.schema_evidence_texts(schema.id, relation="counter")
            if include_evidence
            else ()
        )
        return MemorySubjectView(
            id=schema.id,
            subject_type="schema",
            content=schema.content,
            state=schema.state.value,
            maturity=schema.maturity.value,
            conditions=schema.conditions,
            created_at=schema.created_at,
            updated_at=schema.updated_at,
            supporting_evidence=support,
            opposing_evidence=counter,
            previous_versions=self.archive.schema_previous_versions(schema.id),
        )


__all__ = [
    "DELETE_ALL_CONFIRMATION_TOKEN",
    "MemorySubjectView",
    "PersonalMemoryInspector",
]
