"""Idempotent evaluation of provisional personal-memory candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from openjarvis.memory.adaptation import SchemaAdaptationEngine
from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.personal_models import AdaptationOperation, MemoryCandidate

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_LOW_INFORMATION_WORDS = frozenset(
    {"a", "an", "and", "i", "is", "me", "my", "the", "to", "user"}
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Outcome suitable for worker logging without conversation text."""

    candidate_id: str
    applied: bool
    reason_code: str
    decision_id: str = ""
    superseded_claim_ids: tuple[str, ...] = ()


class MemoryEvaluator:
    """Ground one candidate in user text, then apply deterministic state changes."""

    def __init__(
        self,
        archive: PersonalMemoryArchive,
        *,
        relation_classifier: Any = None,
        adaptation_engine: SchemaAdaptationEngine | None = None,
    ) -> None:
        self._archive = archive
        self._relation_classifier = relation_classifier
        self._adaptation_engine = adaptation_engine or SchemaAdaptationEngine()

    def evaluate(self, candidate_id: str) -> EvaluationResult:
        """Evaluate once; a replay returns without duplicating any state."""
        candidate = self._archive.get_candidate(candidate_id)
        if candidate is None:
            return EvaluationResult(candidate_id, False, "candidate_not_found")
        if candidate.status != "pending":
            return EvaluationResult(
                candidate_id,
                False,
                "candidate_already_evaluated",
            )
        exchange = self._archive.get_exchange(candidate.exchange_id)
        if exchange is None:
            applied = self._archive.reject_candidate(
                candidate_id,
                reason_code="supporting_exchange_missing",
            )
            return EvaluationResult(
                candidate_id,
                False,
                "supporting_exchange_missing"
                if applied
                else "candidate_already_evaluated",
            )
        if not self._supported_by_user_text(candidate, exchange.user_text):
            applied = self._archive.reject_candidate(
                candidate_id,
                reason_code="unsupported_by_user_evidence",
            )
            return EvaluationResult(
                candidate_id,
                False,
                "unsupported_by_user_evidence"
                if applied
                else "candidate_already_evaluated",
            )

        active_claims = self._archive.get_active_claims()
        related_schemas = [
            schema
            for schema in self._archive.get_active_schemas()
            if schema.subject_scope in {"", "user", candidate.subject}
        ][:8]
        relation = None
        if related_schemas and self._relation_classifier is not None:
            relation = self._relation_classifier.classify(
                candidate,
                related_schemas,
            ).relation
        decision = self._adaptation_engine.decide(
            candidate=self._draft_from_candidate(candidate),
            active_claims=active_claims,
            related_schemas=related_schemas,
            relation=relation,
        )
        if decision.operation is AdaptationOperation.NO_OP:
            applied = self._archive.reject_candidate(
                candidate_id,
                reason_code=decision.reason_code,
            )
            return EvaluationResult(
                candidate_id,
                False,
                decision.reason_code if applied else "candidate_already_evaluated",
            )
        persisted = self._archive.apply_candidate_decision(
            candidate_id,
            operation=decision.operation,
            reason_code=decision.reason_code,
            superseded_claim_ids=decision.superseded_claim_ids,
            target_schema_ids=decision.target_schema_ids,
        )
        if persisted is None:
            return EvaluationResult(
                candidate_id,
                False,
                "candidate_already_evaluated",
            )
        return EvaluationResult(
            candidate_id,
            True,
            decision.reason_code,
            decision_id=str(persisted["decision_id"]),
            superseded_claim_ids=tuple(persisted["superseded_claim_ids"]),
        )

    @staticmethod
    def _draft_from_candidate(candidate: MemoryCandidate):
        from openjarvis.memory.personal_models import CandidateDraft

        return CandidateDraft(
            candidate.kind,
            candidate.content,
            candidate.importance,
            candidate.confidence,
            source=candidate.source,
            temporal_scope=candidate.temporal_scope,
            subject=candidate.subject,
            target_claim_id=candidate.target_claim_id,
        )

    @classmethod
    def _supported_by_user_text(
        cls,
        candidate: MemoryCandidate,
        user_text: str,
    ) -> bool:
        candidate_normalized = " ".join(cls._tokens(candidate.content))
        user_normalized = " ".join(cls._tokens(user_text))
        if not candidate_normalized or not user_normalized:
            return False
        if candidate_normalized in user_normalized:
            return True
        candidate_tokens = set(candidate_normalized.split())
        user_tokens = set(user_normalized.split())
        meaningful_candidate = candidate_tokens - _LOW_INFORMATION_WORDS
        meaningful_user = user_tokens - _LOW_INFORMATION_WORDS
        if not meaningful_candidate:
            return False
        overlap = meaningful_candidate & meaningful_user
        return len(overlap) / len(meaningful_candidate) >= 0.5

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [match.group(0).casefold() for match in _WORD_RE.finditer(text)]


__all__ = ["EvaluationResult", "MemoryEvaluator"]
