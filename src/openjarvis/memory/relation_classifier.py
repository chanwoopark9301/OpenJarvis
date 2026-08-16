"""Bounded local-model proposals for candidate-to-schema relationships."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from openjarvis.core.types import Message, Role
from openjarvis.memory.adaptation import RelationProposal
from openjarvis.memory.personal_models import CandidateDraft, PersonalSchema

_SYSTEM_PROMPT = (
    "Classify how one user-evidence candidate relates to the supplied personal "
    "schema summaries. Return exactly one JSON object with relation, scope_fit, "
    "reason, and question_would_help. relation must be supports, contradicts, "
    "conditional_exception, unrelated, or ambiguous. Do not propose actions, "
    "diagnoses, new memories, or schema changes."
)
_OUTPUT_FIELDS = frozenset(
    {"relation", "scope_fit", "reason", "question_would_help"}
)


@dataclass(frozen=True, slots=True)
class RelationAssessment:
    """Validated classifier proposal consumed by deterministic gates."""

    relation: RelationProposal
    scope_fit: float
    reason: str
    question_would_help: bool


class RelationClassifier:
    """Ask one local model for a non-authoritative, structured relation label."""

    def __init__(
        self,
        engine: Any,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self._engine = engine
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.last_error_code = ""

    def classify(
        self,
        candidate: CandidateDraft,
        schemas: Sequence[PersonalSchema],
    ) -> RelationAssessment:
        """Return a proposal or a safe ambiguous fallback; never raise."""
        self.last_error_code = ""
        relevant = tuple(schemas[:8])
        if relevant and all(
            schema.subject_scope not in {"", "user", candidate.subject}
            for schema in relevant
        ):
            return RelationAssessment(
                RelationProposal.UNRELATED,
                0.0,
                "subject_scope_mismatch",
                False,
            )

        payload = {
            "candidate": {
                "kind": candidate.kind.value,
                "content": candidate.content,
                "temporal_scope": candidate.temporal_scope,
                "subject": candidate.subject,
            },
            "schemas": [
                {
                    "schema_id": schema.id,
                    "content": schema.content,
                    "conditions": list(schema.conditions),
                    "subject": schema.subject_scope,
                    "maturity": schema.maturity.value,
                }
                for schema in relevant
            ],
        }
        messages = [
            Message(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ]
        try:
            result = self._engine.generate(
                messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception:  # noqa: BLE001 - relation proposals are best effort
            self.last_error_code = "engine_error"
            return self._ambiguous("relation_engine_unavailable")
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        assessment = self._parse(content)
        if assessment is None:
            self.last_error_code = "invalid_output"
            return self._ambiguous("invalid_relation_output")
        return assessment

    @staticmethod
    def _parse(content: str) -> RelationAssessment | None:
        try:
            raw = json.loads(content)
        except (TypeError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(raw, dict) or set(raw) != _OUTPUT_FIELDS:
            return None
        try:
            relation = RelationProposal(raw["relation"])
        except (TypeError, ValueError):
            return None
        scope_fit = raw["scope_fit"]
        reason = raw["reason"]
        question_would_help = raw["question_would_help"]
        if (
            isinstance(scope_fit, bool)
            or not isinstance(scope_fit, (int, float))
            or not 0.0 <= float(scope_fit) <= 1.0
            or not isinstance(reason, str)
            or not reason.strip()
            or len(reason.strip()) > 500
            or not isinstance(question_would_help, bool)
        ):
            return None
        return RelationAssessment(
            relation,
            float(scope_fit),
            reason.strip(),
            question_would_help,
        )

    @staticmethod
    def _ambiguous(reason: str) -> RelationAssessment:
        return RelationAssessment(RelationProposal.AMBIGUOUS, 0.0, reason, True)


__all__ = ["RelationAssessment", "RelationClassifier"]
