"""Evidence-bounded local reflection that only proposes schema adaptation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Set

from openjarvis.core.types import Message, Role
from openjarvis.memory.personal_models import AdaptationOperation

_SYSTEM_PROMPT = (
    "Recombine the supplied raw user evidence and conflicts into at most one "
    "conditional personal-schema proposal. Return exactly one JSON object with "
    "proposal, operation, scope, support_evidence_ids, counter_evidence_ids, "
    "uncertainties, and requires_user_confirmation. Cite only supplied evidence "
    "IDs, include counter evidence when conflict exists, avoid diagnosis, and do "
    "not force an insight when evidence is insufficient."
)
_FIELDS = frozenset(
    {
        "proposal",
        "operation",
        "scope",
        "support_evidence_ids",
        "counter_evidence_ids",
        "uncertainties",
        "requires_user_confirmation",
    }
)
_ACCOMMODATION_OPERATIONS = frozenset(
    {
        AdaptationOperation.ACCOMMODATE_REFINE,
        AdaptationOperation.ACCOMMODATE_SPLIT,
        AdaptationOperation.ACCOMMODATE_SUPERSEDE,
        AdaptationOperation.ACCOMMODATE_CREATE,
        AdaptationOperation.NO_OP,
    }
)


@dataclass(frozen=True, slots=True)
class ReflectionProposal:
    """A non-authoritative insight proposal with explicit provenance."""

    content: str
    operation: AdaptationOperation
    scope: str
    support_evidence_ids: tuple[str, ...]
    counter_evidence_ids: tuple[str, ...]
    uncertainties: tuple[str, ...]
    requires_user_confirmation: bool


class ReflectionEngine:
    """Ask the local model for a proposal while enforcing evidence boundaries."""

    def __init__(
        self,
        engine: Any,
        model: str,
        *,
        max_evidence: int = 12,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._engine = engine
        self._model = model
        self._max_evidence = max(1, int(max_evidence))
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.last_error_code = ""

    def reflect(
        self,
        *,
        evidence: Mapping[str, str],
        conflict_evidence_ids: Set[str] | set[str],
    ) -> ReflectionProposal | None:
        """Return one validated proposal or no insight; never mutate memory."""
        self.last_error_code = ""
        selected = list(evidence.items())[: self._max_evidence]
        allowed_ids = {evidence_id for evidence_id, _ in selected}
        payload = {
            "evidence": [
                {"evidence_id": evidence_id, "content": content}
                for evidence_id, content in selected
            ],
            "conflict_evidence_ids": sorted(
                set(conflict_evidence_ids) & allowed_ids
            ),
        }
        try:
            result = self._engine.generate(
                [
                    Message(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
                    Message(
                        role=Role.USER,
                        content=json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    ),
                ],
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                _openjarvis_background=True,
            )
        except Exception:  # noqa: BLE001 - reflection is optional background work
            self.last_error_code = "engine_error"
            return None
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        proposal = self._parse(content)
        if proposal is None:
            self.last_error_code = "invalid_output"
            return None
        cited = set(proposal.support_evidence_ids) | set(
            proposal.counter_evidence_ids
        )
        if not cited <= allowed_ids:
            self.last_error_code = "unknown_evidence_id"
            return None
        relevant_conflicts = set(conflict_evidence_ids) & allowed_ids
        if relevant_conflicts and not (
            set(proposal.counter_evidence_ids) & relevant_conflicts
        ):
            self.last_error_code = "missing_counter_evidence"
            return None
        return proposal

    @staticmethod
    def _parse(content: str) -> ReflectionProposal | None:
        try:
            raw = json.loads(content)
        except (TypeError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(raw, dict) or set(raw) != _FIELDS:
            return None
        try:
            operation = AdaptationOperation(raw["operation"])
        except (TypeError, ValueError):
            return None
        if operation not in _ACCOMMODATION_OPERATIONS:
            return None
        proposal = raw["proposal"]
        scope = raw["scope"]
        support = raw["support_evidence_ids"]
        counter = raw["counter_evidence_ids"]
        uncertainties = raw["uncertainties"]
        confirmation = raw["requires_user_confirmation"]
        if (
            not isinstance(proposal, str)
            or not proposal.strip()
            or len(proposal.strip()) > 1000
            or not isinstance(scope, str)
            or not scope.strip()
            or not isinstance(confirmation, bool)
            or not ReflectionEngine._string_list(support)
            or not ReflectionEngine._string_list(counter, allow_empty=True)
            or not ReflectionEngine._string_list(uncertainties, allow_empty=True)
        ):
            return None
        return ReflectionProposal(
            proposal.strip(),
            operation,
            scope.strip(),
            tuple(dict.fromkeys(support)),
            tuple(dict.fromkeys(counter)),
            tuple(dict.fromkeys(uncertainties)),
            confirmation,
        )

    @staticmethod
    def _string_list(value: Any, *, allow_empty: bool = False) -> bool:
        return (
            isinstance(value, list)
            and (allow_empty or bool(value))
            and all(isinstance(item, str) and item.strip() for item in value)
        )


__all__ = ["ReflectionEngine", "ReflectionProposal"]
