"""Deterministic triggers for reviewing unresolved schema conflict."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ConflictObservation:
    """Minimal conflict view needed for reflection scheduling."""

    schema_id: str
    session_id: str
    prediction_error: float
    created_at: float


@dataclass(frozen=True, slots=True)
class ReflectionTrigger:
    """Why reflection should run, without asserting that insight exists."""

    triggered: bool
    reason_codes: tuple[str, ...]


class ConflictMonitor:
    """Start reflection only for explicit, bounded trigger conditions."""

    def __init__(self, *, balance_band: float = 0.15) -> None:
        self._balance_band = max(0.0, float(balance_band))

    def should_reflect(
        self,
        conflicts: Sequence[ConflictObservation],
        *,
        support_mass: float = 0.0,
        counter_mass: float = 0.0,
        user_rejected: bool = False,
        direct_correction: bool = False,
        explicit_analysis_request: bool = False,
    ) -> ReflectionTrigger:
        reasons: list[str] = []
        sessions_by_schema: dict[str, set[str]] = defaultdict(set)
        for conflict in conflicts:
            if conflict.session_id:
                sessions_by_schema[conflict.schema_id].add(conflict.session_id)
        if any(len(session_ids) >= 2 for session_ids in sessions_by_schema.values()):
            reasons.append("repeated_cross_session_conflict")
        if (
            min(float(support_mass), float(counter_mass)) >= 0.3
            and abs(float(support_mass) - float(counter_mass)) <= self._balance_band
        ):
            reasons.append("support_counter_balance")
        if user_rejected:
            reasons.append("user_rejected_used_schema")
        if direct_correction:
            reasons.append("direct_correction_needs_neighbor_review")
        if explicit_analysis_request:
            reasons.append("user_requested_analysis")
        return ReflectionTrigger(bool(reasons), tuple(reasons))


__all__ = ["ConflictMonitor", "ConflictObservation", "ReflectionTrigger"]
