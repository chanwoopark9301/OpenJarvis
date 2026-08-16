"""User-first routing and durable clarification for understanding gaps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from openjarvis.memory.archive import PersonalMemoryArchive


class GapKind(str, Enum):
    """Why local personal understanding is insufficient."""

    INTENT_AMBIGUITY = "intent_ambiguity"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GENERAL_KNOWLEDGE = "general_knowledge"


@dataclass(frozen=True, slots=True)
class UnderstandingGap:
    """A missing piece and the authority needed to resolve it."""

    kind: GapKind
    user_requested_analysis: bool = False
    sensitive: bool = False


@dataclass(frozen=True, slots=True)
class GapRoute:
    """Next safe action for an unresolved gap."""

    action: str
    reason_code: str


class UnderstandingGapGate:
    """Ask the user before considering deidentified outside knowledge."""

    def __init__(self, *, external_mode: str = "ask") -> None:
        if external_mode not in {"off", "ask", "allowed"}:
            raise ValueError("external mode must be off, ask, or allowed")
        self._external_mode = external_mode

    def route(self, gap: UnderstandingGap) -> GapRoute:
        if gap.kind is GapKind.INTENT_AMBIGUITY:
            return GapRoute("ask_user", "user_is_authority_on_intent")
        if gap.kind is GapKind.INSUFFICIENT_EVIDENCE:
            return GapRoute("hold_local", "more_personal_evidence_needed")
        if not gap.user_requested_analysis or self._external_mode == "off":
            return GapRoute("hold_local", "external_analysis_not_requested")
        if gap.sensitive:
            return GapRoute("ask_external_consent", "sensitive_consent_required")
        return GapRoute("offer_external", "general_knowledge_may_help")


@dataclass(frozen=True, slots=True)
class PendingUserQuestion:
    """One clarification whose answer can change a memory decision."""

    id: str
    subject_id: str
    question: str
    decision_effect: str
    state: str
    asked_at: float
    answer_evidence_id: str
    cooldown_until: float
    created_at: float
    updated_at: float


class QuestionCoordinator:
    """Persist one useful clarification and consume a later user answer."""

    def __init__(
        self,
        archive: PersonalMemoryArchive,
        *,
        cooldown_seconds: float = 86400.0,
    ) -> None:
        self._archive = archive
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))

    def create(
        self,
        *,
        subject_id: str,
        question: str,
        decision_effect: str,
    ) -> PendingUserQuestion:
        if not subject_id.strip() or not question.strip():
            raise ValueError("subject and question are required")
        if not decision_effect.strip():
            raise ValueError("a concrete decision effect is required")
        row = self._archive.create_pending_question(
            subject_id=subject_id.strip(),
            question=question.strip(),
            decision_effect=decision_effect.strip(),
        )
        return self._from_row(row)

    def answer(
        self,
        question_id: str,
        answer_text: str,
        *,
        answer_evidence_id: str = "",
    ) -> PendingUserQuestion:
        normalized = " ".join(answer_text.casefold().split())
        unresolved = any(
            marker in normalized
            for marker in ("i do not know", "i don't know", "모르겠", "잘 모르")
        )
        if not unresolved and not answer_evidence_id:
            raise ValueError("a confirming answer requires user evidence")
        row = self._archive.answer_pending_question(
            question_id,
            answer_text=answer_text,
            answer_evidence_id=answer_evidence_id,
            unresolved=unresolved,
            cooldown_seconds=self._cooldown_seconds,
        )
        if row is None:
            raise KeyError(f"unknown or completed question: {question_id}")
        return self._from_row(row)

    def next_pending(self) -> PendingUserQuestion | None:
        row = self._archive.next_pending_question()
        return self._from_row(row) if row is not None else None

    @staticmethod
    def _from_row(row: dict) -> PendingUserQuestion:
        return PendingUserQuestion(
            id=str(row["id"]),
            subject_id=str(row["subject_id"]),
            question=str(row["question"]),
            decision_effect=str(row["decision_effect"]),
            state=str(row["state"]),
            asked_at=float(row["asked_at"]),
            answer_evidence_id=str(row["answer_evidence_id"] or ""),
            cooldown_until=float(row["cooldown_until"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )


__all__ = [
    "GapKind",
    "GapRoute",
    "PendingUserQuestion",
    "QuestionCoordinator",
    "UnderstandingGap",
    "UnderstandingGapGate",
]
