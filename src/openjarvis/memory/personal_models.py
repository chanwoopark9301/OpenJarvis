"""Value objects used by the local-only personal memory archive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CandidateKind = Literal["fact", "episode"]
CandidateStatus = Literal["pending", "accepted", "rejected", "superseded"]


def _score(value: float) -> float:
    """Convert a score to the inclusive range used by the archive."""
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class ConversationExchange:
    """One immutable, locally archived user/assistant conversation pair."""

    id: str
    created_at: float
    archived_at: float
    source: str
    user_text: str
    assistant_text: str
    content_hash: str
    agent_id: str = ""
    session_id: str = ""


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    """A provisional fact or episode before it is persisted for review."""

    kind: CandidateKind
    content: str
    importance: float
    confidence: float

    def __post_init__(self) -> None:
        if self.kind not in ("fact", "episode"):
            raise ValueError("candidate kind must be 'fact' or 'episode'")
        content = self.content.strip()
        if not content:
            raise ValueError("candidate content must not be empty")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "importance", _score(self.importance))
        object.__setattr__(self, "confidence", _score(self.confidence))


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """A persisted provisional memory with its exact supporting exchange."""

    id: str
    exchange_id: str
    kind: CandidateKind
    content: str
    importance: float
    confidence: float
    status: CandidateStatus
    engine_id: str
    extractor_version: str
    created_at: float
    updated_at: float


__all__ = [
    "CandidateDraft",
    "CandidateKind",
    "CandidateStatus",
    "ConversationExchange",
    "MemoryCandidate",
]
