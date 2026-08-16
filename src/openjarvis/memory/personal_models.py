"""Value objects used by the local developmental personal-memory model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CandidateKind(str, Enum):
    """Closed set of provisional user-memory claims."""

    FACT = "fact"
    EPISODE = "episode"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    CORRECTION = "correction"
    ROLE_PREFERENCE = "role_preference"
    CAPABILITY_BOUNDARY = "capability_boundary"
    HYPOTHESIS = "hypothesis"


class CandidateStatus(str, Enum):
    """Review lifecycle for a provisional candidate."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class EvidenceSource(str, Enum):
    """Provenance classes kept separate throughout evaluation."""

    USER_DIRECT = "user_direct"
    USER_CONFIRMED = "user_confirmed"
    SOCIAL_TESTIMONY = "social_testimony"
    LEGACY_IMPORT = "legacy_import"
    ASSISTANT = "assistant"
    EXTERNAL = "external"


class ClaimState(str, Enum):
    """Whether a claim may participate in response context."""

    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    PENDING = "pending"
    ARCHIVED = "archived"


class SchemaMaturity(str, Enum):
    """Developmental state of an evidence-grounded personal schema."""

    TENTATIVE = "tentative"
    EMERGING = "emerging"
    STABLE_CANDIDATE = "stable_candidate"
    STABLE = "stable"
    CHALLENGED = "challenged"
    RETIRED = "retired"


class AdaptationOperation(str, Enum):
    """Only operations deterministic persistence code may apply."""

    ASSIMILATE_REINFORCE = "assimilate_reinforce"
    ASSIMILATE_LINK_ONLY = "assimilate_link_only"
    ASSIMILATE_AS_EXCEPTION = "assimilate_as_exception"
    HOLD_DISEQUILIBRIUM = "hold_disequilibrium"
    ACCOMMODATE_REFINE = "accommodate_refine"
    ACCOMMODATE_SPLIT = "accommodate_split"
    ACCOMMODATE_SUPERSEDE = "accommodate_supersede"
    ACCOMMODATE_CREATE = "accommodate_create"
    NO_OP = "no_op"


class ConflictResolutionState(str, Enum):
    """Resolution lifecycle for schema conflicts."""

    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class InsightState(str, Enum):
    """Validation lifecycle for reflection proposals."""

    DISCOVERED = "discovered"
    PENDING = "pending"
    PROBATION = "probation"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class MemoryJobState(str, Enum):
    """Durable one-shot worker lifecycle."""

    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"
    DEFERRED = "deferred"
    COMPLETE = "complete"


DIRECT_RULE_KINDS = frozenset(
    {
        CandidateKind.CONSTRAINT,
        CandidateKind.CORRECTION,
        CandidateKind.ROLE_PREFERENCE,
        CandidateKind.CAPABILITY_BOUNDARY,
    }
)
DIRECT_USER_SOURCES = frozenset(
    {EvidenceSource.USER_DIRECT, EvidenceSource.USER_CONFIRMED}
)


def _score(value: float) -> float:
    """Convert extractor estimates to the inclusive archive range."""
    return max(0.0, min(1.0, float(value)))


def _unit_score(value: float, *, name: str) -> float:
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return score


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
    """A provisional atomic claim before it is persisted for evaluation."""

    kind: CandidateKind | str
    content: str
    importance: float
    confidence: float
    source: EvidenceSource | str = EvidenceSource.USER_DIRECT
    temporal_scope: str = "unspecified"
    subject: str = "user"
    target_claim_id: str = ""

    def __post_init__(self) -> None:
        try:
            kind = CandidateKind(self.kind)
        except ValueError as exc:
            raise ValueError("unsupported candidate kind") from exc
        try:
            source = EvidenceSource(self.source)
        except ValueError as exc:
            raise ValueError("unsupported evidence source") from exc
        if kind in DIRECT_RULE_KINDS and source not in DIRECT_USER_SOURCES:
            raise ValueError("direct rules require direct user evidence")
        content = self.content.strip()
        if not content:
            raise ValueError("candidate content must not be empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "importance", _score(self.importance))
        object.__setattr__(self, "confidence", _score(self.confidence))
        object.__setattr__(self, "temporal_scope", self.temporal_scope.strip())
        object.__setattr__(self, "subject", self.subject.strip() or "user")
        object.__setattr__(self, "target_claim_id", self.target_claim_id.strip())


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """A persisted provisional memory with its exact supporting exchange."""

    id: str
    exchange_id: str
    kind: CandidateKind | str
    content: str
    importance: float
    confidence: float
    status: CandidateStatus | str
    engine_id: str
    extractor_version: str
    created_at: float
    updated_at: float
    source: EvidenceSource | str = EvidenceSource.USER_DIRECT
    temporal_scope: str = "unspecified"
    subject: str = "user"
    target_claim_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CandidateKind(self.kind))
        object.__setattr__(self, "status", CandidateStatus(self.status))
        object.__setattr__(self, "source", EvidenceSource(self.source))


@dataclass(frozen=True, slots=True)
class EvidenceMass:
    """Independent evidence axes; intentionally has no aggregate score."""

    support_mass: float = 0.0
    counter_mass: float = 0.0
    stability: float = 0.0
    plasticity: float = 1.0
    scope_fit: float = 0.0
    source_quality: float = 0.0
    temporal_validity: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self,
                name,
                _unit_score(getattr(self, name), name=name),
            )


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One immutable user-grounded item linked to an archived exchange."""

    id: str
    exchange_id: str
    content: str
    source: EvidenceSource
    created_at: float
    session_id: str = ""
    temporal_scope: str = "unspecified"
    subject: str = "user"


@dataclass(frozen=True, slots=True)
class PersonalClaim:
    """An atomic fact, preference, or direct rule and its provenance."""

    id: str
    kind: CandidateKind
    content: str
    state: ClaimState
    source: EvidenceSource
    evidence_ids: tuple[str, ...]
    created_at: float
    updated_at: float
    temporal_scope: str = "unspecified"
    subject_scope: str = "user"
    supersedes_id: str = ""
    expires_at: float = 0.0


@dataclass(frozen=True, slots=True)
class PersonalSchema:
    """A versioned, conditional pattern grounded in raw user evidence."""

    id: str
    content: str
    maturity: SchemaMaturity
    state: ClaimState
    current_version_id: str
    evidence_mass: EvidenceMass
    created_at: float
    updated_at: float
    conditions: tuple[str, ...] = field(default_factory=tuple)
    subject_scope: str = "user"
    broad_interpretation: bool = False
    user_confirmed: bool = False


@dataclass(frozen=True, slots=True)
class SchemaConflict:
    """An unresolved mismatch between new evidence and a schema version."""

    id: str
    schema_id: str
    schema_version_id: str
    evidence_id: str
    conflict_type: str
    prediction_error: float
    source_quality: float
    created_at: float
    resolution_state: ConflictResolutionState = ConflictResolutionState.UNRESOLVED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prediction_error",
            _unit_score(self.prediction_error, name="prediction_error"),
        )
        object.__setattr__(
            self,
            "source_quality",
            _unit_score(self.source_quality, name="source_quality"),
        )


@dataclass(frozen=True, slots=True)
class InsightCandidate:
    """A reflection proposal that cannot alter schemas until validated."""

    id: str
    content: str
    operation: AdaptationOperation
    state: InsightState
    support_evidence_ids: tuple[str, ...]
    counter_evidence_ids: tuple[str, ...]
    created_at: float
    requires_user_confirmation: bool = True
    uncertainties: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """Append-only audit record for a personal-memory state transition."""

    id: str
    subject_id: str
    subject_type: str
    operation: AdaptationOperation
    reason_code: str
    affected_ids: tuple[str, ...]
    created_at: float


@dataclass(frozen=True, slots=True)
class MemoryJob:
    """One restart-safe background operation carrying identifiers, not raw text."""

    id: str
    job_type: str
    subject_id: str
    idempotency_key: str
    payload_json: str
    state: MemoryJobState | str
    priority: int
    attempts: int
    error_code: str
    claimed_at: float
    next_attempt_at: float
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", MemoryJobState(self.state))


__all__ = [
    "AdaptationOperation",
    "CandidateDraft",
    "CandidateKind",
    "CandidateStatus",
    "ClaimState",
    "ConflictResolutionState",
    "ConversationExchange",
    "DIRECT_RULE_KINDS",
    "DIRECT_USER_SOURCES",
    "EvidenceItem",
    "EvidenceMass",
    "EvidenceSource",
    "InsightCandidate",
    "InsightState",
    "MemoryCandidate",
    "MemoryDecision",
    "MemoryJob",
    "MemoryJobState",
    "PersonalClaim",
    "PersonalSchema",
    "SchemaConflict",
    "SchemaMaturity",
]
