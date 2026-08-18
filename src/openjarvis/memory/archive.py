"""SQLite persistence for local personal conversations and candidate work."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from openjarvis.memory.personal_models import (
    AdaptationOperation,
    CandidateDraft,
    CandidateKind,
    ClaimState,
    ConversationExchange,
    EvidenceItem,
    EvidenceMass,
    EvidenceSource,
    InsightCandidate,
    InsightState,
    MemoryCandidate,
    MemoryJob,
    PersonalClaim,
    PersonalSchema,
    SchemaMaturity,
)

_SCHEMA_VERSION = 5
_RELEASE_ATTESTATION_GATES = frozenset(
    {
        "assistant_only_rejection",
        "benchmark_10000",
        "chat_priority_and_concurrency",
        "external_isolation",
        "provider_privacy_interception",
        "replay_idempotency",
        "restart_and_direct_rule",
        "user_controls",
    }
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS archive_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_exchanges (
  id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  archived_at REAL NOT NULL,
  source TEXT NOT NULL,
  agent_id TEXT NOT NULL DEFAULT '',
  session_id TEXT NOT NULL DEFAULT '',
  user_text TEXT NOT NULL,
  assistant_text TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  candidate_state TEXT NOT NULL DEFAULT 'pending',
  candidate_attempts INTEGER NOT NULL DEFAULT 0,
  candidate_error TEXT NOT NULL DEFAULT '',
  candidate_claimed_at REAL NOT NULL DEFAULT 0,
  candidate_next_attempt_at REAL NOT NULL DEFAULT 0,
  candidate_extractor_version TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS memory_candidates (
  id TEXT PRIMARY KEY,
  exchange_id TEXT NOT NULL REFERENCES conversation_exchanges(id),
  kind TEXT NOT NULL,
  content TEXT NOT NULL,
  evidence_excerpt TEXT NOT NULL DEFAULT '',
  importance REAL NOT NULL,
  confidence REAL NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  engine_id TEXT NOT NULL,
  extractor_version TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'user_direct',
  temporal_scope TEXT NOT NULL DEFAULT 'unspecified',
  subject TEXT NOT NULL DEFAULT 'user',
  target_claim_id TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE(exchange_id, kind, content)
);

CREATE TABLE IF NOT EXISTS candidate_extraction_runs (
  exchange_id TEXT NOT NULL REFERENCES conversation_exchanges(id),
  extractor_version TEXT NOT NULL,
  candidate_count INTEGER NOT NULL DEFAULT 0,
  completed_at REAL NOT NULL,
  PRIMARY KEY(exchange_id, extractor_version)
);

CREATE INDEX IF NOT EXISTS idx_exchanges_candidate_created
ON conversation_exchanges(candidate_state, created_at DESC);

CREATE TABLE IF NOT EXISTS evidence_items (
  id TEXT PRIMARY KEY,
  exchange_id TEXT NOT NULL REFERENCES conversation_exchanges(id),
  candidate_id TEXT REFERENCES memory_candidates(id),
  source TEXT NOT NULL,
  content TEXT NOT NULL,
  temporal_scope TEXT NOT NULL DEFAULT 'unspecified',
  subject TEXT NOT NULL DEFAULT 'user',
  session_id TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  UNIQUE(candidate_id)
);

CREATE TABLE IF NOT EXISTS personal_claims (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  content TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending',
  source TEXT NOT NULL,
  temporal_scope TEXT NOT NULL DEFAULT 'unspecified',
  subject_scope TEXT NOT NULL DEFAULT 'user',
  supersedes_id TEXT REFERENCES personal_claims(id),
  expires_at REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_state_updated
ON personal_claims(state, updated_at DESC);

CREATE TABLE IF NOT EXISTS claim_evidence_links (
  claim_id TEXT NOT NULL REFERENCES personal_claims(id),
  evidence_id TEXT NOT NULL REFERENCES evidence_items(id),
  created_at REAL NOT NULL,
  PRIMARY KEY(claim_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS personal_schemas (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending',
  maturity TEXT NOT NULL DEFAULT 'tentative',
  current_version_id TEXT,
  subject_scope TEXT NOT NULL DEFAULT 'user',
  broad_interpretation INTEGER NOT NULL DEFAULT 0,
  user_confirmed INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_versions (
  id TEXT PRIMARY KEY,
  schema_id TEXT NOT NULL REFERENCES personal_schemas(id),
  version_number INTEGER NOT NULL,
  content TEXT NOT NULL,
  conditions_json TEXT NOT NULL DEFAULT '[]',
  support_mass REAL NOT NULL DEFAULT 0,
  counter_mass REAL NOT NULL DEFAULT 0,
  stability REAL NOT NULL DEFAULT 0,
  plasticity REAL NOT NULL DEFAULT 1,
  scope_fit REAL NOT NULL DEFAULT 0,
  source_quality REAL NOT NULL DEFAULT 0,
  temporal_validity REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  UNIQUE(schema_id, version_number)
);

CREATE TABLE IF NOT EXISTS schema_evidence_links (
  schema_version_id TEXT NOT NULL REFERENCES schema_versions(id),
  evidence_id TEXT NOT NULL REFERENCES evidence_items(id),
  relation TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY(schema_version_id, evidence_id, relation)
);

CREATE TABLE IF NOT EXISTS schema_conflicts (
  id TEXT PRIMARY KEY,
  schema_id TEXT NOT NULL REFERENCES personal_schemas(id),
  schema_version_id TEXT NOT NULL REFERENCES schema_versions(id),
  evidence_id TEXT NOT NULL REFERENCES evidence_items(id),
  conflict_type TEXT NOT NULL,
  prediction_error REAL NOT NULL,
  source_quality REAL NOT NULL,
  resolution_state TEXT NOT NULL DEFAULT 'unresolved',
  resolution_decision_id TEXT,
  created_at REAL NOT NULL,
  resolved_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS insight_candidates (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT 'user',
  operation TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'discovered',
  support_evidence_json TEXT NOT NULL DEFAULT '[]',
  counter_evidence_json TEXT NOT NULL DEFAULT '[]',
  uncertainties_json TEXT NOT NULL DEFAULT '[]',
  requires_user_confirmation INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS external_knowledge_items (
  id TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  provider_id TEXT NOT NULL,
  source_url TEXT NOT NULL,
  query_text TEXT NOT NULL,
  summary TEXT NOT NULL,
  consent_id TEXT NOT NULL,
  deidentified INTEGER NOT NULL DEFAULT 0,
  retrieved_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_jobs (
  id TEXT PRIMARY KEY,
  job_type TEXT NOT NULL,
  subject_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  payload_json TEXT NOT NULL DEFAULT '{}',
  state TEXT NOT NULL DEFAULT 'pending',
  priority INTEGER NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  error_code TEXT NOT NULL DEFAULT '',
  claimed_at REAL NOT NULL DEFAULT 0,
  next_attempt_at REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_decisions (
  id TEXT PRIMARY KEY,
  subject_id TEXT NOT NULL,
  subject_type TEXT NOT NULL,
  operation TEXT NOT NULL,
  reason_code TEXT NOT NULL,
  affected_ids_json TEXT NOT NULL DEFAULT '[]',
  details_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_feedback (
  id TEXT PRIMARY KEY,
  subject_id TEXT NOT NULL,
  action TEXT NOT NULL,
  user_text TEXT NOT NULL DEFAULT '',
  evidence_id TEXT REFERENCES evidence_items(id),
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_user_questions (
  id TEXT PRIMARY KEY,
  subject_id TEXT NOT NULL,
  question TEXT NOT NULL,
  decision_effect TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending',
  asked_at REAL NOT NULL DEFAULT 0,
  answer_evidence_id TEXT REFERENCES evidence_items(id),
  cooldown_until REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_active_kind
ON personal_claims(state, kind, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_schemas_active_maturity
ON personal_schemas(state, maturity, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_conflicts_open_schema
ON schema_conflicts(resolution_state, schema_id, created_at);

CREATE INDEX IF NOT EXISTS idx_jobs_claimable
ON memory_jobs(state, next_attempt_at, priority, created_at);
"""


class PersonalMemoryArchive:
    """Own the durable, local-only archive and its candidate-job lifecycle."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            version_row = connection.execute(
                "SELECT value FROM archive_metadata WHERE key = 'schema_version'"
            ).fetchone()
            try:
                previous_schema_version = (
                    int(version_row["value"]) if version_row is not None else 0
                )
            except (TypeError, ValueError):
                previous_schema_version = 0
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(conversation_exchanges)"
                ).fetchall()
            }
            if "candidate_next_attempt_at" not in columns:
                connection.execute(
                    """
                    ALTER TABLE conversation_exchanges
                    ADD COLUMN candidate_next_attempt_at REAL NOT NULL DEFAULT 0
                    """
                )
            if "candidate_extractor_version" not in columns:
                connection.execute(
                    """
                    ALTER TABLE conversation_exchanges
                    ADD COLUMN candidate_extractor_version TEXT NOT NULL DEFAULT ''
                    """
                )
            candidate_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(memory_candidates)"
                ).fetchall()
            }
            candidate_migrations = {
                "source": "TEXT NOT NULL DEFAULT 'legacy_import'",
                "temporal_scope": "TEXT NOT NULL DEFAULT 'unspecified'",
                "subject": "TEXT NOT NULL DEFAULT 'user'",
                "target_claim_id": "TEXT NOT NULL DEFAULT ''",
                "evidence_excerpt": "TEXT NOT NULL DEFAULT ''",
            }
            for name, declaration in candidate_migrations.items():
                if name not in candidate_columns:
                    connection.execute(
                        f"ALTER TABLE memory_candidates ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                """
                INSERT INTO candidate_extraction_runs (
                    exchange_id, extractor_version, candidate_count, completed_at
                )
                SELECT conversation_exchanges.id,
                       conversation_exchanges.candidate_extractor_version,
                       (
                         SELECT COUNT(*) FROM memory_candidates
                         WHERE memory_candidates.exchange_id =
                               conversation_exchanges.id
                       ),
                       conversation_exchanges.archived_at
                FROM conversation_exchanges
                WHERE conversation_exchanges.candidate_state = 'complete'
                  AND conversation_exchanges.candidate_extractor_version != ''
                ON CONFLICT(exchange_id, extractor_version) DO NOTHING
                """
            )
            if previous_schema_version == 2 and "source" in candidate_columns:
                quarantined_rows = connection.execute(
                    """
                    SELECT id FROM memory_candidates
                    WHERE status = 'pending' AND source = 'user_direct'
                    """
                ).fetchall()
                connection.execute(
                    """
                    UPDATE memory_candidates
                    SET source = 'legacy_import', status = 'quarantined'
                    WHERE status = 'pending' AND source = 'user_direct'
                    """
                )
                for row in quarantined_rows:
                    candidate_id = str(row["id"])
                    connection.execute(
                        """
                        INSERT INTO memory_decisions (
                          id, subject_id, subject_type, operation, reason_code,
                          affected_ids_json, details_json, created_at
                        ) VALUES (?, ?, 'candidate', 'no_op',
                                  'v3_provenance_quarantine', ?, '{}', ?)
                        """,
                        (
                            str(uuid.uuid4()),
                            candidate_id,
                            json.dumps([candidate_id], separators=(",", ":")),
                            time.time(),
                        ),
                    )
            table_column_migrations = {
                "personal_claims": {
                    "subject_scope": "TEXT NOT NULL DEFAULT 'user'",
                },
                "personal_schemas": {
                    "subject_scope": "TEXT NOT NULL DEFAULT 'user'",
                },
                "insight_candidates": {
                    "scope": "TEXT NOT NULL DEFAULT 'user'",
                },
            }
            for table, migrations in table_column_migrations.items():
                existing = {
                    str(row["name"])
                    for row in connection.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
                for name, declaration in migrations.items():
                    if name not in existing:
                        connection.execute(
                            f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                        )
            connection.execute(
                """
                INSERT INTO archive_metadata(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(_SCHEMA_VERSION),),
            )

    @property
    def path(self) -> Path:
        """Filesystem path of this archive database."""
        return self._path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _content_hash(
        user_text: str,
        assistant_text: str,
        source: str,
        agent_id: str,
        session_id: str,
    ) -> str:
        payload = "\x00".join((user_text, assistant_text, source, agent_id, session_id))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _exchange_from_row(row: sqlite3.Row) -> ConversationExchange:
        return ConversationExchange(
            id=str(row["id"]),
            created_at=float(row["created_at"]),
            archived_at=float(row["archived_at"]),
            source=str(row["source"]),
            user_text=str(row["user_text"]),
            assistant_text=str(row["assistant_text"]),
            content_hash=str(row["content_hash"]),
            agent_id=str(row["agent_id"]),
            session_id=str(row["session_id"]),
        )

    @staticmethod
    def _candidate_from_row(row: sqlite3.Row) -> MemoryCandidate:
        return MemoryCandidate(
            id=str(row["id"]),
            exchange_id=str(row["exchange_id"]),
            kind=str(row["kind"]),  # database values are created from CandidateDraft
            content=str(row["content"]),
            evidence_excerpt=str(row["evidence_excerpt"]),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            status=str(row["status"]),
            engine_id=str(row["engine_id"]),
            extractor_version=str(row["extractor_version"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            source=str(row["source"]),
            temporal_scope=str(row["temporal_scope"]),
            subject=str(row["subject"]),
            target_claim_id=str(row["target_claim_id"]),
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> MemoryJob:
        return MemoryJob(
            id=str(row["id"]),
            job_type=str(row["job_type"]),
            subject_id=str(row["subject_id"]),
            idempotency_key=str(row["idempotency_key"]),
            payload_json=str(row["payload_json"]),
            state=str(row["state"]),
            priority=int(row["priority"]),
            attempts=int(row["attempts"]),
            error_code=str(row["error_code"]),
            claimed_at=float(row["claimed_at"]),
            next_attempt_at=float(row["next_attempt_at"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _claim_from_row(
        row: sqlite3.Row,
        evidence_ids: Sequence[str] = (),
    ) -> PersonalClaim:
        return PersonalClaim(
            id=str(row["id"]),
            kind=CandidateKind(str(row["kind"])),
            content=str(row["content"]),
            state=ClaimState(str(row["state"])),
            source=EvidenceSource(str(row["source"])),
            evidence_ids=tuple(evidence_ids),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            temporal_scope=str(row["temporal_scope"]),
            subject_scope=str(row["subject_scope"]),
            supersedes_id=str(row["supersedes_id"] or ""),
            expires_at=float(row["expires_at"]),
        )

    def schema_version(self) -> int:
        """Return the archive schema version after in-place migration."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM archive_metadata WHERE key = 'schema_version'"
            ).fetchone()
        return int(row["value"]) if row is not None else 0

    def table_names(self) -> set[str]:
        """Expose table names for migration diagnostics and integrity tests."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        return {str(row["name"]) for row in rows}

    def integrity_check(self) -> str:
        """Return SQLite's own archive integrity result."""
        with self._lock, self._connect() as connection:
            row = connection.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row is not None else "unknown"

    def get_metadata(self, key: str, default: str = "") -> str:
        """Read one non-sensitive archive control value."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM archive_metadata WHERE key = ?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row is not None else default

    def set_metadata(self, key: str, value: str) -> None:
        """Persist one non-sensitive archive control value."""
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO archive_metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def record_shadow_composition(
        self,
        *,
        latency_ms: float,
        constraint_count: int,
        schema_count: int,
        episode_count: int,
        raw_evidence_count: int,
    ) -> None:
        """Store aggregate shadow counters without retaining prompt text."""
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM archive_metadata WHERE key = 'shadow_metrics'"
            ).fetchone()
            try:
                metrics = json.loads(str(row["value"])) if row is not None else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                metrics = {}
            metrics["compositions"] = int(metrics.get("compositions", 0)) + 1
            metrics["latency_ms_total"] = float(
                metrics.get("latency_ms_total", 0.0)
            ) + max(0.0, float(latency_ms))
            metrics["latency_ms_max"] = max(
                float(metrics.get("latency_ms_max", 0.0)),
                max(0.0, float(latency_ms)),
            )
            metrics["constraint_items"] = int(
                metrics.get("constraint_items", 0)
            ) + max(0, int(constraint_count))
            metrics["schema_items"] = int(metrics.get("schema_items", 0)) + max(
                0, int(schema_count)
            )
            metrics["episode_items"] = int(
                metrics.get("episode_items", 0)
            ) + max(0, int(episode_count))
            metrics["raw_evidence_items"] = int(
                metrics.get("raw_evidence_items", 0)
            ) + max(0, int(raw_evidence_count))
            payload = json.dumps(metrics, sort_keys=True, separators=(",", ":"))
            connection.execute(
                """
                INSERT INTO archive_metadata(key, value)
                VALUES ('shadow_metrics', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (payload,),
            )
            connection.execute(
                """
                INSERT INTO archive_metadata(key, value)
                VALUES ('shadow_started_at', ?)
                ON CONFLICT(key) DO NOTHING
                """,
                (str(now),),
            )

    def shadow_metrics(self) -> dict[str, int | float]:
        """Return aggregate comparison counters only."""
        raw = self.get_metadata("shadow_metrics", "{}")
        try:
            value = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def release_gate_failures(self) -> tuple[str, ...]:
        """Run content-free structural checks required before active injection."""
        failures: list[str] = []
        with self._lock, self._connect() as connection:
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity is None or str(integrity[0]).casefold() != "ok":
                failures.append("archive_integrity_failed")
            ungrounded = connection.execute(
                """
                SELECT COUNT(*) FROM personal_schemas AS s
                WHERE s.current_version_id IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM schema_evidence_links AS l
                    WHERE l.schema_version_id = s.current_version_id
                  )
                """
            ).fetchone()
            if ungrounded is not None and int(ungrounded[0]) > 0:
                failures.append("ungrounded_personal_schema")
            broad_unconfirmed = connection.execute(
                """
                SELECT COUNT(*) FROM personal_schemas
                WHERE state = 'active' AND broad_interpretation = 1
                  AND user_confirmed = 0
                """
            ).fetchone()
            if broad_unconfirmed is not None and int(broad_unconfirmed[0]) > 0:
                failures.append("broad_schema_unconfirmed")
            unfinished = connection.execute(
                """
                SELECT COUNT(*) FROM memory_jobs
                WHERE state IN ('pending', 'processing')
                """
            ).fetchone()
            if unfinished is not None and int(unfinished[0]) > 0:
                failures.append("background_jobs_not_drained")
        metrics = self.shadow_metrics()
        if int(metrics.get("compositions", 0)) < 1:
            failures.append("shadow_sample_missing")
        if float(metrics.get("latency_ms_max", 0.0)) > 100.0:
            failures.append("shadow_context_latency_exceeded")
        try:
            attestations = json.loads(
                self.get_metadata("release_gate_attestations", "{}")
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            attestations = {}
        for gate in sorted(_RELEASE_ATTESTATION_GATES):
            if attestations.get(gate) is not True:
                failures.append(f"release_attestation_missing:{gate}")
        return tuple(failures)

    def record_release_gate_attestation(self, gate: str, *, passed: bool) -> None:
        """Persist and audit one named release-check result without personal text."""
        if gate not in _RELEASE_ATTESTATION_GATES:
            raise ValueError("unknown release gate")
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM archive_metadata WHERE key = ?",
                ("release_gate_attestations",),
            ).fetchone()
            try:
                values = json.loads(str(row["value"])) if row is not None else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                values = {}
            values[gate] = bool(passed)
            connection.execute(
                """
                INSERT INTO archive_metadata(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (
                    "release_gate_attestations",
                    json.dumps(values, sort_keys=True, separators=(",", ":")),
                ),
            )
            connection.execute(
                """
                INSERT INTO memory_decisions (
                  id, subject_id, subject_type, operation, reason_code,
                  affected_ids_json, details_json, created_at
                ) VALUES (?, 'rollout', 'rollout', 'no_op', ?, '[]', '{}', ?)
                """,
                (str(uuid.uuid4()), f"release_gate_{gate}_{passed}", now),
            )

    def rollout_is_active(self) -> bool:
        """Return whether the explicit, audited release gate was activated."""
        return self.get_metadata("release_ready", "0") == "1"

    def mark_rollout_active(self) -> None:
        """Activate response injection and append its audit in one transaction."""
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for key, value in (
                ("release_ready", "1"),
                ("release_activated_at", str(now)),
            ):
                connection.execute(
                    """
                    INSERT INTO archive_metadata(key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, value),
                )
            connection.execute(
                """
                INSERT INTO memory_decisions (
                  id, subject_id, subject_type, operation, reason_code,
                  affected_ids_json, details_json, created_at
                ) VALUES (?, 'rollout', 'rollout', 'no_op',
                          'release_gates_passed', '[]', '{}', ?)
                """,
                (str(uuid.uuid4()), now),
            )

    def record_exchange(
        self,
        *,
        exchange_id: str,
        user_text: str,
        assistant_text: str,
        source: str,
        agent_id: str = "",
        session_id: str = "",
    ) -> ConversationExchange:
        """Persist an exchange once; replaying the ID returns the original evidence."""
        if not exchange_id:
            raise ValueError("exchange_id must not be empty")
        now = time.time()
        values = (
            exchange_id,
            now,
            now,
            source or "",
            agent_id or "",
            session_id or "",
            user_text or "",
            assistant_text or "",
            self._content_hash(
                user_text or "",
                assistant_text or "",
                source or "",
                agent_id or "",
                session_id or "",
            ),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_exchanges (
                    id, created_at, archived_at, source, agent_id, session_id,
                    user_text, assistant_text, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                values,
            )
            row = connection.execute(
                "SELECT * FROM conversation_exchanges WHERE id = ?", (exchange_id,)
            ).fetchone()
        if row is None:  # pragma: no cover - SQLite transaction guarantee
            raise RuntimeError("archived exchange could not be read")
        return self._exchange_from_row(row)

    def get_exchange(self, exchange_id: str) -> ConversationExchange | None:
        """Return an archived exchange by its stable ID, if present."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_exchanges WHERE id = ?", (exchange_id,)
            ).fetchone()
        return self._exchange_from_row(row) if row is not None else None

    def recent_exchanges(
        self,
        exchange_id: str,
        *,
        limit: int = 6,
    ) -> list[ConversationExchange]:
        """Return earlier exchanges in the current session or source scope."""
        with self._lock, self._connect() as connection:
            current = connection.execute(
                "SELECT created_at, session_id, source "
                "FROM conversation_exchanges WHERE id = ?",
                (exchange_id,),
            ).fetchone()
            if current is None:
                return []
            scope_column = "session_id" if current["session_id"] else "source"
            scope_value = current[scope_column]
            rows = connection.execute(
                f"SELECT * FROM conversation_exchanges "
                f"WHERE {scope_column} = ? AND (created_at < ? "
                f"OR (created_at = ? AND id < ?)) "
                f"ORDER BY created_at DESC, id DESC LIMIT ?",
                (
                    scope_value,
                    current["created_at"],
                    current["created_at"],
                    exchange_id,
                    max(0, int(limit)),
                ),
            ).fetchall()
        return [self._exchange_from_row(row) for row in reversed(rows)]

    def get_latest_unevaluated_user_text(self) -> str:
        """Return only the user's newest raw text awaiting candidate extraction."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_text FROM conversation_exchanges
                WHERE candidate_state != 'complete'
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        return str(row["user_text"]) if row is not None else ""

    def recent_incomplete_exchanges(
        self,
        *,
        limit: int = 6,
    ) -> list[ConversationExchange]:
        """Return newest durable incomplete user exchanges in chronological order."""
        count = max(0, int(limit))
        if count == 0:
            return []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversation_exchanges
                WHERE candidate_state != 'complete' AND TRIM(user_text) != ''
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (count,),
            ).fetchall()
        return [self._exchange_from_row(row) for row in reversed(rows)]

    def get_candidate(self, candidate_id: str) -> MemoryCandidate | None:
        """Return one provisional candidate by stable ID."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
        return self._candidate_from_row(row) if row is not None else None

    def find_candidates(
        self,
        *,
        source: EvidenceSource | str | None = None,
    ) -> list[MemoryCandidate]:
        """List provisional candidates, optionally filtered by provenance."""
        params: tuple[str, ...] = ()
        source_clause = ""
        if source is not None:
            source_clause = " WHERE source = ?"
            params = (EvidenceSource(source).value,)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM memory_candidates{source_clause}
                ORDER BY created_at ASC, id ASC
                """,
                params,
            ).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def get_active_claims(
        self,
        *,
        kind: CandidateKind | str | None = None,
    ) -> list[PersonalClaim]:
        """Return active atomic claims with their exact evidence links."""
        params: list[Any] = []
        kind_clause = ""
        if kind is not None:
            kind_clause = " AND kind = ?"
            params.append(CandidateKind(kind).value)
        params.insert(0, time.time())
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM personal_claims
                WHERE state = 'active'
                  AND (expires_at = 0 OR expires_at > ?){kind_clause}
                ORDER BY updated_at DESC, id ASC
                """,
                params,
            ).fetchall()
            claims = []
            for row in rows:
                evidence_rows = connection.execute(
                    """
                    SELECT evidence_id FROM claim_evidence_links
                    WHERE claim_id = ? ORDER BY created_at ASC, evidence_id ASC
                    """,
                    (str(row["id"]),),
                ).fetchall()
                claims.append(
                    self._claim_from_row(
                        row,
                        [str(item["evidence_id"]) for item in evidence_rows],
                    )
                )
        return claims

    def get_claim(self, claim_id: str) -> PersonalClaim | None:
        """Return an atomic claim in any lifecycle state."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM personal_claims WHERE id = ?", (claim_id,)
            ).fetchone()
            if row is None:
                return None
            evidence_rows = connection.execute(
                """
                SELECT evidence_id FROM claim_evidence_links
                WHERE claim_id = ? ORDER BY created_at ASC, evidence_id ASC
                """,
                (claim_id,),
            ).fetchall()
        return self._claim_from_row(
            row,
            [str(item["evidence_id"]) for item in evidence_rows],
        )

    def list_claims(self) -> list[PersonalClaim]:
        """Return claims in every lifecycle state for user inspection."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM personal_claims ORDER BY updated_at DESC, id ASC"
            ).fetchall()
            claims = []
            for row in rows:
                evidence_rows = connection.execute(
                    """
                    SELECT evidence_id FROM claim_evidence_links
                    WHERE claim_id = ? ORDER BY created_at ASC, evidence_id ASC
                    """,
                    (str(row["id"]),),
                ).fetchall()
                claims.append(
                    self._claim_from_row(
                        row,
                        [str(item["evidence_id"]) for item in evidence_rows],
                    )
                )
        return claims

    def evidence_texts(self, evidence_ids: Sequence[str]) -> tuple[str, ...]:
        """Return only evidence explicitly requested by stable identifier."""
        if not evidence_ids:
            return ()
        placeholders = ",".join("?" for _ in evidence_ids)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, content FROM evidence_items
                WHERE id IN ({placeholders})
                """,
                tuple(evidence_ids),
            ).fetchall()
        content_by_id = {str(row["id"]): str(row["content"]) for row in rows}
        return tuple(
            content_by_id[value] for value in evidence_ids if value in content_by_id
        )

    def evidence_items_for_subject(
        self,
        subject: str,
        *,
        limit: int = 100,
    ) -> list[EvidenceItem]:
        """Return raw personal evidence for one bounded subject scope."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM evidence_items
                WHERE subject = ? AND source IN (
                  'user_direct', 'user_confirmed', 'social_testimony'
                )
                ORDER BY created_at DESC, id ASC LIMIT ?
                """,
                (subject, max(1, int(limit))),
            ).fetchall()
        return [
            EvidenceItem(
                id=str(row["id"]),
                exchange_id=str(row["exchange_id"]),
                content=str(row["content"]),
                source=EvidenceSource(str(row["source"])),
                created_at=float(row["created_at"]),
                session_id=str(row["session_id"]),
                temporal_scope=str(row["temporal_scope"]),
                subject=str(row["subject"]),
            )
            for row in rows
        ]

    def evidence_items_by_id(
        self,
        evidence_ids: Sequence[str],
    ) -> list[EvidenceItem]:
        """Load an explicitly bounded evidence set in caller-supplied order."""
        if not evidence_ids:
            return []
        placeholders = ",".join("?" for _ in evidence_ids)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM evidence_items WHERE id IN ({placeholders})",
                tuple(evidence_ids),
            ).fetchall()
        by_id = {str(row["id"]): row for row in rows}
        return [
            EvidenceItem(
                id=str(by_id[value]["id"]),
                exchange_id=str(by_id[value]["exchange_id"]),
                content=str(by_id[value]["content"]),
                source=EvidenceSource(str(by_id[value]["source"])),
                created_at=float(by_id[value]["created_at"]),
                session_id=str(by_id[value]["session_id"]),
                temporal_scope=str(by_id[value]["temporal_scope"]),
                subject=str(by_id[value]["subject"]),
            )
            for value in evidence_ids
            if value in by_id
        ]

    def set_claim_state(
        self,
        claim_id: str,
        state: ClaimState | str,
        *,
        reason: str,
    ) -> bool:
        """Apply an auditable user-requested claim state transition."""
        resolved_state = ClaimState(state)
        if resolved_state not in {ClaimState.ACTIVE, ClaimState.SUPPRESSED}:
            raise ValueError("unsupported user claim state")
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE personal_claims SET state = ?, updated_at = ?
                WHERE id = ? AND state IN ('active', 'suppressed')
                """,
                (resolved_state.value, now, claim_id),
            )
            if updated.rowcount != 1:
                return False
            connection.execute(
                """
                INSERT INTO memory_feedback (
                  id, subject_id, action, user_text, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    claim_id,
                    resolved_state.value,
                    reason,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO memory_decisions (
                  id, subject_id, subject_type, operation, reason_code,
                  affected_ids_json, details_json, created_at
                ) VALUES (?, ?, 'claim', 'no_op', ?, ?, '{}', ?)
                """,
                (
                    str(uuid.uuid4()),
                    claim_id,
                    f"user_{resolved_state.value}",
                    json.dumps([claim_id], separators=(",", ":")),
                    now,
                ),
            )
        return True

    def delete_claim_subject(
        self,
        claim_id: str,
        *,
        include_raw_evidence: bool = False,
    ) -> dict[str, int]:
        """Delete one claim while preserving its raw evidence unless requested."""
        counts = {"personal_claims": 0, "evidence_items": 0, "exchanges": 0}
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            evidence_rows = connection.execute(
                "SELECT evidence_id FROM claim_evidence_links WHERE claim_id = ?",
                (claim_id,),
            ).fetchall()
            evidence_ids = [str(row["evidence_id"]) for row in evidence_rows]
            if include_raw_evidence and evidence_ids:
                placeholders = ",".join("?" for _ in evidence_ids)
                dependency = connection.execute(
                    f"""
                    SELECT (
                      (SELECT COUNT(*) FROM claim_evidence_links
                       WHERE claim_id != ? AND evidence_id IN ({placeholders})) +
                      (SELECT COUNT(*) FROM schema_evidence_links
                       WHERE evidence_id IN ({placeholders})) +
                      (SELECT COUNT(*) FROM schema_conflicts
                       WHERE evidence_id IN ({placeholders})) +
                      (SELECT COUNT(*) FROM memory_feedback
                       WHERE evidence_id IN ({placeholders})) +
                      (SELECT COUNT(*) FROM pending_user_questions
                       WHERE answer_evidence_id IN ({placeholders}))
                    ) AS count
                    """,
                    (
                        claim_id,
                        *evidence_ids,
                        *evidence_ids,
                        *evidence_ids,
                        *evidence_ids,
                        *evidence_ids,
                    ),
                ).fetchone()
                if dependency is not None and int(dependency["count"]) > 0:
                    raise ValueError("raw evidence is linked to another memory record")
            connection.execute(
                """
                UPDATE personal_claims SET supersedes_id = NULL
                WHERE supersedes_id = ?
                """,
                (claim_id,),
            )
            connection.execute(
                "DELETE FROM claim_evidence_links WHERE claim_id = ?",
                (claim_id,),
            )
            counts["personal_claims"] = connection.execute(
                "DELETE FROM personal_claims WHERE id = ?",
                (claim_id,),
            ).rowcount
            if include_raw_evidence:
                for evidence_id in evidence_ids:
                    evidence = connection.execute(
                        """
                        SELECT exchange_id, candidate_id FROM evidence_items
                        WHERE id = ?
                        """,
                        (evidence_id,),
                    ).fetchone()
                    if evidence is None:
                        continue
                    exchange_id = str(evidence["exchange_id"])
                    candidate_id = str(evidence["candidate_id"] or "")
                    counts["evidence_items"] += connection.execute(
                        "DELETE FROM evidence_items WHERE id = ?",
                        (evidence_id,),
                    ).rowcount
                    if candidate_id:
                        connection.execute(
                            "DELETE FROM memory_candidates WHERE id = ?",
                            (candidate_id,),
                        )
                    remaining = connection.execute(
                        """
                        SELECT
                          (SELECT COUNT(*) FROM evidence_items WHERE exchange_id = ?) +
                          (SELECT COUNT(*) FROM memory_candidates WHERE exchange_id = ?)
                          AS count
                        """,
                        (exchange_id, exchange_id),
                    ).fetchone()
                    if remaining is not None and int(remaining["count"]) == 0:
                        connection.execute(
                            """
                            DELETE FROM candidate_extraction_runs
                            WHERE exchange_id = ?
                            """,
                            (exchange_id,),
                        )
                        counts["exchanges"] += connection.execute(
                            "DELETE FROM conversation_exchanges WHERE id = ?",
                            (exchange_id,),
                        ).rowcount
        return counts

    def personal_table_counts(self) -> dict[str, int]:
        """Return local row counts without exposing any stored content."""
        tables = (
            "conversation_exchanges",
            "candidate_extraction_runs",
            "memory_candidates",
            "evidence_items",
            "personal_claims",
            "claim_evidence_links",
            "personal_schemas",
            "schema_versions",
            "schema_evidence_links",
            "schema_conflicts",
            "insight_candidates",
            "external_knowledge_items",
            "memory_jobs",
            "pending_user_questions",
            "memory_decisions",
            "memory_feedback",
        )
        with self._lock, self._connect() as connection:
            return {
                table: int(
                    connection.execute(
                        f"SELECT COUNT(*) AS count FROM {table}"
                    ).fetchone()["count"]
                )
                for table in tables
            }

    def delete_all_personal_data(self) -> dict[str, int]:
        """Delete personal-memory rows in foreign-key-safe dependency order."""
        counts = self.personal_table_counts()
        tables = (
            "pending_user_questions",
            "memory_feedback",
            "memory_decisions",
            "memory_jobs",
            "candidate_extraction_runs",
            "schema_conflicts",
            "schema_evidence_links",
            "schema_versions",
            "personal_schemas",
            "insight_candidates",
            "external_knowledge_items",
            "claim_evidence_links",
            "personal_claims",
            "evidence_items",
            "memory_candidates",
            "conversation_exchanges",
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for table in tables:
                connection.execute(f"DELETE FROM {table}")
        return counts

    def decision_count(self, *, subject_id: str) -> int:
        """Count append-only decisions for idempotency and audit checks."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM memory_decisions WHERE subject_id = ?",
                (subject_id,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def evidence_count(self, *, candidate_id: str) -> int:
        """Count evidence rows derived from one provisional candidate."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM evidence_items WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def create_pending_question(
        self,
        *,
        subject_id: str,
        question: str,
        decision_effect: str,
    ) -> dict[str, Any]:
        """Create at most one open clarification for a memory subject."""
        now = time.time()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM pending_user_questions
                WHERE subject_id = ? AND state IN ('pending', 'unresolved')
                ORDER BY created_at DESC LIMIT 1
                """,
                (subject_id,),
            ).fetchone()
            if row is None:
                question_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT INTO pending_user_questions (
                      id, subject_id, question, decision_effect, state,
                      created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (question_id, subject_id, question, decision_effect, now, now),
                )
                row = connection.execute(
                    "SELECT * FROM pending_user_questions WHERE id = ?",
                    (question_id,),
                ).fetchone()
        if row is None:  # pragma: no cover - SQLite transaction guarantee
            raise RuntimeError("pending question could not be read")
        return dict(row)

    def answer_pending_question(
        self,
        question_id: str,
        *,
        answer_text: str,
        answer_evidence_id: str = "",
        unresolved: bool = False,
        cooldown_seconds: float = 0.0,
    ) -> dict[str, Any] | None:
        """Record a user answer while preserving uncertainty and cooldown."""
        now = time.time()
        state = "unresolved" if unresolved else "answered"
        cooldown_until = now + max(0.0, float(cooldown_seconds)) if unresolved else 0
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE pending_user_questions
                SET state = ?, answer_evidence_id = ?, cooldown_until = ?,
                    updated_at = ?
                WHERE id = ? AND state IN ('pending', 'unresolved')
                """,
                (
                    state,
                    answer_evidence_id or None,
                    cooldown_until,
                    now,
                    question_id,
                ),
            )
            if updated.rowcount != 1:
                return None
            subject = connection.execute(
                "SELECT subject_id FROM pending_user_questions WHERE id = ?",
                (question_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO memory_feedback (
                  id, subject_id, action, user_text, evidence_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    str(subject["subject_id"]),
                    state,
                    answer_text,
                    answer_evidence_id or None,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO memory_decisions (
                  id, subject_id, subject_type, operation, reason_code,
                  affected_ids_json, details_json, created_at
                ) VALUES (?, ?, 'question', 'no_op', ?, ?, '{}', ?)
                """,
                (
                    str(uuid.uuid4()),
                    str(subject["subject_id"]),
                    f"user_question_{state}",
                    json.dumps([question_id], separators=(",", ":")),
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM pending_user_questions WHERE id = ?",
                (question_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def next_pending_question(self) -> dict[str, Any] | None:
        """Return one askable clarification, respecting unresolved cooldowns."""
        now = time.time()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM pending_user_questions
                WHERE state = 'pending'
                   OR (state = 'unresolved' AND cooldown_until <= ?)
                ORDER BY created_at ASC, id ASC LIMIT 1
                """,
                (now,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_pending_question(self, question_id: str) -> dict[str, Any] | None:
        """Return one clarification row without changing its state."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pending_user_questions WHERE id = ?",
                (question_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def store_external_knowledge(
        self,
        *,
        request_id: str,
        provider_id: str,
        source_url: str,
        query_text: str,
        summary: str,
        consent_id: str,
        deidentified: bool,
    ) -> str:
        """Store general outside knowledge without creating personal evidence."""
        item_id = str(uuid.uuid4())
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO external_knowledge_items (
                  id, request_id, provider_id, source_url, query_text, summary,
                  consent_id, deidentified, retrieved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    request_id,
                    provider_id,
                    source_url,
                    query_text,
                    summary,
                    consent_id,
                    int(bool(deidentified)),
                    time.time(),
                ),
            )
        return item_id

    def store_insight_candidate(
        self,
        *,
        content: str,
        scope: str = "user",
        operation: AdaptationOperation | str,
        support_evidence_ids: Sequence[str],
        counter_evidence_ids: Sequence[str],
        uncertainties: Sequence[str],
        requires_user_confirmation: bool,
    ) -> InsightCandidate:
        """Persist one model proposal separately from active personal schemas."""
        insight_id = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO insight_candidates (
                  id, content, scope, operation, state, support_evidence_json,
                  counter_evidence_json, uncertainties_json,
                  requires_user_confirmation, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?)
                """,
                (
                    insight_id,
                    content,
                    scope,
                    AdaptationOperation(operation).value,
                    json.dumps(tuple(support_evidence_ids), separators=(",", ":")),
                    json.dumps(tuple(counter_evidence_ids), separators=(",", ":")),
                    json.dumps(tuple(uncertainties), separators=(",", ":")),
                    int(requires_user_confirmation),
                    now,
                    now,
                ),
            )
        result = self.get_insight_candidate(insight_id)
        if result is None:  # pragma: no cover - SQLite transaction guarantee
            raise RuntimeError("insight candidate could not be read")
        return result

    def get_insight_candidate(self, insight_id: str) -> InsightCandidate | None:
        """Return one isolated reflection proposal by identifier."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM insight_candidates WHERE id = ?",
                (insight_id,),
            ).fetchone()
        if row is None:
            return None
        return InsightCandidate(
            id=str(row["id"]),
            content=str(row["content"]),
            scope=str(row["scope"]),
            operation=AdaptationOperation(str(row["operation"])),
            state=InsightState(str(row["state"])),
            support_evidence_ids=tuple(
                json.loads(str(row["support_evidence_json"]))
            ),
            counter_evidence_ids=tuple(
                json.loads(str(row["counter_evidence_json"]))
            ),
            created_at=float(row["created_at"]),
            requires_user_confirmation=bool(row["requires_user_confirmation"]),
            uncertainties=tuple(json.loads(str(row["uncertainties_json"]))),
        )

    def set_insight_state(
        self,
        insight_id: str,
        state: InsightState | str,
    ) -> bool:
        """Advance one insight without altering its immutable proposal content."""
        resolved = InsightState(state)
        with self._lock, self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE insight_candidates SET state = ?, updated_at = ?
                WHERE id = ?
                """,
                (resolved.value, time.time(), insight_id),
            )
        return updated.rowcount == 1

    def external_knowledge_count(self) -> int:
        """Return isolated external-knowledge row count for safety checks."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM external_knowledge_items"
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def active_schema_count(self) -> int:
        """Return active personal-schema count without exposing their content."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM personal_schemas WHERE state = 'active'"
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def unresolved_conflict_count(self, *, schema_id: str = "") -> int:
        """Return unresolved conflict count without exposing personal content."""
        clause = ""
        params: tuple[str, ...] = ()
        if schema_id:
            clause = " AND schema_id = ?"
            params = (schema_id,)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS count FROM schema_conflicts
                WHERE resolution_state = 'unresolved'{clause}
                """,
                params,
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def unresolved_conflicts_for_subject(
        self,
        subject_scope: str,
    ) -> list[dict[str, Any]]:
        """Return bounded conflict metadata and raw evidence for one schema scope."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.schema_id, c.evidence_id, c.prediction_error,
                       c.created_at, e.session_id
                FROM schema_conflicts AS c
                JOIN personal_schemas AS s ON s.id = c.schema_id
                JOIN evidence_items AS e ON e.id = c.evidence_id
                WHERE c.resolution_state = 'unresolved'
                  AND s.subject_scope = ?
                ORDER BY c.created_at DESC, c.id ASC
                LIMIT 12
                """,
                (subject_scope,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_active_schemas(self) -> list[PersonalSchema]:
        """Return response-eligible stable schemas with their current versions."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.*, v.content AS version_content, v.conditions_json,
                       v.support_mass, v.counter_mass, v.stability, v.plasticity,
                       v.scope_fit, v.source_quality, v.temporal_validity
                FROM personal_schemas AS s
                JOIN schema_versions AS v ON v.id = s.current_version_id
                WHERE s.state = 'active'
                  AND s.maturity NOT IN ('tentative', 'challenged', 'retired')
                  AND (s.maturity = 'stable' OR s.user_confirmed = 1)
                  AND NOT EXISTS (
                    SELECT 1 FROM schema_conflicts AS c
                    WHERE c.schema_id = s.id
                      AND c.resolution_state = 'unresolved'
                  )
                ORDER BY s.updated_at DESC, s.id ASC
                """
            ).fetchall()
        schemas = []
        for row in rows:
            schemas.append(
                PersonalSchema(
                    id=str(row["id"]),
                    content=str(row["version_content"]),
                    maturity=SchemaMaturity(str(row["maturity"])),
                    state=ClaimState(str(row["state"])),
                    current_version_id=str(row["current_version_id"]),
                    evidence_mass=EvidenceMass(
                        support_mass=float(row["support_mass"]),
                        counter_mass=float(row["counter_mass"]),
                        stability=float(row["stability"]),
                        plasticity=float(row["plasticity"]),
                        scope_fit=float(row["scope_fit"]),
                        source_quality=float(row["source_quality"]),
                        temporal_validity=float(row["temporal_validity"]),
                    ),
                    created_at=float(row["created_at"]),
                    updated_at=float(row["updated_at"]),
                    conditions=tuple(json.loads(str(row["conditions_json"]))),
                    subject_scope=str(row["subject_scope"]),
                    broad_interpretation=bool(row["broad_interpretation"]),
                    user_confirmed=bool(row["user_confirmed"]),
                )
            )
        return schemas

    def get_schema(self, schema_id: str) -> PersonalSchema | None:
        """Return one schema in any maturity state with its current version."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT s.*, v.content AS version_content, v.conditions_json,
                       v.support_mass, v.counter_mass, v.stability, v.plasticity,
                       v.scope_fit, v.source_quality, v.temporal_validity
                FROM personal_schemas AS s
                JOIN schema_versions AS v ON v.id = s.current_version_id
                WHERE s.id = ?
                """,
                (schema_id,),
            ).fetchone()
        if row is None:
            return None
        state_value = str(row["state"])
        if state_value == "retired":
            state_value = ClaimState.ARCHIVED.value
        return PersonalSchema(
            id=str(row["id"]),
            content=str(row["version_content"]),
            maturity=SchemaMaturity(str(row["maturity"])),
            state=ClaimState(state_value),
            current_version_id=str(row["current_version_id"]),
            evidence_mass=EvidenceMass(
                support_mass=float(row["support_mass"]),
                counter_mass=float(row["counter_mass"]),
                stability=float(row["stability"]),
                plasticity=float(row["plasticity"]),
                scope_fit=float(row["scope_fit"]),
                source_quality=float(row["source_quality"]),
                temporal_validity=float(row["temporal_validity"]),
            ),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            conditions=tuple(json.loads(str(row["conditions_json"]))),
            subject_scope=str(row["subject_scope"]),
            broad_interpretation=bool(row["broad_interpretation"]),
            user_confirmed=bool(row["user_confirmed"]),
        )

    def list_schemas(self) -> list[PersonalSchema]:
        """Return every versioned schema for user inspection."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id FROM personal_schemas
                WHERE current_version_id IS NOT NULL
                ORDER BY updated_at DESC, id ASC
                """
            ).fetchall()
        return [
            schema
            for row in rows
            if (schema := self.get_schema(str(row["id"]))) is not None
        ]

    def schema_evidence_texts(
        self,
        schema_id: str,
        *,
        relation: str,
    ) -> tuple[str, ...]:
        """Return evidence attached to the current version of one schema."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.content FROM personal_schemas AS s
                JOIN schema_evidence_links AS l
                  ON l.schema_version_id = s.current_version_id
                JOIN evidence_items AS e ON e.id = l.evidence_id
                WHERE s.id = ? AND l.relation = ?
                ORDER BY l.created_at ASC, e.id ASC
                """,
                (schema_id, relation),
            ).fetchall()
        return tuple(str(row["content"]) for row in rows)

    def schema_evidence_ids(self, schema_id: str) -> tuple[str, ...]:
        """Return current raw evidence links for version-preserving corrections."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT l.evidence_id FROM personal_schemas AS s
                JOIN schema_evidence_links AS l
                  ON l.schema_version_id = s.current_version_id
                WHERE s.id = ? ORDER BY l.created_at ASC, l.evidence_id ASC
                """,
                (schema_id,),
            ).fetchall()
        return tuple(str(row["evidence_id"]) for row in rows)

    def record_user_confirmed_evidence(
        self,
        *,
        user_text: str,
        content: str,
        subject: str,
    ) -> EvidenceItem:
        """Record an explicit memory-control statement as new raw evidence."""
        exchange_id = str(uuid.uuid4())
        exchange = self.record_exchange(
            exchange_id=exchange_id,
            user_text=user_text,
            assistant_text="",
            source="user_memory_control",
        )
        evidence_id = str(uuid.uuid4())
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evidence_items (
                  id, exchange_id, source, content, temporal_scope,
                  subject, session_id, created_at
                ) VALUES (?, ?, 'user_confirmed', ?, 'until_changed', ?, '', ?)
                """,
                (evidence_id, exchange.id, content, subject, time.time()),
            )
        return self.evidence_items_by_id((evidence_id,))[0]

    def set_schema_state(
        self,
        schema_id: str,
        state: ClaimState | str,
        *,
        reason: str,
    ) -> bool:
        """Apply an auditable user-requested schema visibility transition."""
        resolved = ClaimState(state)
        if resolved not in {ClaimState.ACTIVE, ClaimState.SUPPRESSED}:
            raise ValueError("unsupported user schema state")
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE personal_schemas SET state = ?, updated_at = ?
                WHERE id = ? AND state IN ('active', 'suppressed')
                """,
                (resolved.value, now, schema_id),
            )
            if updated.rowcount != 1:
                return False
            connection.execute(
                """
                INSERT INTO memory_feedback (
                  id, subject_id, action, user_text, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    schema_id,
                    resolved.value,
                    reason,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO memory_decisions (
                  id, subject_id, subject_type, operation, reason_code,
                  affected_ids_json, details_json, created_at
                ) VALUES (?, ?, 'schema', 'no_op', ?, ?, '{}', ?)
                """,
                (
                    str(uuid.uuid4()),
                    schema_id,
                    f"user_{resolved.value}",
                    json.dumps([schema_id], separators=(",", ":")),
                    now,
                ),
            )
        return True

    def delete_schema_subject(self, schema_id: str) -> dict[str, int]:
        """Delete one derived schema while retaining its raw user evidence."""
        counts = {"personal_schemas": 0, "schema_versions": 0}
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            version_rows = connection.execute(
                "SELECT id FROM schema_versions WHERE schema_id = ?",
                (schema_id,),
            ).fetchall()
            version_ids = [str(row["id"]) for row in version_rows]
            connection.execute(
                "DELETE FROM schema_conflicts WHERE schema_id = ?",
                (schema_id,),
            )
            connection.execute(
                "UPDATE personal_schemas SET current_version_id = NULL WHERE id = ?",
                (schema_id,),
            )
            for version_id in version_ids:
                connection.execute(
                    "DELETE FROM schema_evidence_links WHERE schema_version_id = ?",
                    (version_id,),
                )
            counts["schema_versions"] = connection.execute(
                "DELETE FROM schema_versions WHERE schema_id = ?",
                (schema_id,),
            ).rowcount
            counts["personal_schemas"] = connection.execute(
                "DELETE FROM personal_schemas WHERE id = ?",
                (schema_id,),
            ).rowcount
        return counts

    def apply_schema_accommodation(
        self,
        *,
        content: str,
        operation: AdaptationOperation | str,
        support_evidence_ids: Sequence[str],
        counter_evidence_ids: Sequence[str] = (),
        conditions: Sequence[str] = (),
        subject_scope: str = "user",
        broad_interpretation: bool = False,
        user_confirmed: bool = False,
        target_schema_id: str = "",
    ) -> PersonalSchema | None:
        """Atomically create or version one evidence-grounded schema."""
        resolved = AdaptationOperation(operation)
        allowed = {
            AdaptationOperation.ACCOMMODATE_CREATE,
            AdaptationOperation.ACCOMMODATE_REFINE,
            AdaptationOperation.ACCOMMODATE_SPLIT,
            AdaptationOperation.ACCOMMODATE_SUPERSEDE,
        }
        if resolved not in allowed or not support_evidence_ids:
            return None
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cited_ids = tuple(
                dict.fromkeys((*support_evidence_ids, *counter_evidence_ids))
            )
            placeholders = ",".join("?" for _ in cited_ids)
            existing = connection.execute(
                f"SELECT id FROM evidence_items WHERE id IN ({placeholders})",
                cited_ids,
            ).fetchall()
            if {str(row["id"]) for row in existing} != set(cited_ids):
                return None

            create_new = resolved in {
                AdaptationOperation.ACCOMMODATE_CREATE,
                AdaptationOperation.ACCOMMODATE_SPLIT,
                AdaptationOperation.ACCOMMODATE_SUPERSEDE,
            }
            if not create_new:
                target = connection.execute(
                    "SELECT * FROM personal_schemas WHERE id = ? AND state = 'active'",
                    (target_schema_id,),
                ).fetchone()
                if target is None:
                    return None
                schema_id = target_schema_id
                version_number = int(
                    connection.execute(
                        """
                        SELECT COALESCE(MAX(version_number), 0) + 1 AS number
                        FROM schema_versions WHERE schema_id = ?
                        """,
                        (schema_id,),
                    ).fetchone()["number"]
                )
            else:
                schema_id = str(uuid.uuid4())
                version_number = 1
                maturity = (
                    SchemaMaturity.STABLE
                    if user_confirmed
                    else SchemaMaturity.EMERGING
                )
                connection.execute(
                    """
                    INSERT INTO personal_schemas (
                      id, content, state, maturity, subject_scope,
                      broad_interpretation, user_confirmed, created_at, updated_at
                    ) VALUES (?, ?, 'active', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        schema_id,
                        content,
                        maturity.value,
                        subject_scope,
                        int(broad_interpretation),
                        int(user_confirmed),
                        now,
                        now,
                    ),
                )
                if (
                    resolved is AdaptationOperation.ACCOMMODATE_SUPERSEDE
                    and target_schema_id
                ):
                    connection.execute(
                        """
                        UPDATE personal_schemas
                        SET state = 'retired', maturity = 'retired', updated_at = ?
                        WHERE id = ? AND state = 'active'
                        """,
                        (now, target_schema_id),
                    )

            version_id = str(uuid.uuid4())
            support_mass = min(1.0, len(set(support_evidence_ids)) / 3.0)
            counter_mass = min(1.0, len(set(counter_evidence_ids)) / 3.0)
            connection.execute(
                """
                INSERT INTO schema_versions (
                  id, schema_id, version_number, content, conditions_json,
                  support_mass, counter_mass, stability, plasticity, scope_fit,
                  source_quality, temporal_validity, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    schema_id,
                    version_number,
                    content,
                    json.dumps(tuple(conditions), separators=(",", ":")),
                    support_mass,
                    counter_mass,
                    1.0 if user_confirmed else 0.5,
                    0.25 if user_confirmed else 0.75,
                    1.0,
                    1.0,
                    1.0,
                    now,
                ),
            )
            for evidence_id in support_evidence_ids:
                connection.execute(
                    """
                    INSERT INTO schema_evidence_links (
                      schema_version_id, evidence_id, relation, created_at
                    ) VALUES (?, ?, 'support', ?)
                    """,
                    (version_id, evidence_id, now),
                )
            for evidence_id in counter_evidence_ids:
                connection.execute(
                    """
                    INSERT INTO schema_evidence_links (
                      schema_version_id, evidence_id, relation, created_at
                    ) VALUES (?, ?, 'counter', ?)
                    """,
                    (version_id, evidence_id, now),
                )
            connection.execute(
                """
                UPDATE personal_schemas
                SET content = ?, current_version_id = ?,
                    user_confirmed = CASE WHEN ? THEN 1 ELSE user_confirmed END,
                    maturity = CASE WHEN ? THEN 'stable' ELSE maturity END,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    content,
                    version_id,
                    int(user_confirmed),
                    int(user_confirmed),
                    now,
                    schema_id,
                ),
            )
            if counter_evidence_ids:
                counter_placeholders = ",".join("?" for _ in counter_evidence_ids)
                connection.execute(
                    f"""
                    UPDATE schema_conflicts
                    SET resolution_state = 'resolved', resolved_at = ?
                    WHERE schema_id = ?
                      AND evidence_id IN ({counter_placeholders})
                    """,
                    (now, target_schema_id or schema_id, *counter_evidence_ids),
                )
            connection.execute(
                """
                INSERT INTO memory_decisions (
                  id, subject_id, subject_type, operation, reason_code,
                  affected_ids_json, details_json, created_at
                ) VALUES (?, ?, 'schema', ?, 'validated_insight', ?, '{}', ?)
                """,
                (
                    str(uuid.uuid4()),
                    schema_id,
                    resolved.value,
                    json.dumps([version_id], separators=(",", ":")),
                    now,
                ),
            )
        return self.get_schema(schema_id)

    def schema_version_count(self, schema_id: str) -> int:
        """Return retained version count for accommodation tests and inspection."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM schema_versions WHERE schema_id = ?",
                (schema_id,),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def schema_previous_versions(self, schema_id: str) -> tuple[str, ...]:
        """Return prior schema text without exposing unrelated versions."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT v.content FROM schema_versions AS v
                JOIN personal_schemas AS s ON s.id = v.schema_id
                WHERE v.schema_id = ? AND v.id != s.current_version_id
                ORDER BY v.version_number DESC
                """,
                (schema_id,),
            ).fetchall()
        return tuple(str(row["content"]) for row in rows)

    def reject_candidate(self, candidate_id: str, *, reason_code: str) -> bool:
        """Reject one pending candidate and audit the reason atomically."""
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE memory_candidates
                SET status = 'rejected', updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now, candidate_id),
            )
            if updated.rowcount != 1:
                return False
            connection.execute(
                """
                INSERT INTO memory_decisions (
                  id, subject_id, subject_type, operation, reason_code,
                  affected_ids_json, details_json, created_at
                ) VALUES (?, ?, 'candidate', 'no_op', ?, '[]', '{}', ?)
                """,
                (str(uuid.uuid4()), candidate_id, reason_code, now),
            )
        return True

    def apply_candidate_decision(
        self,
        candidate_id: str,
        *,
        operation: AdaptationOperation | str,
        reason_code: str,
        superseded_claim_ids: Sequence[str] = (),
        target_schema_ids: Sequence[str] = (),
    ) -> dict[str, Any] | None:
        """Persist evidence, claims, supersession, and audit atomically."""
        now = time.time()
        resolved_operation = AdaptationOperation(operation)
        atomic_claim_kinds = {
            CandidateKind.FACT,
            CandidateKind.PREFERENCE,
            CandidateKind.CONSTRAINT,
            CandidateKind.CORRECTION,
            CandidateKind.ROLE_PREFERENCE,
            CandidateKind.CAPABILITY_BOUNDARY,
        }
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM memory_candidates WHERE id = ? AND status = 'pending'",
                (candidate_id,),
            ).fetchone()
            if row is None:
                return None
            candidate = self._candidate_from_row(row)
            exchange = connection.execute(
                "SELECT session_id FROM conversation_exchanges WHERE id = ?",
                (candidate.exchange_id,),
            ).fetchone()
            if exchange is None:
                raise RuntimeError("candidate evidence exchange is missing")

            evidence_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO evidence_items (
                  id, exchange_id, candidate_id, source, content, temporal_scope,
                  subject, session_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    candidate.exchange_id,
                    candidate.id,
                    candidate.source.value,
                    candidate.evidence_excerpt,
                    candidate.temporal_scope,
                    candidate.subject,
                    str(exchange["session_id"]),
                    now,
                ),
            )

            superseded = tuple(
                dict.fromkeys(str(value) for value in superseded_claim_ids)
            )
            for claim_id in superseded:
                connection.execute(
                    """
                    UPDATE personal_claims
                    SET state = 'superseded', updated_at = ?
                    WHERE id = ? AND state = 'active'
                    """,
                    (now, claim_id),
                )

            affected_ids: list[str] = [evidence_id, *superseded]
            for schema_id in dict.fromkeys(str(value) for value in target_schema_ids):
                schema = connection.execute(
                    """
                    SELECT current_version_id FROM personal_schemas
                    WHERE id = ? AND state = 'active'
                    """,
                    (schema_id,),
                ).fetchone()
                if schema is None or not schema["current_version_id"]:
                    continue
                if resolved_operation is AdaptationOperation.HOLD_DISEQUILIBRIUM:
                    conflict_id = str(uuid.uuid4())
                    connection.execute(
                        """
                        INSERT INTO schema_conflicts (
                          id, schema_id, schema_version_id, evidence_id,
                          conflict_type, prediction_error, source_quality,
                          resolution_state, created_at
                        ) VALUES (?, ?, ?, ?, 'contradiction', ?, ?,
                                  'unresolved', ?)
                        """,
                        (
                            conflict_id,
                            schema_id,
                            str(schema["current_version_id"]),
                            evidence_id,
                            candidate.confidence,
                            candidate.confidence,
                            now,
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE personal_schemas
                        SET maturity = 'challenged', updated_at = ? WHERE id = ?
                        """,
                        (now, schema_id),
                    )
                    affected_ids.append(conflict_id)
                elif resolved_operation in {
                    AdaptationOperation.ASSIMILATE_REINFORCE,
                    AdaptationOperation.ASSIMILATE_AS_EXCEPTION,
                    AdaptationOperation.ASSIMILATE_LINK_ONLY,
                }:
                    relation = (
                        "exception"
                        if resolved_operation
                        is AdaptationOperation.ASSIMILATE_AS_EXCEPTION
                        else "support"
                    )
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO schema_evidence_links (
                          schema_version_id, evidence_id, relation, created_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (str(schema["current_version_id"]), evidence_id, relation, now),
                    )
                    affected_ids.append(schema_id)
            claim_id = ""
            if candidate.kind in atomic_claim_kinds:
                claim_id = str(uuid.uuid4())
                expires_at = (
                    now + (7 * 24 * 60 * 60)
                    if candidate.temporal_scope == "current"
                    else 0.0
                )
                connection.execute(
                    """
                    INSERT INTO personal_claims (
                      id, kind, content, state, source, temporal_scope,
                      subject_scope, supersedes_id, created_at, updated_at,
                      expires_at
                    ) VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim_id,
                        candidate.kind.value,
                        candidate.content,
                        candidate.source.value,
                        candidate.temporal_scope,
                        candidate.subject,
                        superseded[0] if superseded else None,
                        now,
                        now,
                        expires_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO claim_evidence_links(claim_id, evidence_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (claim_id, evidence_id, now),
                )
                affected_ids.append(claim_id)

            connection.execute(
                """
                UPDATE memory_candidates
                SET status = 'accepted', updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (now, candidate_id),
            )
            decision_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO memory_decisions (
                  id, subject_id, subject_type, operation, reason_code,
                  affected_ids_json, details_json, created_at
                ) VALUES (?, ?, 'candidate', ?, ?, ?, '{}', ?)
                """,
                (
                    decision_id,
                    candidate_id,
                    resolved_operation.value,
                    reason_code,
                    json.dumps(affected_ids, separators=(",", ":")),
                    now,
                ),
            )
        return {
            "decision_id": decision_id,
            "evidence_id": evidence_id,
            "claim_id": claim_id,
            "superseded_claim_ids": superseded,
        }

    def claim_candidate_job(self, exchange_id: str) -> ConversationExchange | None:
        """Atomically move retryable work to processing and return its evidence."""
        now = time.time()
        with self._lock, self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE conversation_exchanges
                SET candidate_state = 'processing',
                    candidate_attempts = candidate_attempts + 1,
                    candidate_error = '', candidate_claimed_at = ?,
                    candidate_next_attempt_at = 0
                WHERE id = ? AND candidate_state IN ('pending', 'failed')
                """,
                (now, exchange_id),
            )
            if updated.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT * FROM conversation_exchanges WHERE id = ?", (exchange_id,)
            ).fetchone()
        return self._exchange_from_row(row) if row is not None else None

    def complete_candidate_job(
        self,
        exchange_id: str,
        drafts: Sequence[CandidateDraft],
        *,
        engine_id: str,
        extractor_version: str,
    ) -> list[MemoryCandidate]:
        """Persist provisional candidates and finish their job in one transaction."""
        now = time.time()
        persisted: list[MemoryCandidate] = []
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT candidate_state FROM conversation_exchanges WHERE id = ?",
                (exchange_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown exchange: {exchange_id}")
            if row["candidate_state"] != "processing":
                raise RuntimeError("candidate job must be processing before completion")

            for draft in drafts:
                candidate_id = str(uuid.uuid4())
                connection.execute(
                    """
                    INSERT INTO memory_candidates (
                        id, exchange_id, kind, content, evidence_excerpt, importance,
                        confidence, status,
                        engine_id, extractor_version, source, temporal_scope, subject,
                        target_claim_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(exchange_id, kind, content) DO NOTHING
                    """,
                    (
                        candidate_id,
                        exchange_id,
                        draft.kind,
                        draft.content,
                        draft.evidence_excerpt,
                        draft.importance,
                        draft.confidence,
                        engine_id,
                        extractor_version,
                        draft.source,
                        draft.temporal_scope,
                        draft.subject,
                        draft.target_claim_id,
                        now,
                        now,
                    ),
                )
                candidate_row = connection.execute(
                    """
                    SELECT * FROM memory_candidates
                    WHERE exchange_id = ? AND kind = ? AND content = ?
                    """,
                    (exchange_id, draft.kind, draft.content),
                ).fetchone()
                if candidate_row is not None:
                    persisted.append(self._candidate_from_row(candidate_row))

            connection.execute(
                """
                UPDATE conversation_exchanges
                SET candidate_state = 'complete', candidate_error = '',
                    candidate_claimed_at = 0,
                    candidate_next_attempt_at = 0,
                    candidate_extractor_version = ?
                WHERE id = ?
                """,
                (extractor_version, exchange_id),
            )
            connection.execute(
                """
                INSERT INTO candidate_extraction_runs (
                    exchange_id, extractor_version, candidate_count, completed_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(exchange_id, extractor_version) DO UPDATE SET
                    candidate_count = excluded.candidate_count,
                    completed_at = excluded.completed_at
                """,
                (exchange_id, extractor_version, len(persisted), now),
            )
        return persisted

    def fail_candidate_job(
        self,
        exchange_id: str,
        *,
        error_code: str,
        retry_after: float | None = 0.0,
    ) -> None:
        """Make a claimed job eligible for retry without retaining raw error text."""
        now = time.time()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT candidate_attempts FROM conversation_exchanges WHERE id = ?",
                (exchange_id,),
            ).fetchone()
            if row is None:
                return
            if retry_after is None:
                attempts = max(1, int(row["candidate_attempts"]))
                retry_after = min(60.0, float(2 ** min(attempts - 1, 6)))
            next_attempt_at = now + max(0.0, float(retry_after))
            connection.execute(
                """
                UPDATE conversation_exchanges
                SET candidate_state = 'failed', candidate_error = ?,
                    candidate_claimed_at = 0,
                    candidate_next_attempt_at = ?
                WHERE id = ? AND candidate_state = 'processing'
                """,
                (error_code or "extract_failed", next_attempt_at, exchange_id),
            )

    def release_in_progress_jobs(self) -> int:
        """Return interrupted legacy and developmental jobs to their queues."""
        with self._lock, self._connect() as connection:
            legacy = connection.execute(
                """
                UPDATE conversation_exchanges
                SET candidate_state = 'pending', candidate_claimed_at = 0,
                    candidate_next_attempt_at = 0
                WHERE candidate_state = 'processing'
                """
            )
            jobs = connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'pending', claimed_at = 0, next_attempt_at = 0,
                    updated_at = ?
                WHERE state = 'processing'
                """,
                (time.time(),),
            )
        return max(0, legacy.rowcount) + max(0, jobs.rowcount)

    def pending_exchange_ids(self, *, limit: int) -> list[str]:
        """List durable candidate work that can be tried or retried."""
        count = max(0, int(limit))
        if count == 0:
            return []
        now = time.time()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id FROM conversation_exchanges
                WHERE candidate_state = 'pending'
                   OR (candidate_state = 'failed' AND candidate_next_attempt_at <= ?)
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (now, count),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def recover_stale_candidate_jobs(self, extractor_version: str) -> int:
        """Requeue old zero-candidate completions once for a newer extractor."""
        version = str(extractor_version or "").strip()
        if not version:
            raise ValueError("extractor_version must not be empty")
        now = time.time()
        with self._lock, self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE conversation_exchanges
                SET candidate_state = 'pending', candidate_error = '',
                    candidate_claimed_at = 0, candidate_next_attempt_at = 0
                WHERE candidate_state = 'complete'
                  AND NOT EXISTS (
                    SELECT 1 FROM memory_candidates
                    WHERE memory_candidates.exchange_id = conversation_exchanges.id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM candidate_extraction_runs
                    WHERE candidate_extraction_runs.exchange_id =
                          conversation_exchanges.id
                      AND candidate_extraction_runs.extractor_version = ?
                  )
                """,
                (version,),
            )
            connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'pending', error_code = '', claimed_at = 0,
                    next_attempt_at = 0, updated_at = ?
                WHERE job_type = 'extract_candidates'
                  AND state != 'processing'
                  AND idempotency_key = (
                    'extract_candidates:' || ? || ':' || subject_id
                  )
                  AND EXISTS (
                    SELECT 1 FROM conversation_exchanges
                    WHERE conversation_exchanges.id = memory_jobs.subject_id
                      AND conversation_exchanges.candidate_state = 'pending'
                  )
                """,
                (now, version),
            )
        return max(0, updated.rowcount)

    def recover_pending_candidate_jobs(self) -> list[MemoryJob]:
        """Create idempotent evaluation jobs for candidates from older archives."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id FROM memory_candidates
                WHERE status = 'pending' ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
        return [
            self.enqueue_job(
                job_type="evaluate_candidate",
                subject_id=str(row["id"]),
                idempotency_key=f"evaluate_candidate:{row['id']}",
                priority=90,
            )
            for row in rows
        ]

    def enqueue_job(
        self,
        *,
        job_type: str,
        subject_id: str,
        idempotency_key: str,
        payload: dict[str, Any] | None = None,
        priority: int = 0,
    ) -> MemoryJob:
        """Persist one idempotent unit of work and return the durable row."""
        if (
            not job_type.strip()
            or not subject_id.strip()
            or not idempotency_key.strip()
        ):
            raise ValueError("job type, subject ID, and idempotency key are required")
        now = time.time()
        job_id = str(uuid.uuid4())
        payload_json = json.dumps(
            payload or {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_jobs (
                  id, job_type, subject_id, idempotency_key, payload_json,
                  state, priority, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (
                    job_id,
                    job_type.strip(),
                    subject_id.strip(),
                    idempotency_key.strip(),
                    payload_json,
                    int(priority),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM memory_jobs WHERE idempotency_key = ?",
                (idempotency_key.strip(),),
            ).fetchone()
        if row is None:  # pragma: no cover - SQLite transaction guarantee
            raise RuntimeError("memory job could not be read after enqueue")
        return self._job_from_row(row)

    def pending_job_ids(self, *, limit: int) -> list[str]:
        """List claimable jobs in priority order without changing state."""
        count = max(0, int(limit))
        if count == 0:
            return []
        now = time.time()
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id FROM memory_jobs
                WHERE state = 'pending'
                   OR (state = 'failed' AND next_attempt_at <= ?)
                   OR (state = 'deferred' AND next_attempt_at <= ?)
                ORDER BY priority DESC, created_at ASC, id ASC
                LIMIT ?
                """,
                (now, now, count),
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def get_job(self, job_id: str) -> MemoryJob | None:
        """Return one durable job in any lifecycle state."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM memory_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def claim_job(self, job_id: str) -> MemoryJob | None:
        """Atomically claim a specific queued job after an in-memory wake-up."""
        now = time.time()
        with self._lock, self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'processing', attempts = attempts + 1,
                    error_code = '', claimed_at = ?, next_attempt_at = 0,
                    updated_at = ?
                WHERE id = ? AND (
                  state = 'pending'
                  OR (state = 'failed' AND next_attempt_at <= ?)
                  OR (state = 'deferred' AND next_attempt_at <= ?)
                )
                """,
                (now, now, job_id, now, now),
            )
            if updated.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT * FROM memory_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def claim_next_job(
        self,
        *,
        job_types: Sequence[str] | None = None,
    ) -> MemoryJob | None:
        """Atomically claim the next eligible one-shot job."""
        now = time.time()
        allowed = tuple(str(value).strip() for value in (job_types or ()) if value)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            params: list[Any] = [now, now]
            type_clause = ""
            if allowed:
                placeholders = ",".join("?" for _ in allowed)
                type_clause = f" AND job_type IN ({placeholders})"
                params.extend(allowed)
            row = connection.execute(
                f"""
                SELECT id FROM memory_jobs
                WHERE (state = 'pending'
                   OR (state = 'failed' AND next_attempt_at <= ?)
                   OR (state = 'deferred' AND next_attempt_at <= ?))
                  {type_clause}
                ORDER BY priority DESC, created_at ASC, id ASC
                LIMIT 1
                """,
                params,
            ).fetchone()
            if row is None:
                return None
            job_id = str(row["id"])
            updated = connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'processing', attempts = attempts + 1,
                    error_code = '', claimed_at = ?, next_attempt_at = 0,
                    updated_at = ?
                WHERE id = ? AND state IN ('pending', 'failed', 'deferred')
                """,
                (now, now, job_id),
            )
            if updated.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM memory_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._job_from_row(claimed) if claimed is not None else None

    def complete_job(self, job_id: str) -> None:
        """Mark a claimed job complete exactly once."""
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'complete', error_code = '', claimed_at = 0,
                    next_attempt_at = 0, updated_at = ?
                WHERE id = ? AND state = 'processing'
                """,
                (now, job_id),
            )

    def defer_job(self, job_id: str, *, retry_after: float = 1.0) -> None:
        """Yield heavy work without recording a failure or hot-looping."""
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'deferred', error_code = '', claimed_at = 0,
                    next_attempt_at = ?, updated_at = ?
                WHERE id = ? AND state = 'processing'
                """,
                (now + max(0.1, float(retry_after)), now, job_id),
            )

    def fail_job(
        self,
        job_id: str,
        *,
        error_code: str,
        retry_after: float | None = None,
    ) -> None:
        """Return failed work to a bounded-backoff durable state."""
        now = time.time()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT attempts FROM memory_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return
            if retry_after is None:
                attempts = max(1, int(row["attempts"]))
                retry_after = min(60.0, float(2 ** min(attempts - 1, 6)))
            connection.execute(
                """
                UPDATE memory_jobs
                SET state = 'failed', error_code = ?, claimed_at = 0,
                    next_attempt_at = ?, updated_at = ?
                WHERE id = ? AND state = 'processing'
                """,
                (
                    error_code.strip() or "job_failed",
                    now + max(0.0, float(retry_after)),
                    now,
                    job_id,
                ),
            )


__all__ = ["PersonalMemoryArchive"]
