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
    CandidateDraft,
    ConversationExchange,
    MemoryCandidate,
    MemoryJob,
)

_SCHEMA_VERSION = 2
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
  candidate_next_attempt_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS memory_candidates (
  id TEXT PRIMARY KEY,
  exchange_id TEXT NOT NULL REFERENCES conversation_exchanges(id),
  kind TEXT NOT NULL,
  content TEXT NOT NULL,
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
  supersedes_id TEXT REFERENCES personal_claims(id),
  expires_at REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS personal_schemas (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending',
  maturity TEXT NOT NULL DEFAULT 'tentative',
  current_version_id TEXT,
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
            candidate_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(memory_candidates)"
                ).fetchall()
            }
            candidate_migrations = {
                "source": "TEXT NOT NULL DEFAULT 'user_direct'",
                "temporal_scope": "TEXT NOT NULL DEFAULT 'unspecified'",
                "subject": "TEXT NOT NULL DEFAULT 'user'",
                "target_claim_id": "TEXT NOT NULL DEFAULT ''",
            }
            for name, declaration in candidate_migrations.items():
                if name not in candidate_columns:
                    connection.execute(
                        f"ALTER TABLE memory_candidates ADD COLUMN {name} {declaration}"
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
                        id, exchange_id, kind, content, importance, confidence, status,
                        engine_id, extractor_version, source, temporal_scope, subject,
                        target_claim_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(exchange_id, kind, content) DO NOTHING
                    """,
                    (
                        candidate_id,
                        exchange_id,
                        draft.kind,
                        draft.content,
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
                    candidate_next_attempt_at = 0
                WHERE id = ?
                """,
                (exchange_id,),
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
                ORDER BY priority DESC, created_at ASC, id ASC
                LIMIT ?
                """,
                (now, count),
            ).fetchall()
        return [str(row["id"]) for row in rows]

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
            params: list[Any] = [now]
            type_clause = ""
            if allowed:
                placeholders = ",".join("?" for _ in allowed)
                type_clause = f" AND job_type IN ({placeholders})"
                params.extend(allowed)
            row = connection.execute(
                f"""
                SELECT id FROM memory_jobs
                WHERE (state = 'pending'
                   OR (state = 'failed' AND next_attempt_at <= ?))
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
                WHERE id = ? AND state IN ('pending', 'failed')
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
