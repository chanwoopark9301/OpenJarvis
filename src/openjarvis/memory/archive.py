"""SQLite persistence for local personal conversations and candidate work."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Sequence

from openjarvis.memory.personal_models import (
    CandidateDraft,
    ConversationExchange,
    MemoryCandidate,
)

_SCHEMA = """
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
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE(exchange_id, kind, content)
);
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
                        engine_id, extractor_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
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
        """Return interrupted worker jobs to the durable pending queue."""
        with self._lock, self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE conversation_exchanges
                SET candidate_state = 'pending', candidate_claimed_at = 0,
                    candidate_next_attempt_at = 0
                WHERE candidate_state = 'processing'
                """
            )
        return max(0, updated.rowcount)

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


__all__ = ["PersonalMemoryArchive"]
