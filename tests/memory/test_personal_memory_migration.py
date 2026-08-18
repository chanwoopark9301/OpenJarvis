"""Migration and durable-job tests for the developmental memory archive."""

from __future__ import annotations

import sqlite3

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.personal_models import CandidateDraft

_V1_SCHEMA = """
CREATE TABLE conversation_exchanges (
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

CREATE TABLE memory_candidates (
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


def _create_v1_archive(path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(_V1_SCHEMA)
        connection.execute(
            """
            INSERT INTO conversation_exchanges (
              id, created_at, archived_at, source, user_text, assistant_text,
              content_hash
            ) VALUES ('old-exchange', 1, 1, 'test', 'old user text',
                      'old reply', 'old-hash')
            """
        )
        connection.execute(
            """
            INSERT INTO memory_candidates (
              id, exchange_id, kind, content, importance, confidence, status,
              engine_id, extractor_version, created_at, updated_at
            ) VALUES (
              'old-candidate', 'old-exchange', 'fact', 'old user text',
              0.5, 0.5, 'pending', 'old-engine', 'v1', 1, 1
            )
            """
        )


def test_existing_candidate_archive_migrates_in_place(tmp_path):
    """Opening a v1 database must preserve evidence while adding v2 tables."""
    path = tmp_path / "personal.db"
    _create_v1_archive(path)

    archive = PersonalMemoryArchive(path)

    assert archive.schema_version() == 5
    assert archive.get_exchange("old-exchange").user_text == "old user text"
    assert archive.get_candidate("old-candidate").source.value == "legacy_import"
    assert archive.get_candidate("old-candidate").evidence_excerpt == ""
    recovered = archive.recover_pending_candidate_jobs()
    assert [job.subject_id for job in recovered] == ["old-candidate"]
    assert {
        "archive_metadata",
        "evidence_items",
        "personal_claims",
        "personal_schemas",
        "schema_versions",
        "schema_evidence_links",
        "schema_conflicts",
        "insight_candidates",
        "external_knowledge_items",
        "memory_jobs",
        "memory_decisions",
        "memory_feedback",
        "pending_user_questions",
    } <= archive.table_names()


def test_v2_pending_user_direct_candidate_is_quarantined_on_v3_upgrade(tmp_path):
    path = tmp_path / "personal.db"
    archive = PersonalMemoryArchive(path)
    archive.record_exchange(
        exchange_id="pending-v2",
        user_text="possibly migrated text",
        assistant_text="",
        source="test",
    )
    assert archive.claim_candidate_job("pending-v2") is not None
    candidate = archive.complete_candidate_job(
        "pending-v2",
        [CandidateDraft("fact", "possibly migrated text", 0.5, 0.5)],
        engine_id="old-v2",
        extractor_version="old-v2",
    )[0]
    archive.set_metadata("schema_version", "2")

    reopened = PersonalMemoryArchive(path)

    assert reopened.get_candidate(candidate.id).source.value == "legacy_import"
    assert reopened.get_candidate(candidate.id).status.value == "quarantined"
    assert reopened.recover_pending_candidate_jobs() == []
    assert reopened.pending_job_ids(limit=10) == []
    assert reopened.decision_count(subject_id=candidate.id) == 1
    assert reopened.schema_version() == 5


def test_existing_v4_candidate_archive_adds_blank_evidence_excerpt_idempotently(
    tmp_path,
):
    """A v4 archive must retain its candidate when v5 adds exact evidence."""
    path = tmp_path / "personal.db"
    _create_v1_archive(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE archive_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO archive_metadata(key, value) VALUES ('schema_version', '4')"
        )
        connection.executescript(
            """
            ALTER TABLE memory_candidates
              ADD COLUMN source TEXT NOT NULL DEFAULT 'legacy_import';
            ALTER TABLE memory_candidates
              ADD COLUMN temporal_scope TEXT NOT NULL DEFAULT 'unspecified';
            ALTER TABLE memory_candidates
              ADD COLUMN subject TEXT NOT NULL DEFAULT 'user';
            ALTER TABLE memory_candidates
              ADD COLUMN target_claim_id TEXT NOT NULL DEFAULT '';
            """
        )

    archive = PersonalMemoryArchive(path)

    candidate = archive.get_candidate("old-candidate")
    assert candidate is not None
    assert archive.schema_version() == 5
    assert candidate.content == "old user text"
    assert candidate.evidence_excerpt == ""

    reopened = PersonalMemoryArchive(path)
    assert reopened.schema_version() == 5
    assert reopened.get_candidate("old-candidate").evidence_excerpt == ""


def test_job_idempotency_key_prevents_duplicate_work(tmp_path):
    """Repeated event delivery must produce one durable unit of work."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")

    first = archive.enqueue_job(
        job_type="extract_candidates",
        subject_id="exchange-1",
        idempotency_key="extract_candidates:exchange-1",
        priority=20,
    )
    replay = archive.enqueue_job(
        job_type="extract_candidates",
        subject_id="exchange-1",
        idempotency_key="extract_candidates:exchange-1",
        priority=20,
    )

    assert replay.id == first.id
    assert archive.pending_job_ids(limit=10) == [first.id]


def test_interrupted_memory_job_is_claimable_after_restart(tmp_path):
    """A stopped process must not strand a one-shot job in processing state."""
    path = tmp_path / "personal.db"
    archive = PersonalMemoryArchive(path)
    job = archive.enqueue_job(
        job_type="evaluate_candidate",
        subject_id="candidate-1",
        idempotency_key="evaluate_candidate:candidate-1",
        priority=30,
    )
    claimed = archive.claim_next_job()
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.state == "processing"

    reopened = PersonalMemoryArchive(path)
    assert reopened.release_in_progress_jobs() == 1
    reclaimed = reopened.claim_next_job()

    assert reclaimed is not None
    assert reclaimed.id == job.id
    assert reclaimed.attempts == 2


def test_completed_job_cannot_be_claimed_again(tmp_path):
    """Completion must make a replayed worker wake-up harmless."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.enqueue_job(
        job_type="check_consolidation",
        subject_id="claim-1",
        idempotency_key="check_consolidation:claim-1:v1",
    )
    claimed = archive.claim_next_job()
    assert claimed is not None

    archive.complete_job(claimed.id)

    assert archive.claim_next_job() is None
