"""End-to-end tests for auditable candidate evaluation."""

from __future__ import annotations

import sqlite3

import pytest

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import CandidateDraft, CandidateKind


class _NeverCalledClassifier:
    def classify(self, candidate, schemas):
        raise AssertionError("direct rules must bypass relation classification")


def _candidate(
    archive: PersonalMemoryArchive,
    *,
    exchange_id: str,
    user_text: str,
    assistant_text: str,
    draft: CandidateDraft,
):
    archive.record_exchange(
        exchange_id=exchange_id,
        user_text=user_text,
        assistant_text=assistant_text,
        source="test",
        session_id="session-1",
    )
    assert archive.claim_candidate_job(exchange_id) is not None
    return archive.complete_candidate_job(
        exchange_id,
        [draft],
        engine_id="ollama",
        extractor_version="v2",
    )[0]


def test_evaluator_applies_direct_rule_and_audits_once(tmp_path):
    """A replay must not create duplicate evidence, claims, or decisions."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    candidate = _candidate(
        archive,
        exchange_id="e1",
        user_text="Do not mention timers unless I ask.",
        assistant_text="Understood.",
        draft=CandidateDraft(
            CandidateKind.CONSTRAINT,
            "Do not mention timers unless I ask.",
            1.0,
            1.0,
            temporal_scope="until_changed",
            subject="topic:timers",
        ),
    )
    evaluator = MemoryEvaluator(archive, relation_classifier=_NeverCalledClassifier())

    first = evaluator.evaluate(candidate.id)
    replay = evaluator.evaluate(candidate.id)

    claims = archive.get_active_claims(kind=CandidateKind.CONSTRAINT)
    assert first.applied is True
    assert replay.applied is False
    assert replay.reason_code == "candidate_already_evaluated"
    assert len(claims) == 1
    assert claims[0].content == "Do not mention timers unless I ask."
    assert archive.decision_count(subject_id=candidate.id) == 1
    assert archive.evidence_count(candidate_id=candidate.id) == 1


def test_evaluator_rejects_assistant_only_claim(tmp_path):
    """An extractor mistake must not turn assistant suggestions into user evidence."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    candidate = _candidate(
        archive,
        exchange_id="e2",
        user_text="I feel tired.",
        assistant_text="You enjoy ten-minute focus sessions.",
        draft=CandidateDraft(
            CandidateKind.FACT,
            "The user enjoys ten-minute focus sessions.",
            0.8,
            0.9,
        ),
    )

    result = MemoryEvaluator(archive).evaluate(candidate.id)

    assert result.applied is False
    assert result.reason_code == "unsupported_by_user_evidence"
    assert archive.get_active_claims() == []
    assert archive.evidence_count(candidate_id=candidate.id) == 0
    assert archive.get_candidate(candidate.id).status == "rejected"


def test_new_direct_rule_supersedes_explicit_target_atomically(tmp_path):
    """A correction and its supersession audit must become visible together."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    first = _candidate(
        archive,
        exchange_id="e3",
        user_text="Offer timers when I study.",
        assistant_text="Okay.",
        draft=CandidateDraft(
            CandidateKind.PREFERENCE,
            "Offer timers when I study.",
            0.8,
            1.0,
            subject="topic:timers",
        ),
    )
    evaluator = MemoryEvaluator(archive)
    assert evaluator.evaluate(first.id).applied is True
    old_claim = archive.get_active_claims()[0]

    correction = _candidate(
        archive,
        exchange_id="e4",
        user_text="Correction: do not offer timers unless I ask.",
        assistant_text="Understood.",
        draft=CandidateDraft(
            CandidateKind.CORRECTION,
            "Do not offer timers unless I ask.",
            1.0,
            1.0,
            subject="topic:timers",
            target_claim_id=old_claim.id,
        ),
    )

    result = evaluator.evaluate(correction.id)

    assert result.applied is True
    assert result.superseded_claim_ids == (old_claim.id,)
    assert archive.get_claim(old_claim.id).state == "superseded"
    assert [claim.kind for claim in archive.get_active_claims()] == [
        CandidateKind.CORRECTION
    ]


def test_audit_insert_failure_rolls_back_evidence_and_claim(tmp_path):
    """A partial transaction must never expose a claim without its audit decision."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    candidate = _candidate(
        archive,
        exchange_id="e5",
        user_text="Keep answers short.",
        assistant_text="Okay.",
        draft=CandidateDraft(
            CandidateKind.PREFERENCE,
            "Keep answers short.",
            0.8,
            1.0,
            subject="response_style",
        ),
    )
    with sqlite3.connect(archive.path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_decision_insert
            BEFORE INSERT ON memory_decisions
            BEGIN
              SELECT RAISE(ABORT, 'audit unavailable');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="audit unavailable"):
        MemoryEvaluator(archive).evaluate(candidate.id)

    assert archive.get_candidate(candidate.id).status == "pending"
    assert archive.get_active_claims() == []
    assert archive.evidence_count(candidate_id=candidate.id) == 0
