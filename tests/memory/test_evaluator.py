"""End-to-end tests for auditable candidate evaluation."""

from __future__ import annotations

import sqlite3

import pytest

from openjarvis.memory.adaptation import RelationProposal
from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import (
    AdaptationOperation,
    CandidateDraft,
    CandidateKind,
)
from openjarvis.memory.relation_classifier import RelationAssessment


class _NeverCalledClassifier:
    def classify(self, candidate, schemas):
        raise AssertionError("direct rules must bypass relation classification")


class _ContradictionClassifier:
    def classify(self, candidate, schemas):
        assert schemas
        return RelationAssessment(
            RelationProposal.CONTRADICTS,
            1.0,
            "new user evidence conflicts",
            True,
        )


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
            evidence_excerpt="Do not mention timers unless I ask.",
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
    assert archive.evidence_texts(claims[0].evidence_ids) == (
        "Do not mention timers unless I ask.",
    )
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
            evidence_excerpt="You enjoy ten-minute focus sessions.",
        ),
    )

    result = MemoryEvaluator(archive).evaluate(candidate.id)

    assert result.applied is False
    assert result.reason_code == "unsupported_by_user_evidence"
    assert archive.get_active_claims() == []
    assert archive.evidence_count(candidate_id=candidate.id) == 0
    assert archive.get_candidate(candidate.id).status == "rejected"


@pytest.mark.parametrize("evidence_excerpt", ["", "   ", "I like timers."])
def test_evaluator_rejects_empty_or_unlinked_evidence_excerpt(
    tmp_path,
    evidence_excerpt,
):
    """Only an exact excerpt from the linked user message is admissible evidence."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    candidate = _candidate(
        archive,
        exchange_id="negative",
        user_text="I do not like timers.   ",
        assistant_text="",
        draft=CandidateDraft(
            CandidateKind.PREFERENCE,
            "The user dislikes timers.",
            0.8,
            0.9,
            evidence_excerpt=evidence_excerpt,
        ),
    )

    result = MemoryEvaluator(archive).evaluate(candidate.id)

    assert result.applied is False
    assert result.reason_code == "unsupported_by_user_evidence"


def test_evaluator_uses_exact_user_excerpt_not_normalized_claim_text(tmp_path):
    """Evidence must preserve what the user said, not the model's normalized claim."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    candidate = _candidate(
        archive,
        exchange_id="exact-excerpt",
        user_text="아니,  공박사라고.  꼭 기억해.",
        assistant_text="알겠습니다.",
        draft=CandidateDraft(
            CandidateKind.ROLE_PREFERENCE,
            "The assistant name is 공박사.",
            1.0,
            1.0,
            subject="assistant.name",
            evidence_excerpt="  공박사라고.  ",
        ),
    )

    result = MemoryEvaluator(archive).evaluate(candidate.id)
    claim = archive.get_active_claims()[0]

    assert result.applied is True
    assert archive.evidence_texts(claim.evidence_ids) == ("  공박사라고.  ",)


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
            evidence_excerpt="Offer timers when I study.",
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
            evidence_excerpt="do not offer timers unless I ask.",
        ),
    )

    result = evaluator.evaluate(correction.id)

    assert result.applied is True
    assert result.superseded_claim_ids == (old_claim.id,)
    assert archive.get_claim(old_claim.id).state == "superseded"
    assert [claim.kind for claim in archive.get_active_claims()] == [
        CandidateKind.CORRECTION
    ]


def test_correction_supersedes_active_direct_claim_with_same_subject(tmp_path):
    """A typed correction can replace an active direct rule without a target ID."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    first = _candidate(
        archive,
        exchange_id="old-name",
        user_text="Call yourself 조비서.",
        assistant_text="Okay.",
        draft=CandidateDraft(
            CandidateKind.ROLE_PREFERENCE,
            "The assistant name is 조비서.",
            1.0,
            1.0,
            subject="assistant.name",
            evidence_excerpt="Call yourself 조비서.",
        ),
    )
    evaluator = MemoryEvaluator(archive)
    assert evaluator.evaluate(first.id).applied is True
    old_claim = archive.get_active_claims()[0]
    correction = _candidate(
        archive,
        exchange_id="new-name",
        user_text="아니, 공박사라고.",
        assistant_text="알겠습니다.",
        draft=CandidateDraft(
            CandidateKind.CORRECTION,
            "The assistant name is 공박사.",
            1.0,
            1.0,
            subject="assistant.name",
            evidence_excerpt="공박사라고.",
        ),
    )

    result = evaluator.evaluate(correction.id)
    active_claim = archive.get_active_claims()[0]

    assert result.applied is True
    assert result.superseded_claim_ids == (old_claim.id,)
    assert archive.get_claim(old_claim.id).state.value == "superseded"
    assert active_claim.kind is CandidateKind.CORRECTION
    assert active_claim.subject_scope == "assistant.name"
    assert active_claim.supersedes_id == old_claim.id
    assert archive.evidence_texts(active_claim.evidence_ids) == ("공박사라고.",)


def test_audit_failure_rolls_back_selected_supersession_evidence_and_claim(tmp_path):
    """The old active rule must survive if correction audit persistence fails."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    old_candidate = _candidate(
        archive,
        exchange_id="old-response-style",
        user_text="Use detailed answers.",
        assistant_text="Okay.",
        draft=CandidateDraft(
            CandidateKind.ROLE_PREFERENCE,
            "Use detailed answers.",
            1.0,
            1.0,
            subject="response_style",
            evidence_excerpt="Use detailed answers.",
        ),
    )
    evaluator = MemoryEvaluator(archive)
    assert evaluator.evaluate(old_candidate.id).applied is True
    old_claim = archive.get_active_claims()[0]
    correction = _candidate(
        archive,
        exchange_id="correct-response-style",
        user_text="Actually, keep answers short.",
        assistant_text="Okay.",
        draft=CandidateDraft(
            CandidateKind.CORRECTION,
            "Keep answers short.",
            1.0,
            1.0,
            subject="response_style",
            evidence_excerpt="keep answers short.",
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
        evaluator.evaluate(correction.id)

    assert archive.get_candidate(correction.id).status == "pending"
    assert archive.get_claim(old_claim.id).state.value == "active"
    assert [claim.id for claim in archive.get_active_claims()] == [old_claim.id]
    assert len(archive.list_claims()) == 1
    assert archive.evidence_count(candidate_id=correction.id) == 0
    assert archive.decision_count(subject_id=correction.id) == 0


def test_contradiction_challenges_schema_and_leaves_auditable_conflict(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    for index in range(3):
        candidate = _candidate(
            archive,
            exchange_id=f"support-{index}",
            user_text=f"Running improved my mood on day {index}.",
            assistant_text="",
            draft=CandidateDraft(
                CandidateKind.FACT,
                f"Running improved my mood on day {index}.",
                0.8,
                0.9,
                subject="running_and_mood",
                evidence_excerpt=f"Running improved my mood on day {index}.",
            ),
        )
        assert MemoryEvaluator(archive).evaluate(candidate.id).applied
    evidence_ids = tuple(
        evidence_id
        for claim in archive.get_active_claims()
        for evidence_id in claim.evidence_ids
    )
    schema = archive.apply_schema_accommodation(
        content="Running improves my mood.",
        operation=AdaptationOperation.ACCOMMODATE_CREATE,
        support_evidence_ids=evidence_ids,
        subject_scope="running_and_mood",
        user_confirmed=True,
    )
    assert schema is not None
    contradiction = _candidate(
        archive,
        exchange_id="counter",
        user_text="Running made my mood worse today.",
        assistant_text="",
        draft=CandidateDraft(
            CandidateKind.FACT,
            "Running made my mood worse today.",
            1.0,
            1.0,
            subject="running_and_mood",
            evidence_excerpt="Running made my mood worse today.",
        ),
    )

    result = MemoryEvaluator(
        archive,
        relation_classifier=_ContradictionClassifier(),
    ).evaluate(contradiction.id)

    assert result.applied is True
    assert archive.unresolved_conflict_count(schema_id=schema.id) == 1
    assert archive.get_active_schemas() == []
