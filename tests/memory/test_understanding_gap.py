"""Tests for user-first resolution of personal understanding gaps."""

from __future__ import annotations

import time

import pytest

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import CandidateDraft
from openjarvis.memory.understanding_gap import (
    GapKind,
    QuestionCoordinator,
    UnderstandingGap,
    UnderstandingGapGate,
)


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (UnderstandingGap(GapKind.INTENT_AMBIGUITY), "ask_user"),
        (UnderstandingGap(GapKind.INSUFFICIENT_EVIDENCE), "hold_local"),
        (
            UnderstandingGap(
                GapKind.GENERAL_KNOWLEDGE,
                user_requested_analysis=True,
            ),
            "offer_external",
        ),
        (UnderstandingGap(GapKind.GENERAL_KNOWLEDGE), "hold_local"),
    ],
)
def test_gap_routes_user_intent_before_external_search(gap, expected):
    """Changing route order could send a personal ambiguity outside the device."""
    assert UnderstandingGapGate().route(gap).action == expected


def test_sensitive_gap_requires_consent_even_when_external_mode_is_allowed():
    """A global setting must not waive consent for sensitive interpretation."""
    result = UnderstandingGapGate(external_mode="allowed").route(
        UnderstandingGap(
            GapKind.GENERAL_KNOWLEDGE,
            user_requested_analysis=True,
            sensitive=True,
        )
    )

    assert result.action == "ask_external_consent"


def test_question_requires_a_concrete_decision_effect(tmp_path):
    """The assistant must not offload every vague thought as a user question."""
    coordinator = QuestionCoordinator(
        PersonalMemoryArchive(tmp_path / "personal.db")
    )

    with pytest.raises(ValueError, match="decision effect"):
        coordinator.create(
            subject_id="conflict-1",
            question="What do you think?",
            decision_effect="",
        )


def test_i_do_not_know_keeps_gap_unresolved_and_starts_cooldown(tmp_path):
    """An uncertain answer must not be converted into a false personal fact."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    coordinator = QuestionCoordinator(archive, cooldown_seconds=3600)
    question = coordinator.create(
        subject_id="conflict-1",
        question="Is this fatigue or lack of interest?",
        decision_effect="choose whether to refine the fatigue schema",
    )

    answered = coordinator.answer(question.id, "I do not know")

    assert answered.state == "unresolved"
    assert answered.answer_evidence_id == ""
    assert answered.cooldown_until > time.time()
    assert coordinator.next_pending() is None


def test_confirming_answer_links_user_evidence_and_resumes_subject(tmp_path):
    """A later user answer must remain traceable before validation resumes."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    coordinator = QuestionCoordinator(archive)
    question = coordinator.create(
        subject_id="insight-1",
        question="Does this pattern fit your experience?",
        decision_effect="confirm or reject the insight",
    )
    archive.record_exchange(
        exchange_id="answer-exchange",
        user_text="Yes, when I am rested.",
        assistant_text="Understood.",
        source="test",
    )
    assert archive.claim_candidate_job("answer-exchange") is not None
    candidate = archive.complete_candidate_job(
        "answer-exchange",
        [
            CandidateDraft(
                "fact",
                "Yes, when I am rested.",
                0.8,
                1.0,
                evidence_excerpt="Yes, when I am rested.",
            )
        ],
        engine_id="ollama",
        extractor_version="v2",
    )[0]
    assert MemoryEvaluator(archive).evaluate(candidate.id).applied is True
    evidence_id = archive.get_active_claims()[0].evidence_ids[0]

    answered = coordinator.answer(
        question.id,
        "Yes, when I am rested.",
        answer_evidence_id=evidence_id,
    )

    assert answered.state == "answered"
    assert answered.answer_evidence_id == evidence_id
    assert answered.subject_id == "insight-1"
