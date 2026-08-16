"""Tests for bounded local-model relation proposals."""

from __future__ import annotations

from openjarvis.memory.adaptation import RelationProposal
from openjarvis.memory.personal_models import (
    CandidateDraft,
    CandidateKind,
    ClaimState,
    EvidenceMass,
    PersonalSchema,
    SchemaMaturity,
)
from openjarvis.memory.relation_classifier import RelationClassifier


class _FakeEngine:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"content": self.content}


def _candidate(subject: str = "exercise_and_mood") -> CandidateDraft:
    return CandidateDraft(
        CandidateKind.EPISODE,
        "Running did not improve my mood today.",
        0.6,
        0.8,
        temporal_scope="current",
        subject=subject,
    )


def _schema(index: int = 1, subject: str = "exercise_and_mood") -> PersonalSchema:
    return PersonalSchema(
        id=f"schema-{index}",
        content="Running usually improves the user's mood.",
        maturity=SchemaMaturity.EMERGING,
        state=ClaimState.ACTIVE,
        current_version_id=f"version-{index}",
        evidence_mass=EvidenceMass(),
        created_at=1.0,
        updated_at=1.0,
        subject_scope=subject,
    )


def test_classifier_accepts_only_the_closed_relation_contract():
    """A valid relation proposal must preserve all bounded fields."""
    engine = _FakeEngine(
        '{"relation":"conditional_exception","scope_fit":0.7,'
        '"reason":"Today is limited by fatigue.","question_would_help":true}'
    )

    result = RelationClassifier(engine, "qwen3.5:9b").classify(
        _candidate(), [_schema()]
    )

    assert result.relation is RelationProposal.CONDITIONAL_EXCEPTION
    assert result.scope_fit == 0.7
    assert result.question_would_help is True


def test_malformed_or_action_bearing_output_becomes_ambiguous():
    """Model output must not smuggle a persistence action through classification."""
    engine = _FakeEngine(
        '{"relation":"supports","scope_fit":1,"reason":"x",'
        '"question_would_help":false,"action":"delete_schema"}'
    )
    classifier = RelationClassifier(engine, "qwen3.5:9b")

    result = classifier.classify(_candidate(), [_schema()])

    assert result.relation is RelationProposal.AMBIGUOUS
    assert classifier.last_error_code == "invalid_output"


def test_clear_subject_mismatch_is_unrelated_without_a_model_call():
    """A model support label must not override non-overlapping subject scope."""
    engine = _FakeEngine("not used")

    result = RelationClassifier(engine, "qwen3.5:9b").classify(
        _candidate("sleep_and_concentration"), [_schema()]
    )

    assert result.relation is RelationProposal.UNRELATED
    assert result.scope_fit == 0.0
    assert engine.calls == []


def test_classifier_passes_no_more_than_eight_schema_summaries():
    """Removing the cap would make model memory proportional to archive size."""
    engine = _FakeEngine(
        '{"relation":"ambiguous","scope_fit":0.4,"reason":"unclear",'
        '"question_would_help":true}'
    )
    schemas = [_schema(index) for index in range(12)]

    RelationClassifier(engine, "qwen3.5:9b").classify(_candidate(), schemas)

    messages, _ = engine.calls[0]
    assert messages[1].content.count('"schema_id"') == 8


def test_engine_failure_returns_ambiguous_without_raising():
    """A relation model failure must hold evidence instead of losing it."""

    class _FailingEngine:
        def generate(self, messages, **kwargs):
            raise RuntimeError("offline")

    classifier = RelationClassifier(_FailingEngine(), "qwen3.5:9b")

    result = classifier.classify(_candidate(), [_schema()])

    assert result.relation is RelationProposal.AMBIGUOUS
    assert classifier.last_error_code == "engine_error"
