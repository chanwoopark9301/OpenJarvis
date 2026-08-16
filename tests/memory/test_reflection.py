"""Tests for evidence-bounded reflection proposals."""

from __future__ import annotations

from openjarvis.memory.personal_models import AdaptationOperation
from openjarvis.memory.reflection import ReflectionEngine


class _FakeEngine:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"content": self.content}


def _valid_response(**overrides) -> str:
    values = {
        "proposal": "Exercise helps mood when physical fatigue is manageable.",
        "operation": "accommodate_refine",
        "scope": "exercise_and_mood",
        "support_evidence_ids": ["e1", "e2", "e3"],
        "counter_evidence_ids": ["e4"],
        "uncertainties": ["effect of sleep"],
        "requires_user_confirmation": True,
    }
    values.update(overrides)
    import json

    return json.dumps(values)


def test_reflection_accepts_a_grounded_bounded_proposal():
    """A valid proposal must preserve supporting, opposing, and uncertain parts."""
    engine = _FakeEngine(_valid_response())

    proposal = ReflectionEngine(engine, "qwen3.5:9b").reflect(
        evidence={"e1": "a", "e2": "b", "e3": "c", "e4": "counter"},
        conflict_evidence_ids={"e4"},
    )

    assert proposal is not None
    assert proposal.operation is AdaptationOperation.ACCOMMODATE_REFINE
    assert proposal.support_evidence_ids == ("e1", "e2", "e3")
    assert proposal.counter_evidence_ids == ("e4",)


def test_reflection_rejects_invented_evidence_ids():
    """A fluent proposal cannot cite evidence that was not supplied."""
    engine = _FakeEngine(
        _valid_response(support_evidence_ids=["e1", "invented"])
    )
    reflection = ReflectionEngine(engine, "qwen3.5:9b")

    proposal = reflection.reflect(
        evidence={"e1": "a", "e4": "counter"},
        conflict_evidence_ids={"e4"},
    )

    assert proposal is None
    assert reflection.last_error_code == "unknown_evidence_id"


def test_reflection_requires_counter_coverage_when_conflict_exists():
    """Ignoring available conflict would create self-confirming insight."""
    reflection = ReflectionEngine(
        _FakeEngine(_valid_response(counter_evidence_ids=[])),
        "qwen3.5:9b",
    )

    proposal = reflection.reflect(
        evidence={"e1": "a", "e2": "b", "e3": "c", "e4": "counter"},
        conflict_evidence_ids={"e4"},
    )

    assert proposal is None
    assert reflection.last_error_code == "missing_counter_evidence"


def test_reflection_caps_supplied_evidence_at_twelve():
    """Reflection prompt size must remain fixed as the archive grows."""
    engine = _FakeEngine(_valid_response())
    evidence = {f"e{index}": str(index) for index in range(1, 20)}

    ReflectionEngine(engine, "qwen3.5:9b").reflect(
        evidence=evidence,
        conflict_evidence_ids={"e4"},
    )

    messages, _ = engine.calls[0]
    assert messages[1].content.count('"evidence_id"') == 12
