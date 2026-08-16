"""Long-timeline acceptance checks for developmental personal memory."""

from __future__ import annotations

import json
from pathlib import Path

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.context_composer import ContextComposer
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import CandidateDraft, CandidateKind

_FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "personal_memory"
    / "dialogue_timeline.json"
)


def _apply(archive: PersonalMemoryArchive, item: dict[str, str]) -> None:
    archive.record_exchange(
        exchange_id=item["id"],
        user_text=item["text"],
        assistant_text="Acknowledged.",
        source="acceptance",
        session_id=f"session:{item['id']}",
    )
    assert archive.claim_candidate_job(item["id"]) is not None
    candidate = archive.complete_candidate_job(
        item["id"],
        [
            CandidateDraft(
                CandidateKind(item["kind"]),
                item["text"],
                1.0,
                1.0,
                temporal_scope=item["scope"],
                subject=item["subject"],
            )
        ],
        engine_id="ollama",
        extractor_version="acceptance",
    )[0]
    assert MemoryEvaluator(archive).evaluate(candidate.id).applied


def test_direct_rules_and_current_state_survive_restart(tmp_path):
    path = tmp_path / "personal.db"
    archive = PersonalMemoryArchive(path)
    timeline = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for index, item in enumerate(timeline):
        _apply(archive, item)
        context = ContextComposer(archive).compose("Continue our conversation")
        expected_constraints = min(index + 1, 2)
        assert len(context.constraints) == expected_constraints

    reopened = PersonalMemoryArchive(path)
    context = ContextComposer(reopened).compose("I still feel tired this week.")

    assert len(context.constraints) == 2
    assert context.current_states == ("I feel tired lately.",)
    assert "faster three kilometer" not in context.render()
    assert all("old coaching" not in item for item in context.suggestions)


def test_assistant_text_never_becomes_personal_evidence(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="assistant-only",
        user_text="Hello.",
        assistant_text="You are definitely a runner.",
        source="acceptance",
    )
    assert archive.claim_candidate_job("assistant-only") is not None
    candidate = archive.complete_candidate_job(
        "assistant-only",
        [CandidateDraft(CandidateKind.FACT, "You are definitely a runner.", 1, 1)],
        engine_id="ollama",
        extractor_version="acceptance",
    )[0]

    result = MemoryEvaluator(archive).evaluate(candidate.id)

    assert result.applied is False
    assert archive.evidence_count(candidate_id=candidate.id) == 0


def test_conflicting_old_direct_rule_is_absent_immediately_after_restart(tmp_path):
    path = tmp_path / "personal.db"
    archive = PersonalMemoryArchive(path)
    _apply(
        archive,
        {
            "id": "old-rule",
            "text": "Proactively mention topic alpha.",
            "kind": "role_preference",
            "scope": "until_changed",
            "subject": "topic:alpha",
        },
    )
    old_claim = archive.get_active_claims()[0]
    archive.record_exchange(
        exchange_id="correction",
        user_text="Do not mention topic alpha unless I ask.",
        assistant_text="",
        source="acceptance",
    )
    assert archive.claim_candidate_job("correction") is not None
    candidate = archive.complete_candidate_job(
        "correction",
        [
            CandidateDraft(
                CandidateKind.CORRECTION,
                "Do not mention topic alpha unless I ask.",
                1.0,
                1.0,
                temporal_scope="until_changed",
                subject="topic:alpha",
                target_claim_id=old_claim.id,
            )
        ],
        engine_id="ollama",
        extractor_version="acceptance",
    )[0]
    assert MemoryEvaluator(archive).evaluate(candidate.id).applied

    reopened = PersonalMemoryArchive(path)
    context = ContextComposer(reopened).compose("Continue.")

    assert context.constraints == ("Do not mention topic alpha unless I ask.",)
    assert "Proactively mention topic alpha." not in context.render()
