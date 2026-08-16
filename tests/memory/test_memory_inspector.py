"""Tests for transparent, immediately effective personal-memory controls."""

from __future__ import annotations

import pytest

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.context_composer import ContextComposer
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.inspector import PersonalMemoryInspector
from openjarvis.memory.personal_models import CandidateDraft, CandidateKind


def _accepted_claim(archive: PersonalMemoryArchive, content: str):
    archive.record_exchange(
        exchange_id="original",
        user_text=content,
        assistant_text="Acknowledged.",
        source="test",
    )
    assert archive.claim_candidate_job("original") is not None
    candidate = archive.complete_candidate_job(
        "original",
        [
            CandidateDraft(
                CandidateKind.CONSTRAINT,
                content,
                1.0,
                1.0,
                temporal_scope="until_changed",
                subject="topic:timer",
            )
        ],
        engine_id="ollama",
        extractor_version="test",
    )[0]
    assert MemoryEvaluator(archive).evaluate(candidate.id).applied
    return archive.get_active_claims()[0]


def test_list_and_explain_claim_with_evidence(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    claim = _accepted_claim(archive, "Do not proactively mention timers.")
    inspector = PersonalMemoryInspector(archive)

    listed = inspector.list_subjects()
    detail = inspector.explain(claim.id)

    assert listed[0].id == claim.id
    assert listed[0].state == "active"
    assert detail is not None
    assert detail.supporting_evidence == ("Do not proactively mention timers.",)
    assert detail.last_use_reason == ""


def test_suppress_and_restore_affect_next_context(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    claim = _accepted_claim(archive, "Do not proactively mention timers.")
    inspector = PersonalMemoryInspector(archive)

    assert inspector.suppress(claim.id, "user requested")
    assert ContextComposer(archive).compose("hello").constraints == ()
    assert inspector.restore(claim.id)
    assert ContextComposer(archive).compose("hello").constraints == (
        "Do not proactively mention timers.",
    )


def test_correction_supersedes_instead_of_rewriting(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    claim = _accepted_claim(archive, "Do not proactively mention timers.")
    inspector = PersonalMemoryInspector(archive)

    replacement = "Only mention timers when I ask about them."
    corrected = inspector.correct(claim.id, replacement, replacement)

    assert corrected is not None
    assert archive.get_claim(claim.id).state.value == "superseded"
    assert archive.get_claim(claim.id).content == "Do not proactively mention timers."
    assert ContextComposer(archive).compose("hello").constraints == (replacement,)


def test_delete_subject_preserves_raw_evidence_by_default(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    claim = _accepted_claim(archive, "Do not proactively mention timers.")
    inspector = PersonalMemoryInspector(archive)

    counts = inspector.delete_subject(claim.id)

    assert counts["personal_claims"] == 1
    assert archive.get_claim(claim.id) is None
    assert archive.get_exchange("original") is not None


def test_delete_all_requires_exact_confirmation_token(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    _accepted_claim(archive, "Do not proactively mention timers.")
    inspector = PersonalMemoryInspector(archive)

    preview = inspector.deletion_preview()
    assert preview["personal_claims"] == 1
    with pytest.raises(ValueError, match="confirmation token"):
        inspector.delete_all_personal_memory("wrong")

    removed = inspector.delete_all_personal_memory("DELETE ALL PERSONAL MEMORY")

    assert removed["personal_claims"] == 1
    assert archive.get_active_claims() == []
