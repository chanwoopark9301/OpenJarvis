"""Golden regressions for the conversation failure that motivated the model."""

from __future__ import annotations

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.context_composer import ContextComposer
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import CandidateDraft, CandidateKind


def _accept(archive, exchange_id, user_text, draft):
    archive.record_exchange(
        exchange_id=exchange_id,
        user_text=user_text,
        assistant_text="Understood.",
        source="test",
    )
    assert archive.claim_candidate_job(exchange_id) is not None
    candidate = archive.complete_candidate_job(
        exchange_id,
        [draft],
        engine_id="ollama",
        extractor_version="v2",
    )[0]
    assert MemoryEvaluator(archive).evaluate(candidate.id).applied is True


def test_direct_topic_bans_survive_related_study_conversation(tmp_path):
    """Mentioning study later must not resurrect prohibited coaching suggestions."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    _accept(
        archive,
        "old",
        "I used to ask for study plans.",
        CandidateDraft(
            CandidateKind.PREFERENCE,
            "I used to ask for study plans.",
            0.7,
            1.0,
            subject="study_coaching",
        ),
    )
    for index, (topic, content) in enumerate(
        [
            ("exam", "Do not proactively mention the counselor appointment exam."),
            ("music", "Do not proactively suggest background music."),
            ("timer", "Do not proactively suggest focus timers."),
        ]
    ):
        _accept(
            archive,
            f"ban-{index}",
            content,
            CandidateDraft(
                CandidateKind.CONSTRAINT,
                content,
                1.0,
                1.0,
                temporal_scope="until_changed",
                subject=f"topic:{topic}",
            ),
        )

    context = ContextComposer(archive).compose("I did not study this week.")

    assert len(context.constraints) == 3
    assert all("Do not proactively" in item for item in context.constraints)
    assert all("study plans" not in item for item in context.suggestions)


def test_capability_boundary_blocks_claiming_unavailable_actions(tmp_path):
    """The assistant must not claim it can perform an unavailable action."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    content = "Do not claim you can play music or start focus timers."
    _accept(
        archive,
        "capability",
        content,
        CandidateDraft(
            CandidateKind.CAPABILITY_BOUNDARY,
            content,
            1.0,
            1.0,
            temporal_scope="until_changed",
            subject="assistant_capabilities",
        ),
    )

    context = ContextComposer(archive).compose("Keep me company for a bit.")

    assert context.constraints == (content,)
