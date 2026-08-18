"""Behavior tests for the local-only personal conversation archive."""

from __future__ import annotations

import sqlite3

import pytest


def _archive_api():
    """Import inside a test so an absent new API is a useful RED failure."""
    try:
        from openjarvis.memory.archive import PersonalMemoryArchive
        from openjarvis.memory.personal_models import CandidateDraft
    except ImportError:
        pytest.fail("personal memory archive API is missing")
    return PersonalMemoryArchive, CandidateDraft


def _seed_three_ordered_exchanges(tmp_path):
    PersonalMemoryArchive, _ = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    for index in range(1, 4):
        archive.record_exchange(
            exchange_id=f"exchange-{index}",
            user_text=f"user {index}",
            assistant_text=f"assistant {index}",
            source="cli.chat",
            session_id="session-1",
        )
    with sqlite3.connect(archive.path) as connection:
        for index in range(1, 4):
            connection.execute(
                "UPDATE conversation_exchanges SET created_at = ? WHERE id = ?",
                (float(index), f"exchange-{index}"),
            )
    return archive


def test_stage_profile_candidates_preserves_legacy_metadata_keyword(tmp_path):
    """The pre-builder public keyword must retain import and replay behavior."""
    PersonalMemoryArchive, CandidateDraft = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    records = [
        (
            "legacy-stage-exchange",
            "legacy-profile:/profile/USER.md:1",
            CandidateDraft("fact", "Legacy staged fact", 0.1, 0.1),
        )
    ]

    first = archive.stage_profile_candidates(
        records,
        metadata={"legacy_stage_marker": "first"},
    )
    replay = archive.stage_profile_candidates(
        records,
        metadata={"legacy_stage_marker": "second"},
    )

    assert first == (1, 0)
    assert replay == (0, 1)
    assert archive.get_metadata("legacy_stage_marker") == "second"
    assert len(archive.find_candidates()) == 1


@pytest.mark.parametrize("provided", ("neither", "both"))
def test_stage_profile_candidates_requires_one_metadata_source(tmp_path, provided):
    """Static metadata and an atomic builder are mutually exclusive contracts."""
    PersonalMemoryArchive, _ = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    kwargs = {}
    if provided == "both":
        kwargs = {
            "metadata": {"legacy_stage_marker": "static"},
            "metadata_builder": lambda _raw: {"legacy_stage_marker": "builder"},
        }

    with pytest.raises(ValueError, match="exactly one metadata source"):
        archive.stage_profile_candidates([], **kwargs)

    assert archive.get_metadata("legacy_stage_marker", "") == ""


def test_recent_exchanges_are_bounded_and_end_before_current(tmp_path):
    """Dialogue context must not include its current or future exchange."""
    archive = _seed_three_ordered_exchanges(tmp_path)

    recent = archive.recent_exchanges("exchange-3", limit=2)

    assert [item.id for item in recent] == ["exchange-1", "exchange-2"]


def test_recent_exchanges_use_id_to_order_same_timestamp_records(tmp_path):
    """Earlier IDs at the current timestamp must remain available as context."""
    PersonalMemoryArchive, _ = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    for index in range(1, 5):
        archive.record_exchange(
            exchange_id=f"exchange-{index}",
            user_text=f"user {index}",
            assistant_text=f"assistant {index}",
            source="cli.chat",
            session_id="session-1",
        )
    with sqlite3.connect(archive.path) as connection:
        connection.execute("UPDATE conversation_exchanges SET created_at = 1")

    recent = archive.recent_exchanges("exchange-3", limit=6)

    assert [item.id for item in recent] == ["exchange-1", "exchange-2"]


def test_replaying_exchange_id_keeps_the_first_archived_conversation(tmp_path):
    """Changing an already-recorded exchange ID must not rewrite its evidence."""
    PersonalMemoryArchive, _ = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")

    first = archive.record_exchange(
        exchange_id="exchange-1",
        user_text="I study Korean.",
        assistant_text="I can help.",
        source="cli.chat",
    )
    replay = archive.record_exchange(
        exchange_id="exchange-1",
        user_text="changed",
        assistant_text="changed",
        source="cli.chat",
    )

    assert replay == first
    stored = archive.get_exchange("exchange-1")
    assert stored is not None
    assert stored.user_text == "I study Korean."


def test_failed_candidate_work_is_available_for_a_later_retry(tmp_path):
    """An extractor failure must not discard its archived conversation."""
    PersonalMemoryArchive, _ = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="exchange-2",
        user_text="I want a study plan.",
        assistant_text="Let's make one.",
        source="server.chat",
    )

    assert archive.claim_candidate_job("exchange-2") is not None
    archive.fail_candidate_job("exchange-2", error_code="invalid_output")

    assert archive.pending_exchange_ids(limit=10) == ["exchange-2"]


def test_failed_candidate_work_can_wait_before_a_retry(tmp_path):
    """Worker backoff must prevent a broken extractor from spinning immediately."""
    PersonalMemoryArchive, _ = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="exchange-delayed",
        user_text="Retry later.",
        assistant_text="Okay.",
        source="test",
    )

    assert archive.claim_candidate_job("exchange-delayed") is not None
    archive.fail_candidate_job(
        "exchange-delayed",
        error_code="invalid_output",
        retry_after=60.0,
    )

    assert archive.pending_exchange_ids(limit=10) == []


def test_completed_candidate_keeps_a_link_to_its_exact_conversation(tmp_path):
    """A provisional candidate must retain its supporting archived exchange."""
    PersonalMemoryArchive, CandidateDraft = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="exchange-3",
        user_text="I prefer short answers.",
        assistant_text="Noted.",
        source="cli.chat",
    )

    assert archive.claim_candidate_job("exchange-3") is not None
    candidates = archive.complete_candidate_job(
        "exchange-3",
        [CandidateDraft("fact", "User prefers short answers.", 0.8, 0.9)],
        engine_id="ollama",
        extractor_version="v1",
    )

    assert candidates[0].exchange_id == "exchange-3"
    assert candidates[0].status == "pending"


def test_completed_candidate_keeps_its_exact_user_evidence_excerpt(tmp_path):
    """Candidate content alone must not replace the user's original wording."""
    PersonalMemoryArchive, CandidateDraft = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="exchange-evidence",
        user_text="공박사라고 불러 주세요.",
        assistant_text="알겠습니다.",
        source="cli.chat",
    )
    assert archive.claim_candidate_job("exchange-evidence") is not None

    candidate = archive.complete_candidate_job(
        "exchange-evidence",
        [
            CandidateDraft(
                "correction",
                "The assistant name is 공박사.",
                1,
                1,
                evidence_excerpt="공박사라고 불러 주세요.",
            )
        ],
        engine_id="ollama",
        extractor_version="v1",
    )[0]

    assert candidate.evidence_excerpt == "공박사라고 불러 주세요."


def test_completed_candidate_preserves_rule_provenance_and_scope(tmp_path):
    """Losing direct-rule metadata would make later priority gates unsafe."""
    PersonalMemoryArchive, CandidateDraft = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="exchange-rule",
        user_text="Do not mention timers unless I ask.",
        assistant_text="Understood.",
        source="cli.chat",
    )
    assert archive.claim_candidate_job("exchange-rule") is not None

    candidates = archive.complete_candidate_job(
        "exchange-rule",
        [
            CandidateDraft(
                "constraint",
                "Do not mention timers unless asked.",
                1.0,
                1.0,
                temporal_scope="until_changed",
                subject="assistant_behavior",
            )
        ],
        engine_id="ollama",
        extractor_version="v2",
    )

    assert candidates[0].kind == "constraint"
    assert candidates[0].source == "user_direct"
    assert candidates[0].temporal_scope == "until_changed"
    assert candidates[0].subject == "assistant_behavior"


def test_completed_zero_candidate_exchange_is_recovered_once_per_version(tmp_path):
    """Removing extractor-version recovery must strand old empty results forever."""
    PersonalMemoryArchive, _ = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="exchange-empty-old",
        user_text="앞으로 농담을 하지 마.",
        assistant_text="알겠습니다.",
        source="cli.chat",
    )
    assert archive.claim_candidate_job("exchange-empty-old") is not None
    archive.complete_candidate_job(
        "exchange-empty-old",
        [],
        engine_id="ollama",
        extractor_version="personal-memory-v1",
    )

    assert archive.recover_stale_candidate_jobs("personal-memory-v3") == 1
    assert archive.pending_exchange_ids(limit=10) == ["exchange-empty-old"]
    assert archive.recover_stale_candidate_jobs("personal-memory-v3") == 0


def test_current_zero_candidate_result_is_not_recovered(tmp_path):
    """A legitimate empty result must not be reevaluated on every startup."""
    PersonalMemoryArchive, _ = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="exchange-current-empty",
        user_text="안녕?",
        assistant_text="안녕하세요.",
        source="cli.chat",
    )
    assert archive.claim_candidate_job("exchange-current-empty") is not None
    archive.complete_candidate_job(
        "exchange-current-empty",
        [],
        engine_id="ollama",
        extractor_version="personal-memory-v3",
    )

    assert archive.recover_stale_candidate_jobs("personal-memory-v3") == 0
    assert archive.pending_exchange_ids(limit=10) == []


def test_zero_candidate_success_is_remembered_for_every_extractor_version(tmp_path):
    """Rolling back versions must not repeat work already completed by that version."""
    PersonalMemoryArchive, _ = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="exchange-version-history",
        user_text="안녕?",
        assistant_text="안녕하세요.",
        source="cli.chat",
    )
    assert archive.claim_candidate_job("exchange-version-history") is not None
    archive.complete_candidate_job(
        "exchange-version-history",
        [],
        engine_id="ollama",
        extractor_version="personal-memory-v1",
    )
    assert archive.recover_stale_candidate_jobs("personal-memory-v3") == 1
    assert archive.claim_candidate_job("exchange-version-history") is not None
    archive.complete_candidate_job(
        "exchange-version-history",
        [],
        engine_id="ollama",
        extractor_version="personal-memory-v3",
    )

    assert archive.recover_stale_candidate_jobs("personal-memory-v1") == 0


def test_recovery_reopens_a_completed_job_for_a_pending_exchange(tmp_path):
    """A stale complete job marker must not strand recoverable conversation work."""
    PersonalMemoryArchive, _ = _archive_api()
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.record_exchange(
        exchange_id="exchange-stranded",
        user_text="앞으로 농담을 하지 마.",
        assistant_text="알겠습니다.",
        source="cli.chat",
    )
    assert archive.claim_candidate_job("exchange-stranded") is not None
    archive.complete_candidate_job(
        "exchange-stranded",
        [],
        engine_id="ollama",
        extractor_version="personal-memory-v1",
    )
    job = archive.enqueue_job(
        job_type="extract_candidates",
        subject_id="exchange-stranded",
        idempotency_key=("extract_candidates:personal-memory-v3:exchange-stranded"),
        priority=100,
    )
    assert archive.claim_job(job.id) is not None
    archive.complete_job(job.id)

    assert archive.recover_stale_candidate_jobs("personal-memory-v3") == 1

    assert job.id in archive.pending_job_ids(limit=10)
