"""Restart acceptance coverage for model-led canonical personal memory."""

from __future__ import annotations

import time
from pathlib import Path

from openjarvis.core.config import (
    JarvisConfig,
    MemoryFilesConfig,
    PersonalMemoryConfig,
    effective_chat_memory_files,
)
from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.context_composer import (
    ContextComposer,
    compose_configured_personal_context,
)
from openjarvis.memory.personal_models import CandidateDraft, CandidateKind
from openjarvis.memory.personal_service import PersonalMemoryService
from openjarvis.memory.profile_migration import (
    LegacyProfileMigrator,
    activate_canonical_profile,
)


class _AcceptanceExtractor:
    """A deterministic stand-in for already-validated model proposals."""

    engine_id = "ollama"
    last_error_code = ""

    def __init__(self, proposals: dict[str, tuple[CandidateDraft, ...]]) -> None:
        self._proposals = proposals
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def extract(self, exchange, *, recent_exchanges=()):
        self.calls.append(
            (
                exchange.user_text,
                tuple(item.user_text for item in recent_exchanges),
            )
        )
        return list(self._proposals.get(exchange.user_text, ()))


def _start_runtime(
    archive_path: Path,
    proposals: dict[str, tuple[CandidateDraft, ...]],
) -> tuple[PersonalMemoryService, _AcceptanceExtractor]:
    extractor = _AcceptanceExtractor(proposals)
    service = PersonalMemoryService(
        PersonalMemoryArchive(archive_path),
        extractor,
        idle_before_reflection_seconds=0,
    )
    service.start()
    return service, extractor


def _complete_turn(
    service: PersonalMemoryService,
    user_text: str,
    assistant_text: str,
) -> str:
    exchange = service.archive_exchange(
        user_text=user_text,
        assistant_text=assistant_text,
        source="test.acceptance",
        session_id="restart-acceptance",
    )
    assert service.enqueue_exchange(exchange.id) is True
    return exchange.id


def _wait_for(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "personal-memory background work did not finish"


def _stage_and_activate(
    archive: PersonalMemoryArchive,
    tmp_path: Path,
) -> tuple[Path, Path, tuple[Path, ...]]:
    user_path = tmp_path / "USER.md"
    memory_path = tmp_path / "MEMORY.md"
    user_path.write_text("- The old assistant name is 조visor.\n", encoding="utf-8")
    memory_path.write_text("- Legacy dynamic note.\n", encoding="utf-8")
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    return user_path, memory_path, staged.backup_paths


def test_name_correction_survives_restart_and_canonical_activation(tmp_path):
    """Shadow-mode activation must not drop the canonical name correction."""
    archive_path = tmp_path / "personal.db"
    old_turn = "너의 이름은 조visor야."
    correction_turn = "공박사라고."
    proposals = {
        old_turn: (
            CandidateDraft(
                CandidateKind.ROLE_PREFERENCE,
                "The assistant name is 조visor.",
                1.0,
                1.0,
                temporal_scope="until_changed",
                subject="assistant.name",
                evidence_excerpt=old_turn,
            ),
        ),
        correction_turn: (
            CandidateDraft(
                CandidateKind.CORRECTION,
                "The assistant name is 공박사.",
                1.0,
                1.0,
                temporal_scope="until_changed",
                subject="assistant.name",
                evidence_excerpt=correction_turn,
            ),
        ),
    }
    first, extractor = _start_runtime(archive_path, proposals)
    try:
        _complete_turn(first, old_turn, "알겠어.")
        _wait_for(lambda: len(first.archive.get_active_claims()) == 1)
        old_claim = first.archive.get_active_claims()[0]
        _complete_turn(first, correction_turn, "네, 공박사입니다.")
        _wait_for(
            lambda: any(
                claim.subject_scope == "assistant.name" and "공박사" in claim.content
                for claim in first.archive.get_active_claims()
            )
        )
    finally:
        first.stop()

    restarted = PersonalMemoryArchive(archive_path)
    _stage_and_activate(restarted, tmp_path)
    config = JarvisConfig()
    config.personal_memory = PersonalMemoryConfig(
        enabled=True,
        mode="shadow",
        archive_path=str(archive_path),
    )

    context = compose_configured_personal_context(
        config,
        "너 이름이 뭐야?",
        engine_key="ollama",
        archive=restarted,
    )

    assert context is not None
    assert "공박사" in context.render()
    assert "조visor" not in context.render()
    assert restarted.get_claim(old_claim.id).state.value == "superseded"
    assert (
        any(
            correction_turn in recent
            for user_text, recent in extractor.calls
            if user_text == correction_turn
        )
        is False
    )
    assert any(
        old_turn in recent
        for user_text, recent in extractor.calls
        if user_text == correction_turn
    )


def test_direct_bans_survive_restart_without_static_profile_files(tmp_path):
    """Dropping canonical constraints on restart would revive unwanted topics."""
    archive_path = tmp_path / "personal.db"
    user_text = "임용 시험, BGM, 뽀모도로 이야기를 먼저 꺼내지 마."
    proposals = {
        user_text: tuple(
            CandidateDraft(
                CandidateKind.CONSTRAINT,
                f"Do not proactively mention {topic}.",
                1.0,
                1.0,
                temporal_scope="until_changed",
                subject=f"assistant_behavior.{subject}",
                evidence_excerpt=user_text,
            )
            for topic, subject in (
                ("the teaching appointment exam", "exam"),
                ("BGM", "bgm"),
                ("Pomodoro", "pomodoro"),
            )
        )
    }
    first, _ = _start_runtime(archive_path, proposals)
    try:
        _complete_turn(first, user_text, "알겠어.")
        _wait_for(lambda: len(first.archive.get_active_claims()) == 3)
    finally:
        first.stop()

    rendered = (
        ContextComposer(PersonalMemoryArchive(archive_path)).compose("안녕?").render()
    )

    assert "teaching appointment exam" in rendered
    assert "BGM" in rendered
    assert "Pomodoro" in rendered


def test_ambiguous_psychology_remains_evidence_but_is_not_promoted(tmp_path):
    """A broad hypothesis must not become an active personal claim."""
    archive_path = tmp_path / "personal.db"
    user_text = "요즘 내가 왜 그러는지는 나도 잘 모르겠어."
    proposals = {
        user_text: (
            CandidateDraft(
                CandidateKind.HYPOTHESIS,
                "The user may have an unknown psychological motive.",
                0.4,
                0.3,
                subject="mental_health",
                evidence_excerpt=user_text,
            ),
        )
    }
    runtime, _ = _start_runtime(archive_path, proposals)
    try:
        _complete_turn(runtime, user_text, "그럴 수 있어.")
        _wait_for(
            lambda: (
                bool(runtime.archive.find_candidates())
                and runtime.archive.find_candidates()[0].status.value == "accepted"
            )
        )
        assert all(
            claim.subject_scope
            not in {"identity", "mental_health", "relationship_motive"}
            for claim in runtime.archive.get_active_claims()
        )
    finally:
        runtime.stop()


def test_pending_natural_dialogue_is_context_not_a_direct_rule(tmp_path):
    """Restart continuity must keep raw dialogue out of the system-rule block."""
    archive_path = tmp_path / "personal.db"
    archive = PersonalMemoryArchive(archive_path)
    archive.record_exchange(
        exchange_id="pending-natural-dialogue",
        user_text="오늘은 그냥 이런저런 이야기를 하고 싶어.",
        assistant_text="좋아, 편하게 이야기하자.",
        source="cli.chat",
    )

    context = ContextComposer(PersonalMemoryArchive(archive_path)).compose("계속하자.")

    assert context.constraints == ()
    assert context.recent_pending_user_messages == (
        "오늘은 그냥 이런저런 이야기를 하고 싶어.",
    )
    assert "이런저런" not in context.render()
    assert "편하게 이야기하자" not in context.render()


def test_canonical_activation_keeps_soul_and_disables_static_dynamic_files(tmp_path):
    """Activation must leave identity style while removing competing facts."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path, _ = _stage_and_activate(archive, tmp_path)
    soul_path = tmp_path / "SOUL.md"
    soul_path.write_text("A calm assistant persona.\n", encoding="utf-8")
    original = MemoryFilesConfig(
        soul_path=str(soul_path),
        user_path=str(user_path),
        memory_path=str(memory_path),
    )

    effective = effective_chat_memory_files(JarvisConfig(), original, archive)

    assert effective.soul_path == str(soul_path)
    assert effective.user_path == ""
    assert effective.memory_path == ""
