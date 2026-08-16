"""Tests for conservative, idempotent legacy fact import."""

from __future__ import annotations

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.legacy_import import (
    LegacyFactImporter,
    evaluate_rollout_readiness,
)
from openjarvis.memory.personal_models import EvidenceSource


def test_legacy_fact_import_is_low_trust_and_idempotent(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    facts = tmp_path / "memory_facts.jsonl"
    facts.write_text(
        '{"text":"User likes timers","source":"old"}\n',
        encoding="utf-8",
    )

    importer = LegacyFactImporter(archive)
    first = importer.run(facts)
    second = importer.run(facts)

    candidates = archive.find_candidates(source=EvidenceSource.LEGACY_IMPORT)
    assert first.imported == 1
    assert second.imported == 0
    assert len(candidates) == 1
    assert candidates[0].status.value == "pending"
    assert archive.get_active_claims() == []
    exchange = archive.get_exchange(candidates[0].exchange_id)
    assert exchange is not None
    assert exchange.user_text == ""
    assert exchange.assistant_text == ""


def test_import_accepts_old_content_key_and_skips_invalid_lines(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    facts = tmp_path / "memory_facts.jsonl"
    facts.write_text(
        "not-json\n"
        '{"content":"Prefers concise replies"}\n'
        '{"text":""}\n',
        encoding="utf-8",
    )

    result = LegacyFactImporter(archive).run(facts)

    assert result.imported == 1
    assert result.skipped == 2
    candidate = archive.find_candidates(source="legacy_import")[0]
    assert candidate.content == "Prefers concise replies"


def test_rollout_requires_backup_and_seven_days_or_manual_override(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    facts = tmp_path / "memory_facts.jsonl"
    facts.write_text('{"text":"legacy"}\n', encoding="utf-8")
    backup = LegacyFactImporter.backup(facts, tmp_path / "legacy.backup")
    archive.set_metadata("shadow_started_at", "1000")

    too_soon = evaluate_rollout_readiness(
        archive,
        backup_path=backup,
        now=1001,
    )
    overridden = evaluate_rollout_readiness(
        archive,
        backup_path=backup,
        manual_override=True,
        now=1001,
    )

    assert too_soon.ready is False
    assert too_soon.failing_gates == ("seven_shadow_days_incomplete",)
    assert overridden.ready is True
