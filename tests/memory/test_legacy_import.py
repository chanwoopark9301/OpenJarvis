"""Tests for conservative, idempotent legacy fact import."""

from __future__ import annotations

from types import SimpleNamespace

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.legacy_import import (
    LegacyFactImporter,
    activate_rollout,
    evaluate_rollout_readiness,
)
from openjarvis.memory.personal_models import EvidenceSource
from openjarvis.memory.store import legacy_fact_context_enabled

_RELEASE_GATES = (
    "assistant_only_rejection",
    "benchmark_10000",
    "chat_priority_and_concurrency",
    "external_isolation",
    "provider_privacy_interception",
    "replay_idempotency",
    "restart_and_direct_rule",
    "user_controls",
)


def _attest_release_checks(archive: PersonalMemoryArchive) -> None:
    for gate in _RELEASE_GATES:
        archive.record_release_gate_attestation(gate, passed=True)


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
    archive.record_shadow_composition(
        latency_ms=1.0,
        constraint_count=0,
        schema_count=0,
        episode_count=0,
        raw_evidence_count=0,
    )
    archive.set_metadata("shadow_started_at", "1000")
    _attest_release_checks(archive)

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


def test_active_rollout_requires_explicit_successful_activation(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    facts = tmp_path / "memory_facts.jsonl"
    facts.write_text('{"text":"legacy"}\n', encoding="utf-8")
    backup = LegacyFactImporter.backup(facts, tmp_path / "legacy.backup")
    archive.record_shadow_composition(
        latency_ms=1.0,
        constraint_count=0,
        schema_count=0,
        episode_count=0,
        raw_evidence_count=0,
    )
    archive.set_metadata("shadow_started_at", "1000")
    _attest_release_checks(archive)

    result = activate_rollout(archive, backup_path=backup, now=1000 + 604801)

    assert result.ready is True
    assert archive.rollout_is_active() is True


def test_explicit_legacy_context_opt_out_is_immediate(tmp_path):
    archive_path = tmp_path / "personal.db"
    archive = PersonalMemoryArchive(archive_path)
    config = SimpleNamespace(
        personal_memory=SimpleNamespace(
            enabled=True,
            legacy_context_injection=False,
            archive_path=str(archive_path),
        )
    )

    assert legacy_fact_context_enabled(config) is False
    archive.set_metadata("release_ready", "1")
    assert legacy_fact_context_enabled(config) is False


def test_legacy_context_opt_out_survives_personal_memory_disablement(tmp_path):
    config = SimpleNamespace(
        personal_memory=SimpleNamespace(
            enabled=False,
            legacy_context_injection=False,
            archive_path=str(tmp_path / "personal.db"),
        )
    )

    assert legacy_fact_context_enabled(config) is False
