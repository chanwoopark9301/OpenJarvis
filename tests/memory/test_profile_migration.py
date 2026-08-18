"""Tests for reversible migration from static profile prompt files."""

from __future__ import annotations

from pathlib import Path

import pytest

from openjarvis.core.config import (
    JarvisConfig,
    MemoryFilesConfig,
    effective_chat_memory_files,
)
from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.personal_models import EvidenceSource
from openjarvis.memory.profile_migration import (
    LegacyProfileMigrator,
    activate_canonical_profile,
)


def _legacy_files(tmp_path: Path) -> tuple[Path, Path]:
    user_path = tmp_path / "USER.md"
    memory_path = tmp_path / "MEMORY.md"
    user_path.write_text(
        "# User\n\n- Prefers concise replies\n- Do not mention timers\n",
        encoding="utf-8",
    )
    memory_path.write_text(
        "# Memory\n\n* Preparing for an exam\n",
        encoding="utf-8",
    )
    return user_path, memory_path


def test_staging_backs_up_profile_files_and_keeps_them_active(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)

    result = LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert len(result.backup_paths) == 2
    assert all(path.is_file() for path in result.backup_paths)
    assert [path.read_text(encoding="utf-8") for path in result.backup_paths] == [
        user_path.read_text(encoding="utf-8"),
        memory_path.read_text(encoding="utf-8"),
    ]
    assert archive.get_metadata("canonical_profile_active", "0") == "0"


def test_staging_records_only_nonempty_bullets_as_low_trust_candidates(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)

    LegacyProfileMigrator(archive).stage(user_path, memory_path)

    candidates = archive.find_candidates(source=EvidenceSource.LEGACY_IMPORT)
    assert [candidate.content for candidate in candidates] == [
        "Prefers concise replies",
        "Do not mention timers",
        "Preparing for an exam",
    ]
    assert all(candidate.status.value == "pending" for candidate in candidates)
    assert archive.get_active_claims() == []
    provenance = [
        archive.get_exchange(candidate.exchange_id).source  # type: ignore[union-attr]
        for candidate in candidates
    ]
    assert provenance == [
        f"legacy-profile:{user_path}:3",
        f"legacy-profile:{user_path}:4",
        f"legacy-profile:{memory_path}:3",
    ]


def test_activation_refuses_without_readable_backups(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")

    with pytest.raises(ValueError, match="backup"):
        activate_canonical_profile(
            archive,
            backup_paths=(tmp_path / "missing.backup",),
        )

    assert archive.get_metadata("canonical_profile_active", "0") == "0"


def test_activation_refuses_when_local_archive_is_missing(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    backup = tmp_path / "USER.md.backup"
    backup.write_text("recoverable", encoding="utf-8")
    archive.path.unlink()

    with pytest.raises(ValueError, match="archive"):
        activate_canonical_profile(archive, backup_paths=(backup,))


def test_activation_stops_static_dynamic_profile_injection(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    backups = LegacyProfileMigrator(archive).stage(user_path, memory_path).backup_paths
    config = JarvisConfig()
    original_files = MemoryFilesConfig(
        soul_path=str(tmp_path / "SOUL.md"),
        user_path=str(user_path),
        memory_path=str(memory_path),
    )

    activate_canonical_profile(archive, backup_paths=backups)
    effective = effective_chat_memory_files(config, original_files, archive)

    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert effective.soul_path == original_files.soul_path
    assert effective.user_path == ""
    assert effective.memory_path == ""


def test_activation_cannot_rehydrate_named_persona_user_or_memory(
    tmp_path,
    monkeypatch,
):
    import openjarvis.prompt.builder as builder_module
    from openjarvis.prompt.builder import SystemPromptBuilder

    persona = tmp_path / "personas" / "private"
    persona.mkdir(parents=True)
    (persona / "SOUL.md").write_text("NAMED SOUL", encoding="utf-8")
    (persona / "USER.md").write_text("PRIVATE NAMED USER", encoding="utf-8")
    (persona / "MEMORY.md").write_text("PRIVATE NAMED MEMORY", encoding="utf-8")
    monkeypatch.setattr(builder_module, "get_config_dir", lambda: tmp_path)
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    archive.set_metadata("canonical_profile_active", "1")

    effective = effective_chat_memory_files(
        JarvisConfig(),
        MemoryFilesConfig(persona_name="private"),
        archive,
    )
    prompt = SystemPromptBuilder("agent", memory_files_config=effective).build()

    assert "NAMED SOUL" in prompt
    assert "PRIVATE NAMED USER" not in prompt
    assert "PRIVATE NAMED MEMORY" not in prompt


def test_staging_is_idempotent_for_the_same_profile_lines(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    migrator = LegacyProfileMigrator(archive)

    first = migrator.stage(user_path, memory_path)
    second = migrator.stage(user_path, memory_path)

    assert first.imported == 3
    assert second.imported == 0
    assert len(archive.find_candidates(source=EvidenceSource.LEGACY_IMPORT)) == 3
