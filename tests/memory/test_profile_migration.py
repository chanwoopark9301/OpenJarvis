"""Tests for reversible migration from static profile prompt files."""

from __future__ import annotations

import json
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


def _staging_manifest(archive: PersonalMemoryArchive) -> dict:
    return json.loads(archive.get_metadata("canonical_profile_staging_manifest"))


def test_activation_requires_backups_from_the_trusted_staging_manifest(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    unrelated = tmp_path / "unrelated.backup"
    unrelated.write_text("readable but untrusted", encoding="utf-8")

    with pytest.raises(ValueError, match="staged manifest"):
        activate_canonical_profile(archive, backup_paths=(unrelated,))

    assert archive.get_metadata("canonical_profile_active", "0") == "0"
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)


def test_activation_rejects_a_staged_backup_snapshot_mismatch(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    staged.backup_paths[0].write_text("changed after staging", encoding="utf-8")

    with pytest.raises(ValueError, match="snapshot"):
        activate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert archive.get_metadata("canonical_profile_active", "0") == "0"


def test_staging_imports_exact_backup_snapshot_not_mutable_source(
    tmp_path,
    monkeypatch,
):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    original_copy2 = __import__("shutil").copy2
    copy_count = 0

    def mutate_after_copy(source, target):
        nonlocal copy_count
        copied = original_copy2(source, target)
        copy_count += 1
        if copy_count == 1:
            Path(source).write_text("- MUTATED LIVE SOURCE", encoding="utf-8")
        return copied

    monkeypatch.setattr(
        "openjarvis.memory.profile_migration.shutil.copy2", mutate_after_copy
    )

    LegacyProfileMigrator(archive).stage(user_path, memory_path)

    contents = [candidate.content for candidate in archive.find_candidates()]
    assert "Prefers concise replies" in contents
    assert "MUTATED LIVE SOURCE" not in contents


def test_staging_rejects_symlink_source_without_importing(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    real_user, memory_path = _legacy_files(tmp_path)
    linked_user = tmp_path / "LINKED_USER.md"
    linked_user.symlink_to(real_user)

    with pytest.raises(ValueError, match="symlink"):
        LegacyProfileMigrator(archive).stage(linked_user, memory_path)

    assert archive.find_candidates() == []
    assert archive.get_metadata("canonical_profile_staging_manifest", "") == ""


def test_staging_rejects_backup_destination_collision_without_overwrite(
    tmp_path,
    monkeypatch,
):
    class FixedUuid:
        hex = "fixed"

    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    backup_dir = tmp_path / ".canonical-profile-backups"
    backup_dir.mkdir()
    collision = backup_dir / "USER.md.fixed.backup"
    collision.write_text("must remain", encoding="utf-8")
    monkeypatch.setattr(
        "openjarvis.memory.profile_migration.uuid.uuid4", lambda: FixedUuid()
    )

    with pytest.raises(ValueError, match="collision"):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert collision.read_text(encoding="utf-8") == "must remain"
    assert archive.find_candidates() == []


def test_invalid_utf8_staging_is_atomic(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path = tmp_path / "USER.md"
    memory_path = tmp_path / "MEMORY.md"
    user_path.write_text("- valid candidate", encoding="utf-8")
    memory_path.write_bytes(b"- invalid \xff")

    with pytest.raises(UnicodeError):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert archive.find_candidates() == []
    assert archive.get_metadata("canonical_profile_staging_manifest", "") == ""


def test_repeated_zero_and_partial_staging_preserves_recovery_manifest(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    migrator = LegacyProfileMigrator(archive)
    migrator.stage(user_path, memory_path)
    original = _staging_manifest(archive)
    user_path.unlink()
    memory_path.unlink()

    assert migrator.stage(user_path, memory_path).backup_paths == ()
    assert _staging_manifest(archive) == original

    user_path.write_text("- new user snapshot", encoding="utf-8")
    partial = migrator.stage(user_path, memory_path)
    merged = _staging_manifest(archive)
    assert len(partial.backup_paths) == 1
    assert {entry["source_path"] for entry in merged["entries"]} == {
        str(user_path.resolve()),
        str(memory_path.resolve()),
    }


def test_legacy_candidate_identity_is_case_sensitive(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path = tmp_path / "USER.md"
    memory_path = tmp_path / "MEMORY.md"
    user_path.write_text("- US", encoding="utf-8")
    migrator = LegacyProfileMigrator(archive)
    migrator.stage(user_path, memory_path)
    user_path.write_text("- us", encoding="utf-8")

    migrator.stage(user_path, memory_path)

    assert [candidate.content for candidate in archive.find_candidates()] == [
        "US",
        "us",
    ]


def test_repeated_activation_is_audited(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    backups = LegacyProfileMigrator(archive).stage(user_path, memory_path).backup_paths

    activate_canonical_profile(archive, backup_paths=backups)
    activate_canonical_profile(archive, backup_paths=backups)

    assert archive.decision_count(subject_id="canonical_profile") == 2


def test_existing_corrupt_archive_fails_closed_for_static_profile(tmp_path):
    archive_path = tmp_path / "corrupt.db"
    archive_path.write_bytes(b"not sqlite")
    user_path, memory_path = _legacy_files(tmp_path)
    config = JarvisConfig()
    config.personal_memory.archive_path = str(archive_path)
    original = MemoryFilesConfig(
        soul_path=str(tmp_path / "SOUL.md"),
        user_path=str(user_path),
        memory_path=str(memory_path),
    )

    effective = effective_chat_memory_files(config, original)

    assert effective.soul_path == original.soul_path
    assert effective.user_path == ""
    assert effective.memory_path == ""
