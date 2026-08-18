"""Tests for reversible migration from static profile prompt files."""

from __future__ import annotations

import gc
import json
import os
import sqlite3
import stat
from contextlib import ExitStack
from pathlib import Path

import pytest

import openjarvis.memory.profile_migration as migration_module
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
    deactivate_canonical_profile,
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


def test_deactivation_preserves_current_static_files_and_audits_cutover(tmp_path):
    """Rollback changes prompt authority without restoring stale snapshot bytes."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    user_path.write_text("CURRENT USER PROFILE", encoding="utf-8")
    memory_path.write_text("CURRENT MEMORY PROFILE", encoding="utf-8")

    deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert archive.get_metadata("canonical_profile_active", "1") == "0"
    assert archive.get_metadata("canonical_profile_deactivated_at", "")
    assert user_path.read_text(encoding="utf-8") == "CURRENT USER PROFILE"
    assert memory_path.read_text(encoding="utf-8") == "CURRENT MEMORY PROFILE"
    assert archive.decision_count(subject_id="canonical_profile") == 2


@pytest.mark.parametrize("failure", ("missing", "tampered", "replaced", "mode"))
def test_deactivation_fails_closed_when_attested_backup_changes(tmp_path, failure):
    """Rollback requires the same private snapshots attested at activation."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    target = staged.backup_paths[0]
    if failure == "missing":
        target.unlink()
    elif failure == "tampered":
        target.write_text("tampered", encoding="utf-8")
        target.chmod(0o600)
    elif failure == "replaced":
        original = target.read_bytes()
        target.unlink()
        target.write_bytes(original)
        target.chmod(0o600)
    else:
        target.chmod(0o644)

    with pytest.raises(ValueError, match="backup"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1


@pytest.mark.parametrize(
    ("target_kind", "weakened_mode"),
    (("file", 0o644), ("snapshot", 0o755), ("root", 0o755)),
)
def test_deactivation_revalidates_modes_changed_during_hash(
    tmp_path,
    monkeypatch,
    target_kind,
    weakened_mode,
):
    """Hash-time chmod must not bypass the private-backup boundary."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    backup = staged.backup_paths[0]
    targets = {
        "file": backup,
        "snapshot": backup.parent,
        "root": backup.parent.parent,
    }
    original_hash = migration_module._hash_open_file
    mutated = False

    def weaken_mode_while_hashing(fd):
        nonlocal mutated
        if not mutated:
            targets[target_kind].chmod(weakened_mode)
            mutated = True
        return original_hash(fd)

    monkeypatch.setattr(migration_module, "_hash_open_file", weaken_mode_while_hashing)

    with pytest.raises(ValueError, match="backup"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert mutated is True
    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1


@pytest.mark.parametrize("target_kind", ("file", "snapshot"))
def test_deactivation_revalidates_earlier_entry_after_later_hash(
    tmp_path,
    monkeypatch,
    target_kind,
):
    """Later validation work must not create a gap for an earlier snapshot."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    first_backup = staged.backup_paths[0]
    target = first_backup if target_kind == "file" else first_backup.parent
    weakened_mode = 0o644 if target_kind == "file" else 0o755
    original_hash = migration_module._hash_open_file
    hash_count = 0

    def weaken_first_entry_during_second_hash(fd):
        nonlocal hash_count
        hash_count += 1
        if hash_count == 2:
            target.chmod(weakened_mode)
        return original_hash(fd)

    monkeypatch.setattr(
        migration_module,
        "_hash_open_file",
        weaken_first_entry_during_second_hash,
    )

    with pytest.raises(ValueError, match="backup"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert hash_count == 2
    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1


@pytest.mark.parametrize("target_kind", ("file", "snapshot", "root"))
@pytest.mark.parametrize("mutation", ("owner", "type"))
def test_deactivation_revalidates_owner_and_type_after_hash(
    tmp_path,
    monkeypatch,
    target_kind,
    mutation,
):
    """Final descriptor metadata must be trusted, not the pre-hash snapshot."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    backup = staged.backup_paths[0]
    targets = {
        "file": backup,
        "snapshot": backup.parent,
        "root": backup.parent.parent,
    }
    target_stat = targets[target_kind].stat()
    target_identity = (target_stat.st_dev, target_stat.st_ino)
    original_hash = migration_module._hash_open_file
    original_fstat = migration_module.os.fstat
    armed = False

    def arm_after_hash(fd):
        nonlocal armed
        digest = original_hash(fd)
        armed = True
        return digest

    def altered_fstat(fd):
        observed = original_fstat(fd)
        if not armed or (observed.st_dev, observed.st_ino) != target_identity:
            return observed
        fields = list(observed)
        if mutation == "owner":
            fields[4] = os.geteuid() + 1
        else:
            replacement_type = (
                stat.S_IFDIR if stat.S_ISREG(observed.st_mode) else stat.S_IFREG
            )
            fields[0] = replacement_type | stat.S_IMODE(observed.st_mode)
        return os.stat_result(fields)

    monkeypatch.setattr(migration_module, "_hash_open_file", arm_after_hash)
    monkeypatch.setattr(migration_module.os, "fstat", altered_fstat)

    with pytest.raises(ValueError, match="backup"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert armed is True
    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1


@pytest.mark.parametrize("target_kind", ("file", "snapshot", "root"))
def test_deactivation_detects_path_identity_replacement_during_hash(
    tmp_path,
    monkeypatch,
    target_kind,
):
    """Every pinned backup path must still resolve to its attested inode."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    backup = staged.backup_paths[0]
    original_hash = migration_module._hash_open_file
    replaced = False

    def replace_path_after_hash(fd):
        nonlocal replaced
        digest = original_hash(fd)
        if replaced:
            return digest
        replaced = True
        if target_kind == "file":
            original = backup.with_name(f"{backup.name}.original")
            content = backup.read_bytes()
            backup.rename(original)
            backup.write_bytes(content)
            backup.chmod(0o600)
        elif target_kind == "snapshot":
            snapshot = backup.parent
            original = snapshot.with_name(f"{snapshot.name}.original")
            content = backup.read_bytes()
            snapshot.rename(original)
            snapshot.mkdir(mode=0o700)
            replacement = snapshot / backup.name
            replacement.write_bytes(content)
            replacement.chmod(0o600)
        else:
            root = backup.parent.parent
            original = root.with_name(f"{root.name}.original")
            root.rename(original)
            root.mkdir(mode=0o700)
        return digest

    monkeypatch.setattr(migration_module, "_hash_open_file", replace_path_after_hash)

    with pytest.raises(ValueError, match="backup"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert replaced is True
    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1


def test_deactivation_revalidates_after_validation_before_cutover(
    tmp_path,
    monkeypatch,
):
    """The DB cutover must recheck descriptors after initial validation."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    original_mark = archive.mark_canonical_profile_inactive
    mutated = False

    def mutate_before_mark(**kwargs):
        nonlocal mutated
        staged.backup_paths[0].chmod(0o644)
        mutated = True
        return original_mark(**kwargs)

    monkeypatch.setattr(archive, "mark_canonical_profile_inactive", mutate_before_mark)

    with pytest.raises(ValueError, match="backup"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert mutated is True
    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1


def test_deactivation_rehashes_after_validation_before_cutover(
    tmp_path,
    monkeypatch,
):
    """The pinned snapshot hash must still match inside the DB transaction."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    original_mark = archive.mark_canonical_profile_inactive
    mutated = False

    def mutate_before_mark(**kwargs):
        nonlocal mutated
        staged.backup_paths[0].write_text("tampered", encoding="utf-8")
        staged.backup_paths[0].chmod(0o600)
        mutated = True
        return original_mark(**kwargs)

    monkeypatch.setattr(archive, "mark_canonical_profile_inactive", mutate_before_mark)

    with pytest.raises(ValueError, match="snapshot"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert mutated is True
    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1


def test_deactivation_requires_exact_active_manifest_and_backup_order(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)

    with pytest.raises(ValueError, match="manifest"):
        deactivate_canonical_profile(
            archive,
            backup_paths=tuple(reversed(staged.backup_paths)),
        )

    manifest = archive.get_metadata("canonical_profile_staging_manifest")
    archive.set_metadata(
        "canonical_profile_staging_manifest",
        manifest.replace('"version":1', '"version":2'),
    )
    with pytest.raises(ValueError, match="manifest"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)
    assert archive.get_metadata("canonical_profile_active", "0") == "1"


def test_deactivation_metadata_and_audit_are_atomic(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    with sqlite3.connect(archive.path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_deactivation_audit
            BEFORE INSERT ON memory_decisions
            WHEN NEW.reason_code = 'canonical_profile_deactivated'
            BEGIN
              SELECT RAISE(ABORT, 'deactivation audit unavailable');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="audit unavailable"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1


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
    original_copy = migration_module._copy_fd_bytes
    copy_count = 0

    def mutate_after_copy(source_fd, target_fd):
        nonlocal copy_count
        original_copy(source_fd, target_fd)
        copy_count += 1
        if copy_count == 1:
            user_path.write_text("- MUTATED LIVE SOURCE", encoding="utf-8")

    monkeypatch.setattr(
        "openjarvis.memory.profile_migration._copy_fd_bytes", mutate_after_copy
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
    collision = backup_dir / "fixed"
    collision.mkdir()
    marker = collision / "must-remain"
    marker.write_text("must remain", encoding="utf-8")
    monkeypatch.setattr(
        "openjarvis.memory.profile_migration.uuid.uuid4", lambda: FixedUuid()
    )

    with pytest.raises(ValueError, match="collision"):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert marker.read_text(encoding="utf-8") == "must remain"
    assert archive.find_candidates() == []


def test_source_replacement_at_copy_boundary_uses_opened_descriptor(
    tmp_path,
    monkeypatch,
):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    attacker = tmp_path / "attacker.txt"
    attacker.write_text("- ATTACKER CONTENT", encoding="utf-8")
    original_copy = migration_module._copy_fd_bytes
    swapped = False

    def replace_source(source_fd, target_fd):
        nonlocal swapped
        if not swapped:
            user_path.unlink()
            user_path.symlink_to(attacker)
            swapped = True
        original_copy(source_fd, target_fd)

    monkeypatch.setattr(
        "openjarvis.memory.profile_migration._copy_fd_bytes", replace_source
    )

    result = LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert b"Prefers concise replies" in result.backup_paths[0].read_bytes()
    assert b"ATTACKER CONTENT" not in result.backup_paths[0].read_bytes()


def test_source_ancestor_replacement_does_not_read_attacker_file(
    tmp_path,
    monkeypatch,
):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    user_path = profile_dir / "USER.md"
    user_path.write_text("- ORIGINAL PROFILE", encoding="utf-8")
    attacker_dir = tmp_path / "attacker"
    attacker_dir.mkdir()
    (attacker_dir / "USER.md").write_text("- ATTACKER PROFILE", encoding="utf-8")
    original_copy = migration_module._copy_fd_bytes
    swapped = False

    def replace_ancestor(source_fd, target_fd):
        nonlocal swapped
        if not swapped:
            profile_dir.rename(tmp_path / "original-profile")
            profile_dir.symlink_to(attacker_dir, target_is_directory=True)
            swapped = True
        original_copy(source_fd, target_fd)

    monkeypatch.setattr(migration_module, "_copy_fd_bytes", replace_ancestor)

    result = LegacyProfileMigrator(archive).stage(user_path, tmp_path / "MEMORY.md")

    assert result.backup_paths[0].read_text(encoding="utf-8") == "- ORIGINAL PROFILE"


def test_backup_ancestor_replacement_cannot_write_attacker_tree(
    tmp_path,
    monkeypatch,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    archive = PersonalMemoryArchive(state_dir / "personal.db")
    user_path = tmp_path / "USER.md"
    user_path.write_text("- ORIGINAL PROFILE", encoding="utf-8")
    attacker_dir = tmp_path / "attacker-state"
    attacker_dir.mkdir()
    sentinel = attacker_dir / "sentinel"
    sentinel.write_text("untouched", encoding="utf-8")
    original_copy = migration_module._copy_fd_bytes

    def replace_ancestor(source_fd, target_fd):
        state_dir.rename(tmp_path / "original-state")
        state_dir.symlink_to(attacker_dir, target_is_directory=True)
        original_copy(source_fd, target_fd)

    monkeypatch.setattr(migration_module, "_copy_fd_bytes", replace_ancestor)

    with pytest.raises(ValueError, match="identity"):
        LegacyProfileMigrator(archive).stage(user_path, tmp_path / "MEMORY.md")

    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert list(attacker_dir.iterdir()) == [sentinel]


def test_snapshot_permissions_are_private(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)

    result = LegacyProfileMigrator(archive).stage(user_path, memory_path)

    backup_root = tmp_path / ".canonical-profile-backups"
    assert stat.S_IMODE(backup_root.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        for path in result.backup_paths
    )
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600 for path in result.backup_paths
    )


def test_snapshot_preserves_source_mtime(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    timestamp_ns = 1_700_000_000_123_456_789
    os.utime(user_path, ns=(timestamp_ns, timestamp_ns))

    result = LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert result.backup_paths[0].stat().st_mtime_ns == timestamp_ns


def test_unsupported_platform_refuses_before_backup_artifacts(tmp_path, monkeypatch):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    monkeypatch.setattr(
        migration_module,
        "_secure_snapshot_primitives_available",
        lambda: False,
    )

    with pytest.raises(ValueError, match="not supported"):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert not (tmp_path / ".canonical-profile-backups").exists()


def _fd_count() -> int:
    gc.collect()
    return len(os.listdir("/dev/fd"))


def test_backup_root_mode_is_repaired_before_root_identity_read(
    tmp_path,
    monkeypatch,
):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    backup_root = tmp_path / ".canonical-profile-backups"
    backup_root.mkdir(mode=0o700)
    backup_root.chmod(0o777)
    original_fstat = migration_module.os.fstat
    calls = 0

    def fail_root_fstat(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected root fstat failure")
        return original_fstat(fd)

    monkeypatch.setattr(migration_module.os, "fstat", fail_root_fstat)
    before = _fd_count()

    with pytest.raises(OSError, match="root fstat failure"):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert _fd_count() == before
    assert stat.S_IMODE(backup_root.stat().st_mode) == 0o700


def test_destination_mode_is_repaired_before_destination_identity_read(
    tmp_path,
    monkeypatch,
):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    original_open = migration_module.os.open
    original_fstat = migration_module.os.fstat
    destination_fd: int | None = None

    def open_with_restrictive_file_umask(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal destination_fd
        if mode != 0o600:
            return original_open(path, flags, mode, dir_fd=dir_fd)
        previous_umask = os.umask(0o700)
        try:
            destination_fd = original_open(path, flags, mode, dir_fd=dir_fd)
            return destination_fd
        finally:
            os.umask(previous_umask)

    def fail_destination_fstat(fd):
        if fd == destination_fd:
            raise OSError("injected destination fstat failure")
        return original_fstat(fd)

    monkeypatch.setattr(migration_module.os, "open", open_with_restrictive_file_umask)
    monkeypatch.setattr(migration_module.os, "fstat", fail_destination_fstat)
    monkeypatch.setattr(
        migration_module,
        "_secure_snapshot_primitives_available",
        lambda: True,
    )
    before = _fd_count()

    with pytest.raises(OSError, match="destination fstat failure"):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert _fd_count() == before
    backup_root = tmp_path / ".canonical-profile-backups"
    assert stat.S_IMODE(backup_root.stat().st_mode) == 0o700
    artifacts = list(backup_root.rglob("*"))
    assert any(path.is_file() for path in artifacts)
    for path in artifacts:
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected


def test_failed_close_does_not_retry_a_reused_descriptor(monkeypatch):
    original_close = migration_module.os.close
    victim_fd = os.open(os.devnull, os.O_RDONLY)
    remaining_fd = os.open(os.devnull, os.O_RDONLY)
    reused_fd: int | None = None
    released = False

    def release_then_raise(fd):
        nonlocal released, reused_fd
        if fd == victim_fd and not released:
            released = True
            original_close(fd)
            reused_fd = os.open(os.devnull, os.O_RDONLY)
            assert reused_fd == victim_fd
            raise OSError("close released descriptor before failing")
        original_close(fd)

    monkeypatch.setattr(migration_module.os, "close", release_then_raise)
    try:
        with pytest.raises(OSError, match="released descriptor"):
            with ExitStack() as stack:
                migration_module._stack_fd(stack, remaining_fd)
                migration_module._stack_fd(stack, victim_fd)

        assert reused_fd == victim_fd
        os.fstat(reused_fd)
        with pytest.raises(OSError):
            os.fstat(remaining_fd)
    finally:
        for fd in {victim_fd, remaining_fd, reused_fd} - {None}:
            try:
                original_close(fd)
            except OSError:
                pass


@pytest.mark.parametrize(
    "failure_hook",
    ("copy", "identity", "fsync", "fstat", "chmod", "cleanup"),
)
def test_snapshot_failure_closes_fds_and_leaves_private_modes(
    tmp_path,
    monkeypatch,
    failure_hook,
):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    if failure_hook == "copy":
        monkeypatch.setattr(
            migration_module,
            "_copy_fd_bytes",
            lambda *_args: (_ for _ in ()).throw(OSError("injected copy failure")),
        )
    elif failure_hook == "identity":
        monkeypatch.setattr(
            migration_module,
            "_verify_backup_path_identity",
            lambda *_args: (_ for _ in ()).throw(OSError("identity failure")),
        )
    elif failure_hook == "fsync":
        monkeypatch.setattr(
            migration_module.os,
            "fsync",
            lambda *_args: (_ for _ in ()).throw(OSError("fsync failure")),
        )
    elif failure_hook == "fstat":
        original_fstat = migration_module.os.fstat
        calls = 0

        def fail_fstat(fd):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("fstat failure")
            return original_fstat(fd)

        monkeypatch.setattr(migration_module.os, "fstat", fail_fstat)
    elif failure_hook == "chmod":
        original_fchmod = migration_module.os.fchmod
        failed = False

        def fail_chmod(fd, mode):
            nonlocal failed
            if mode == 0o600 and not failed:
                failed = True
                raise OSError("chmod failure")
            return original_fchmod(fd, mode)

        monkeypatch.setattr(migration_module.os, "fchmod", fail_chmod)
    else:
        original_close = migration_module._close_fd
        failed = False

        def fail_cleanup(fd):
            nonlocal failed
            original_close(fd)
            if not failed:
                failed = True
                raise OSError("cleanup failure")

        monkeypatch.setattr(migration_module, "_close_fd", fail_cleanup)

    before = _fd_count()
    with pytest.raises((OSError, ValueError)):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)
    assert _fd_count() == before
    backup_root = tmp_path / ".canonical-profile-backups"
    if backup_root.exists():
        assert stat.S_IMODE(backup_root.stat().st_mode) == 0o700
        for path in backup_root.rglob("*"):
            expected = 0o700 if path.is_dir() else 0o600
            assert stat.S_IMODE(path.stat().st_mode) == expected


def test_directory_chmod_failure_is_repaired_before_refusal(tmp_path, monkeypatch):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    backup_root = tmp_path / ".canonical-profile-backups"
    backup_root.mkdir(mode=0o777)
    backup_root.chmod(0o777)
    original_fchmod = migration_module.os.fchmod
    failed = False

    def fail_once(fd, mode):
        nonlocal failed
        if mode == 0o700 and not failed:
            failed = True
            raise OSError("injected directory chmod failure")
        return original_fchmod(fd, mode)

    monkeypatch.setattr(migration_module.os, "fchmod", fail_once)
    before = _fd_count()

    with pytest.raises(OSError, match="chmod failure"):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert _fd_count() == before
    assert stat.S_IMODE(backup_root.stat().st_mode) == 0o700


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


def test_supplied_archive_sqlite_error_fails_closed_for_static_profile(tmp_path):
    class CorruptArchive:
        def get_metadata(self, key, default=""):
            raise sqlite3.DatabaseError("database disk image is malformed")

    user_path, memory_path = _legacy_files(tmp_path)
    original = MemoryFilesConfig(
        soul_path=str(tmp_path / "SOUL.md"),
        user_path=str(user_path),
        memory_path=str(memory_path),
    )

    effective = effective_chat_memory_files(
        JarvisConfig(),
        original,
        CorruptArchive(),
    )

    assert effective.soul_path == original.soul_path
    assert effective.user_path == ""
    assert effective.memory_path == ""
