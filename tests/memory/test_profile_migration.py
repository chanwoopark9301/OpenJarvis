"""Tests for reversible migration from static profile prompt files."""

from __future__ import annotations

import gc
import json
import os
import sqlite3
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import pytest

import openjarvis.memory.archive as archive_module
import openjarvis.memory.profile_migration as migration_module
from openjarvis.core.config import (
    JarvisConfig,
    MemoryFilesConfig,
    effective_chat_memory_files,
)
from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import EvidenceSource
from openjarvis.memory.profile_migration import (
    LegacyProfileMigrator,
    activate_canonical_profile,
    deactivate_canonical_profile,
    resolve_staged_profile_manifest,
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


def test_repair_restores_only_automatically_rejected_staged_candidates(tmp_path):
    """A past worker rejection must not permanently erase explicit-review work."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    LegacyProfileMigrator(archive).stage(user_path, memory_path)
    candidate = archive.find_candidates(source=EvidenceSource.LEGACY_IMPORT)[0]
    job = archive.enqueue_job(
        job_type="evaluate_candidate",
        subject_id=candidate.id,
        idempotency_key=f"evaluate_candidate:{candidate.id}",
    )
    assert archive.claim_job(job.id) is not None
    result = MemoryEvaluator(archive).evaluate(candidate.id)
    assert result.reason_code == "unsupported_by_user_evidence"
    assert archive.get_candidate(candidate.id).status.value == "rejected"

    assert archive.repair_legacy_profile_candidate_lifecycle() == 1
    assert archive.repair_legacy_profile_candidate_lifecycle() == 0

    assert archive.get_candidate(candidate.id).status.value == "pending"
    assert archive.decision_count(subject_id=candidate.id) == 2
    assert archive.evidence_count(candidate_id=candidate.id) == 0
    assert archive.get_active_claims() == []
    assert archive.get_job(job.id).state.value == "complete"
    assert archive.recover_pending_candidate_jobs() == []


def test_repair_preserves_explicit_rejection_of_a_staged_candidate(tmp_path):
    """Only the known automatic evidence rejection is repairable."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    LegacyProfileMigrator(archive).stage(user_path, memory_path)
    candidate = archive.find_candidates(source=EvidenceSource.LEGACY_IMPORT)[0]
    assert archive.reject_candidate(candidate.id, reason_code="user_review_rejected")

    assert archive.repair_legacy_profile_candidate_lifecycle() == 0

    assert archive.get_candidate(candidate.id).status.value == "rejected"
    assert archive.decision_count(subject_id=candidate.id) == 1


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


def test_activation_attests_full_temporal_content_identity(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    backups = LegacyProfileMigrator(archive).stage(user_path, memory_path).backup_paths

    activate_canonical_profile(archive, backup_paths=backups)

    attestation = json.loads(
        archive.get_metadata("canonical_profile_active_backup_identities")
    )
    assert attestation["version"] == 2
    expected_fields = {
        "path",
        "device",
        "inode",
        "uid",
        "mode",
        "size",
        "mtime_ns",
        "ctime_ns",
    }
    assert set(attestation["root"]) == expected_fields
    assert all(
        set(identity) == expected_fields
        for entry in attestation["entries"]
        for identity in (entry["directory"], entry["file"])
    )


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

    assert hash_count >= 2
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


@pytest.mark.parametrize("target_kind", ("file", "snapshot", "root"))
def test_deactivation_reopens_paths_after_every_precommit_hash(
    tmp_path,
    monkeypatch,
    target_kind,
):
    """A pathname replaced during the final hash pass must fail closed."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    original_user = user_path.read_bytes()
    original_memory = memory_path.read_bytes()
    backup = staged.backup_paths[0]
    original_mark = archive.mark_canonical_profile_inactive
    original_hash = migration_module._hash_open_file
    replaced = False

    def mark_with_hash_race(**kwargs):
        hash_count = 0

        def replace_after_first_hash(fd):
            nonlocal hash_count, replaced
            hash_count += 1
            digest = original_hash(fd)
            if hash_count != 1:
                return digest
            replaced = True
            if target_kind == "file":
                content = backup.read_bytes()
                backup.rename(backup.with_name(f"{backup.name}.original"))
                backup.write_bytes(content)
                backup.chmod(0o600)
            elif target_kind == "snapshot":
                snapshot = backup.parent
                content = backup.read_bytes()
                snapshot.rename(snapshot.with_name(f"{snapshot.name}.original"))
                snapshot.mkdir(mode=0o700)
                replacement = snapshot / backup.name
                replacement.write_bytes(content)
                replacement.chmod(0o600)
            else:
                root = backup.parent.parent
                replacements = [
                    (path.parent.name, path.name, path.read_bytes())
                    for path in staged.backup_paths
                ]
                root.rename(root.with_name(f"{root.name}.original"))
                root.mkdir(mode=0o700)
                for snapshot_name, backup_name, content in replacements:
                    snapshot = root / snapshot_name
                    snapshot.mkdir(mode=0o700, exist_ok=True)
                    replacement = snapshot / backup_name
                    replacement.write_bytes(content)
                    replacement.chmod(0o600)
            return digest

        monkeypatch.setattr(
            migration_module,
            "_hash_open_file",
            replace_after_first_hash,
        )
        return original_mark(**kwargs)

    monkeypatch.setattr(
        archive,
        "mark_canonical_profile_inactive",
        mark_with_hash_race,
    )

    with pytest.raises(ValueError, match="backup"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert replaced is True
    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1
    assert user_path.read_bytes() == original_user
    assert memory_path.read_bytes() == original_memory


def test_deactivation_revalidates_after_uncommitted_audit_write(
    tmp_path,
    monkeypatch,
):
    """The last filesystem check must follow all uncommitted DB writes."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    original_user = user_path.read_bytes()
    original_memory = memory_path.read_bytes()
    first_backup = staged.backup_paths[0]
    original_uuid4 = archive_module.uuid.uuid4
    mutated = False

    def mutate_during_audit_write():
        nonlocal mutated
        if not mutated:
            first_backup.write_bytes(first_backup.read_bytes() + b"!")
            first_backup.chmod(0o600)
            mutated = True
        return original_uuid4()

    monkeypatch.setattr(archive_module.uuid, "uuid4", mutate_during_audit_write)

    with pytest.raises(ValueError, match="backup"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert mutated is True
    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1
    assert user_path.read_bytes() == original_user
    assert memory_path.read_bytes() == original_memory


def test_deactivation_rechecks_earlier_content_after_later_precommit_hash(
    tmp_path,
    monkeypatch,
):
    """Later hashes must not leave earlier pinned content unchecked."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    original_user = user_path.read_bytes()
    original_memory = memory_path.read_bytes()
    first_backup = staged.backup_paths[0]
    original_mark = archive.mark_canonical_profile_inactive
    original_hash = migration_module._hash_open_file
    mutated = False

    def mark_with_hash_race(**kwargs):
        hash_count = 0

        def mutate_first_during_second_hash(fd):
            nonlocal hash_count, mutated
            hash_count += 1
            if hash_count == 2:
                original = first_backup.read_bytes()
                original_stat = first_backup.stat()
                replacement = (b"!" if original[:1] != b"!" else b"?") + original[1:]
                first_backup.write_bytes(replacement)
                os.utime(
                    first_backup,
                    ns=(
                        original_stat.st_atime_ns,
                        original_stat.st_mtime_ns + 1_000_000_000,
                    ),
                )
                first_backup.chmod(0o600)
                mutated = True
            return original_hash(fd)

        monkeypatch.setattr(
            migration_module,
            "_hash_open_file",
            mutate_first_during_second_hash,
        )
        return original_mark(**kwargs)

    monkeypatch.setattr(
        archive,
        "mark_canonical_profile_inactive",
        mark_with_hash_race,
    )

    with pytest.raises(ValueError, match="backup"):
        deactivate_canonical_profile(archive, backup_paths=staged.backup_paths)

    assert mutated is True
    assert archive.get_metadata("canonical_profile_active", "0") == "1"
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == 1
    assert user_path.read_bytes() == original_user
    assert memory_path.read_bytes() == original_memory


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


def test_concurrent_stagers_publish_every_owned_snapshot_without_orphans(
    tmp_path,
    monkeypatch,
):
    """Two publish-ready stages must merge under the same write transaction."""
    archive_path = tmp_path / "personal.db"
    archives = (
        PersonalMemoryArchive(archive_path),
        PersonalMemoryArchive(archive_path),
    )
    source_pairs = []
    for owner in ("alpha", "beta"):
        source_dir = tmp_path / owner
        source_dir.mkdir()
        user_path = source_dir / "USER.md"
        memory_path = source_dir / "MEMORY.md"
        user_path.write_text(f"- {owner} user", encoding="utf-8")
        memory_path.write_text(f"- {owner} memory", encoding="utf-8")
        source_pairs.append((user_path, memory_path))
    publish_barrier = threading.Barrier(2)
    original_stage = PersonalMemoryArchive.stage_profile_candidates

    def publish_together(self, *args, **kwargs):
        publish_barrier.wait(timeout=10)
        return original_stage(self, *args, **kwargs)

    monkeypatch.setattr(
        PersonalMemoryArchive,
        "stage_profile_candidates",
        publish_together,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(LegacyProfileMigrator(archive).stage, *sources)
            for archive, sources in zip(archives, source_pairs)
        ]
        results = [future.result(timeout=10) for future in futures]

    manifest = resolve_staged_profile_manifest(archives[0])
    expected_sources = {
        str(path) for source_pair in source_pairs for path in source_pair
    }
    created_backups = {path for result in results for path in result.backup_paths}
    backup_root = tmp_path / ".canonical-profile-backups"
    disk_backups = {
        path
        for snapshot_dir in backup_root.iterdir()
        for path in snapshot_dir.iterdir()
    }

    assert {entry.source_path for entry in manifest.entries} == expected_sources
    assert set(manifest.backup_paths) == created_backups
    assert (
        tuple(
            Path(path)
            for path in json.loads(
                archives[0].get_metadata("canonical_profile_backup_paths")
            )
        )
        == manifest.backup_paths
    )
    assert disk_backups == created_backups


def test_staging_transaction_failure_cleans_only_new_snapshots(tmp_path):
    """A rolled-back publish must not orphan or delete another owner's snapshot."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    backup_root = tmp_path / ".canonical-profile-backups"
    foreign_snapshot = backup_root / "foreign-snapshot"
    foreign_snapshot.mkdir(parents=True, mode=0o700)
    foreign_marker = foreign_snapshot / "foreign-marker"
    foreign_marker.write_text("foreign", encoding="utf-8")
    foreign_marker.chmod(0o600)
    with sqlite3.connect(archive.path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_profile_stage
            BEFORE INSERT ON memory_candidates
            BEGIN
              SELECT RAISE(ABORT, 'profile stage unavailable');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="stage unavailable"):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    assert list(backup_root.iterdir()) == [foreign_snapshot]
    assert foreign_marker.read_text(encoding="utf-8") == "foreign"
    assert archive.get_metadata("canonical_profile_staging_manifest", "") == ""
    assert archive.find_candidates() == []


def test_snapshot_copy_failure_removes_owned_partial_snapshot(
    tmp_path,
    monkeypatch,
):
    """A copy error after exclusive creation must not leave an owned orphan."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)

    def fail_copy(*_args):
        raise OSError("injected copy failure")

    monkeypatch.setattr(migration_module, "_copy_fd_bytes", fail_copy)

    with pytest.raises(OSError, match="copy failure"):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    backup_root = tmp_path / ".canonical-profile-backups"
    assert list(backup_root.iterdir()) == []
    assert archive.get_metadata("canonical_profile_staging_manifest", "") == ""


def test_snapshot_identity_failure_removes_owned_empty_directory(
    tmp_path,
    monkeypatch,
):
    """Ownership starts at exclusive mkdir, before snapshot identity completes."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    original_fstat = migration_module.os.fstat
    fstat_calls = 0

    def fail_first_snapshot_identity(fd):
        nonlocal fstat_calls
        fstat_calls += 1
        if fstat_calls == 3:
            raise OSError("injected snapshot identity failure")
        return original_fstat(fd)

    monkeypatch.setattr(migration_module.os, "fstat", fail_first_snapshot_identity)

    with pytest.raises(OSError, match="snapshot identity failure"):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    backup_root = tmp_path / ".canonical-profile-backups"
    assert list(backup_root.iterdir()) == []
    assert archive.get_metadata("canonical_profile_staging_manifest", "") == ""


def test_staging_revalidates_manifest_inside_publish_transaction_and_cleans(
    tmp_path,
    monkeypatch,
):
    """A manifest changed after preflight must fail before publish without orphans."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    original_stage = archive.stage_profile_candidates

    def corrupt_before_transaction(*args, **kwargs):
        archive.set_metadata("canonical_profile_staging_manifest", '{"invalid":true}')
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(
        archive,
        "stage_profile_candidates",
        corrupt_before_transaction,
    )

    with pytest.raises(ValueError, match="trusted staged manifest is invalid"):
        LegacyProfileMigrator(archive).stage(user_path, memory_path)

    backup_root = tmp_path / ".canonical-profile-backups"
    assert list(backup_root.iterdir()) == []
    assert archive.find_candidates() == []


def test_staging_canonicalizes_legacy_alias_before_normal_restage(tmp_path):
    """A legacy alias and its normal spelling must remain one parseable entry."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    alias_segment = tmp_path / "alias-segment"
    alias_segment.mkdir()
    aliased_user_path = alias_segment / ".." / user_path.name
    migrator = LegacyProfileMigrator(archive)
    migrator.stage(aliased_user_path, memory_path)
    legacy_manifest = _staging_manifest(archive)
    legacy_manifest["entries"][0]["source_path"] = str(aliased_user_path)
    archive.set_metadata(
        "canonical_profile_staging_manifest",
        json.dumps(legacy_manifest, separators=(",", ":")),
    )

    migrator.stage(user_path, tmp_path / "missing-memory.md")

    manifest = resolve_staged_profile_manifest(archive)
    assert [entry.source_path for entry in manifest.entries] == [
        str(user_path),
        str(memory_path),
    ]
    assert len(set(manifest.backup_paths)) == 2
    assert all(".." not in Path(entry.source_path).parts for entry in manifest.entries)
    assert all(".." not in path.parts for path in manifest.backup_paths)


def test_staging_rejects_duplicate_logical_source_alias_without_artifacts(tmp_path):
    """One stage cannot create two snapshots for the same lexical source."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, _ = _legacy_files(tmp_path)
    alias_segment = tmp_path / "private-alias-segment"
    alias_segment.mkdir()
    aliased_user_path = alias_segment / ".." / user_path.name

    with pytest.raises(ValueError) as caught:
        LegacyProfileMigrator(archive).stage(aliased_user_path, user_path)

    assert str(caught.value) == "legacy profile paths are not unique"
    assert str(user_path) not in str(caught.value)
    assert "Prefers concise replies" not in str(caught.value)
    assert not (tmp_path / ".canonical-profile-backups").exists()
    assert archive.get_metadata("canonical_profile_staging_manifest", "") == ""


def test_archive_path_alias_stages_and_activates_canonical_backups(tmp_path):
    """Archive dot segments cannot split staging and activation root identity."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    alias_segment = tmp_path / "archive-alias-segment"
    alias_segment.mkdir()
    archive = PersonalMemoryArchive(
        alias_segment / ".." / state_dir.name / "personal.db"
    )
    user_path, memory_path = _legacy_files(tmp_path)

    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    activate_canonical_profile(archive, backup_paths=staged.backup_paths)

    manifest = resolve_staged_profile_manifest(archive)
    assert manifest.backup_paths == staged.backup_paths
    assert all(".." not in path.parts for path in manifest.backup_paths)
    assert archive.get_metadata("canonical_profile_active", "0") == "1"


def _staging_manifest(archive: PersonalMemoryArchive) -> dict:
    return json.loads(archive.get_metadata("canonical_profile_staging_manifest"))


_CORRUPT_MANIFEST_CASES = (
    "missing_source_path",
    "missing_backup_path",
    "missing_sha256",
    "source_path_not_string",
    "backup_path_not_string",
    "sha256_not_string",
    "sha256_wrong_length",
    "sha256_not_hex",
    "entries_not_list",
    "version_not_integer",
    "version_boolean",
    "version_float",
    "top_level_not_object",
    "entry_not_object",
    "unexpected_top_level_key",
    "source_path_empty",
    "backup_path_empty",
    "source_path_not_absolute",
    "backup_path_not_absolute",
    "source_path_contains_nul",
    "backup_path_contains_nul",
    "duplicate_source_path",
    "duplicate_backup_path",
    "duplicate_backup_path_alias",
    "duplicate_source_parent_alias",
    "duplicate_backup_parent_alias",
    "pathologically_nested_json",
)
_PRIVATE_MANIFEST_MARKER = "private-manifest-marker"


def _corrupt_staging_manifest(
    archive: PersonalMemoryArchive,
    corruption: str,
) -> None:
    manifest = _staging_manifest(archive)
    if corruption == "pathologically_nested_json":
        archive.set_metadata(
            "canonical_profile_staging_manifest",
            "[" * 10_000 + "0" + "]" * 10_000,
        )
        return
    if corruption == "top_level_not_object":
        archive.set_metadata(
            "canonical_profile_staging_manifest",
            json.dumps([_PRIVATE_MANIFEST_MARKER]),
        )
        return
    if corruption == "entry_not_object":
        manifest["entries"][0] = _PRIVATE_MANIFEST_MARKER
        archive.set_metadata(
            "canonical_profile_staging_manifest",
            json.dumps(manifest, separators=(",", ":")),
        )
        return
    entry = manifest["entries"][0]
    entry["source_path"] = f"/{_PRIVATE_MANIFEST_MARKER}"
    if corruption == "missing_source_path":
        entry.pop("source_path")
    elif corruption == "missing_backup_path":
        entry.pop("backup_path")
    elif corruption == "missing_sha256":
        entry.pop("sha256")
    elif corruption == "source_path_not_string":
        entry["source_path"] = {"private": _PRIVATE_MANIFEST_MARKER}
    elif corruption == "backup_path_not_string":
        entry["backup_path"] = [_PRIVATE_MANIFEST_MARKER]
    elif corruption == "sha256_not_string":
        entry["sha256"] = 7
    elif corruption == "sha256_wrong_length":
        entry["sha256"] = "a" * 63
    elif corruption == "sha256_not_hex":
        entry["sha256"] = "z" * 64
    elif corruption == "entries_not_list":
        manifest["entries"] = {"private": _PRIVATE_MANIFEST_MARKER}
    elif corruption == "version_not_integer":
        manifest["version"] = "1"
    elif corruption == "version_boolean":
        manifest["version"] = True
    elif corruption == "version_float":
        manifest["version"] = 1.0
    elif corruption == "unexpected_top_level_key":
        manifest["private"] = _PRIVATE_MANIFEST_MARKER
    elif corruption == "source_path_empty":
        entry["source_path"] = ""
    elif corruption == "backup_path_empty":
        entry["backup_path"] = ""
    elif corruption == "source_path_not_absolute":
        entry["source_path"] = _PRIVATE_MANIFEST_MARKER
    elif corruption == "backup_path_not_absolute":
        entry["backup_path"] = _PRIVATE_MANIFEST_MARKER
    elif corruption == "source_path_contains_nul":
        entry["source_path"] = f"/{_PRIVATE_MANIFEST_MARKER}\0suffix"
    elif corruption == "backup_path_contains_nul":
        entry["backup_path"] = f"/{_PRIVATE_MANIFEST_MARKER}\0suffix"
    elif corruption == "duplicate_source_path":
        manifest["entries"][1]["source_path"] = entry["source_path"]
    elif corruption == "duplicate_backup_path":
        manifest["entries"][1]["backup_path"] = entry["backup_path"]
        manifest["entries"][1]["sha256"] = entry["sha256"]
    elif corruption == "duplicate_backup_path_alias":
        backup = Path(entry["backup_path"])
        alias = backup.parent / "." / backup.name
        manifest["entries"][1]["backup_path"] = f"{alias.parent}/./{alias.name}"
        manifest["entries"][1]["sha256"] = entry["sha256"]
    elif corruption == "duplicate_source_parent_alias":
        source = Path(entry["source_path"])
        manifest["entries"][1]["source_path"] = f"/alias-segment/../{source.name}"
    elif corruption == "duplicate_backup_parent_alias":
        backup = Path(entry["backup_path"])
        manifest["entries"][1]["backup_path"] = (
            f"{backup.parent}/../{backup.parent.name}/{backup.name}"
        )
        manifest["entries"][1]["sha256"] = entry["sha256"]
    else:  # pragma: no cover - test fixture contract
        raise AssertionError(f"unknown corruption: {corruption}")
    archive.set_metadata(
        "canonical_profile_staging_manifest",
        json.dumps(manifest, separators=(",", ":")),
    )


@pytest.mark.parametrize("operation", ("activate", "deactivate"))
@pytest.mark.parametrize("corruption", _CORRUPT_MANIFEST_CASES)
def test_canonical_profile_api_rejects_corrupt_staging_manifest(
    tmp_path,
    operation,
    corruption,
):
    """Malformed trusted metadata must remain a non-content ValueError."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    staged = LegacyProfileMigrator(archive).stage(user_path, memory_path)
    if operation == "deactivate":
        activate_canonical_profile(archive, backup_paths=staged.backup_paths)
    user_path.write_text("CURRENT PRIVATE USER", encoding="utf-8")
    memory_path.write_text("CURRENT PRIVATE MEMORY", encoding="utf-8")
    active_before = archive.get_metadata("canonical_profile_active", "0")
    decisions_before = archive.decision_count(subject_id="canonical_profile")
    _corrupt_staging_manifest(archive, corruption)

    command = (
        activate_canonical_profile
        if operation == "activate"
        else deactivate_canonical_profile
    )
    with pytest.raises(ValueError) as caught:
        command(archive, backup_paths=staged.backup_paths)

    expected_error = (
        "trusted staged manifest is unreadable"
        if corruption == "pathologically_nested_json"
        else "trusted staged manifest is invalid"
    )
    assert str(caught.value) == expected_error
    assert _PRIVATE_MANIFEST_MARKER not in str(caught.value)
    assert str(staged.backup_paths[0]) not in str(caught.value)
    assert archive.get_metadata("canonical_profile_active", "0") == active_before
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == decisions_before
    assert user_path.read_text(encoding="utf-8") == "CURRENT PRIVATE USER"
    assert memory_path.read_text(encoding="utf-8") == "CURRENT PRIVATE MEMORY"


@pytest.mark.parametrize(
    "corruption",
    ("missing_backup_path", "duplicate_source_parent_alias"),
)
def test_staging_rejects_corrupt_manifest_before_creating_snapshots(
    tmp_path,
    corruption,
):
    """Trusted metadata must be parsed before staging writes recovery files."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    migrator = LegacyProfileMigrator(archive)
    migrator.stage(user_path, memory_path)
    _corrupt_staging_manifest(archive, corruption)
    manifest_before = archive.get_metadata("canonical_profile_staging_manifest")
    candidates_before = len(archive.find_candidates())
    backup_root = tmp_path / ".canonical-profile-backups"
    snapshots_before = {path.name for path in backup_root.iterdir()}
    user_before = user_path.read_bytes()
    memory_before = memory_path.read_bytes()

    with pytest.raises(
        ValueError, match="trusted staged manifest is invalid"
    ) as caught:
        migrator.stage(user_path, memory_path)

    assert _PRIVATE_MANIFEST_MARKER not in str(caught.value)
    assert {path.name for path in backup_root.iterdir()} == snapshots_before
    assert archive.get_metadata("canonical_profile_staging_manifest") == manifest_before
    assert len(archive.find_candidates()) == candidates_before
    assert user_path.read_bytes() == user_before
    assert memory_path.read_bytes() == memory_before


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


@pytest.mark.parametrize("replace_owned_path", (False, True))
def test_snapshot_close_after_release_cleans_only_matching_owned_path(
    tmp_path,
    monkeypatch,
    replace_owned_path,
):
    """ExitStack teardown errors retain ownership without retrying a closed fd."""
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    user_path, memory_path = _legacy_files(tmp_path)
    backup_root = tmp_path / ".canonical-profile-backups"
    foreign_snapshot = backup_root / "foreign-snapshot"
    foreign_snapshot.mkdir(parents=True, mode=0o700)
    foreign_marker = foreign_snapshot / "foreign-marker"
    foreign_marker.write_text("foreign", encoding="utf-8")
    foreign_marker.chmod(0o600)
    original_open = migration_module.os.open
    original_close = migration_module._close_fd
    destination_fd: int | None = None
    reused_fd: int | None = None
    close_calls: dict[int, int] = {}
    replacement: Path | None = None

    def capture_destination(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal destination_fd
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if mode == 0o600 and destination_fd is None:
            destination_fd = fd
        return fd

    def release_then_raise(fd):
        nonlocal reused_fd, replacement
        close_calls[fd] = close_calls.get(fd, 0) + 1
        original_close(fd)
        if fd != destination_fd or reused_fd is not None:
            return
        reused_fd = original_open(os.devnull, os.O_RDONLY)
        assert reused_fd == fd
        if replace_owned_path:
            owned_snapshot = next(
                path for path in backup_root.iterdir() if path != foreign_snapshot
            )
            owned_backup = next(owned_snapshot.iterdir())
            owned_backup.rename(owned_snapshot / "owned-original")
            replacement = owned_snapshot / owned_backup.name
            replacement.write_text("replacement", encoding="utf-8")
            replacement.chmod(0o600)
        raise OSError("released destination close")

    monkeypatch.setattr(migration_module.os, "open", capture_destination)
    monkeypatch.setattr(migration_module, "_close_fd", release_then_raise)
    monkeypatch.setattr(
        migration_module,
        "_secure_snapshot_primitives_available",
        lambda: True,
    )
    before = _fd_count()
    try:
        with pytest.raises(OSError, match="released destination close"):
            LegacyProfileMigrator(archive).stage(user_path, memory_path)

        assert destination_fd is not None
        assert reused_fd == destination_fd
        assert close_calls[destination_fd] == 1
        os.fstat(reused_fd)
        assert foreign_marker.read_text(encoding="utf-8") == "foreign"
        if replace_owned_path:
            assert replacement is not None
            assert replacement.read_text(encoding="utf-8") == "replacement"
            assert len(list(backup_root.iterdir())) == 2
        else:
            assert list(backup_root.iterdir()) == [foreign_snapshot]
        assert archive.get_metadata("canonical_profile_staging_manifest", "") == ""
        assert archive.find_candidates() == []
    finally:
        if reused_fd is not None:
            os.close(reused_fd)
    assert _fd_count() == before


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
