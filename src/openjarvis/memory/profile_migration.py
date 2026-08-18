"""Reversible migration from static profile files to canonical memory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.personal_models import (
    CandidateDraft,
    CandidateKind,
    EvidenceSource,
)

_BULLET = re.compile(r"^\s*[-*+]\s+(.+?)\s*$")
_BACKUP_METADATA_KEY = "canonical_profile_backup_paths"
_STAGING_MANIFEST_KEY = "canonical_profile_staging_manifest"


@dataclass(frozen=True, slots=True)
class ProfileMigrationResult:
    """Non-sensitive totals and recovery paths from one staging run."""

    backup_paths: tuple[Path, ...]
    imported: int
    skipped: int


class LegacyProfileMigrator:
    """Back up legacy files and stage bullets as untrusted candidates."""

    def __init__(self, archive: PersonalMemoryArchive) -> None:
        self.archive = archive

    def stage(
        self,
        user_path: Path,
        memory_path: Path,
    ) -> ProfileMigrationResult:
        """Copy, preflight, then atomically import exact profile snapshots."""
        sources = tuple(
            Path(path).expanduser().absolute() for path in (user_path, memory_path)
        )
        backup_dir = self.archive.path.parent / ".canonical-profile-backups"
        backup_root_fd = _open_private_backup_root(backup_dir)
        snapshots: list[tuple[Path, Path, bytes]] = []
        try:
            for source in sources:
                if not source.exists():
                    continue
                if source.is_symlink():
                    raise ValueError(
                        f"legacy profile path must not be a symlink: {source}"
                    )
                if not source.is_file():
                    raise ValueError(f"legacy profile path is not a file: {source}")
                backup, snapshot = _copy2_private_snapshot(
                    source,
                    backup_dir,
                    backup_root_fd,
                )
                snapshot.decode("utf-8")
                snapshots.append((source, backup, snapshot))
        finally:
            os.close(backup_root_fd)

        old_manifest = _read_manifest(self.archive)
        merged = {
            str(entry["source_path"]): entry
            for entry in old_manifest.get("entries", [])
        }
        records: list[tuple[str, str, CandidateDraft]] = []
        for source, backup, snapshot in snapshots:
            snapshot_hash = hashlib.sha256(snapshot).hexdigest()
            merged[str(source)] = {
                "source_path": str(source),
                "backup_path": str(backup),
                "sha256": snapshot_hash,
            }
            for line_number, raw_line in enumerate(
                snapshot.decode("utf-8").splitlines(), start=1
            ):
                match = _BULLET.match(raw_line)
                content = match.group(1).strip() if match is not None else ""
                if not content:
                    continue
                digest_input = f"{source}:{line_number}:{content}"
                digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
                exchange_id = f"legacy-profile-{digest}"
                records.append(
                    (
                        exchange_id,
                        f"legacy-profile:{source}:{line_number}",
                        CandidateDraft(
                            CandidateKind.FACT,
                            content,
                            0.1,
                            0.1,
                            source=EvidenceSource.LEGACY_IMPORT,
                            temporal_scope="unspecified",
                            subject="legacy_profile",
                        ),
                    )
                )

        if not snapshots:
            return ProfileMigrationResult(backup_paths=(), imported=0, skipped=0)
        manifest = {"version": 1, "entries": list(merged.values())}
        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        all_backup_paths = [entry["backup_path"] for entry in manifest["entries"]]
        imported, skipped = self.archive.stage_profile_candidates(
            records,
            metadata={
                _STAGING_MANIFEST_KEY: manifest_json,
                _BACKUP_METADATA_KEY: json.dumps(
                    all_backup_paths, separators=(",", ":")
                ),
            },
        )
        return ProfileMigrationResult(
            backup_paths=tuple(backup for _, backup, _ in snapshots),
            imported=imported,
            skipped=skipped,
        )


def _directory_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_private_backup_root(backup_dir: Path) -> int:
    """Open or atomically create the non-symlink private backup root."""
    try:
        os.mkdir(backup_dir, mode=0o700)
    except FileExistsError:
        pass
    try:
        root_fd = os.open(backup_dir, _directory_open_flags())
    except OSError as exc:
        raise ValueError("profile backup directory is unsafe") from exc
    root_stat = os.fstat(root_fd)
    if not stat.S_ISDIR(root_stat.st_mode):
        os.close(root_fd)
        raise ValueError("profile backup directory is unsafe")
    os.fchmod(root_fd, 0o700)
    return root_fd


def _copy2_private_snapshot(
    source: Path,
    backup_dir: Path,
    backup_root_fd: int,
) -> tuple[Path, bytes]:
    """Reserve an unreplaceable destination, then copy metadata and bytes."""
    snapshot_name = uuid.uuid4().hex
    try:
        os.mkdir(snapshot_name, mode=0o700, dir_fd=backup_root_fd)
    except FileExistsError as exc:
        raise ValueError("profile backup destination collision") from exc
    snapshot_fd = os.open(
        snapshot_name,
        _directory_open_flags(),
        dir_fd=backup_root_fd,
    )
    backup = (backup_dir / snapshot_name / source.name).absolute()
    file_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        backup_fd = os.open(source.name, file_flags, 0o600, dir_fd=snapshot_fd)
    except Exception:
        os.close(snapshot_fd)
        raise
    root_identity = os.fstat(backup_root_fd)
    snapshot_identity = os.fstat(snapshot_fd)
    backup_identity = os.fstat(backup_fd)
    try:
        # Removing directory write permission makes the reserved inode impossible
        # to replace between the validation and copy2's destination open.
        os.fchmod(snapshot_fd, 0o500)
        os.fchmod(backup_root_fd, 0o500)
        path_root = os.lstat(backup_dir)
        path_snapshot = os.lstat(backup.parent)
        path_backup = os.lstat(backup)
        if (
            (path_root.st_dev, path_root.st_ino)
            != (root_identity.st_dev, root_identity.st_ino)
            or (path_snapshot.st_dev, path_snapshot.st_ino)
            != (snapshot_identity.st_dev, snapshot_identity.st_ino)
            or (path_backup.st_dev, path_backup.st_ino)
            != (backup_identity.st_dev, backup_identity.st_ino)
            or not stat.S_ISREG(path_backup.st_mode)
        ):
            raise ValueError("profile backup destination is unsafe")
        shutil.copy2(source, backup, follow_symlinks=False)
        copied = os.lstat(backup)
        if (copied.st_dev, copied.st_ino) != (
            backup_identity.st_dev,
            backup_identity.st_ino,
        ) or not stat.S_ISREG(copied.st_mode):
            raise ValueError("profile backup destination is unsafe")
        os.fchmod(backup_fd, 0o600)
        os.fsync(backup_fd)
        os.lseek(backup_fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(backup_fd), "rb") as snapshot_handle:
            snapshot = snapshot_handle.read()
    finally:
        os.fchmod(snapshot_fd, 0o700)
        os.fchmod(backup_root_fd, 0o700)
        os.close(backup_fd)
        os.close(snapshot_fd)
    return backup, snapshot


def _read_manifest(archive: PersonalMemoryArchive) -> dict:
    raw = archive.get_metadata(_STAGING_MANIFEST_KEY, "")
    if not raw:
        return {"version": 1, "entries": []}
    try:
        manifest = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("trusted staged manifest is unreadable") from exc
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or not isinstance(entries, list)
    ):
        raise ValueError("trusted staged manifest is invalid")
    required = {"source_path", "backup_path", "sha256"}
    if any(not isinstance(entry, dict) or set(entry) != required for entry in entries):
        raise ValueError("trusted staged manifest is invalid")
    return manifest


def validate_local_personal_archive(path: str | Path) -> None:
    """Validate an existing archive without creating or mutating it."""
    archive_path = Path(path).expanduser()
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("local personal-memory archive is not readable")
    try:
        uri = f"{archive_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or str(integrity[0]).casefold() != "ok":
                raise ValueError("local personal-memory archive failed integrity check")
            table = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='archive_metadata'"
            ).fetchone()
            if table is None:
                raise ValueError("local personal-memory archive is not compatible")
    except (OSError, sqlite3.Error) as exc:
        raise ValueError("local personal-memory archive is not readable") from exc


def activate_canonical_profile(
    archive: PersonalMemoryArchive,
    *,
    backup_paths: tuple[Path, ...],
) -> None:
    """Activate canonical prompt memory only with readable recovery inputs."""
    if not isinstance(archive, PersonalMemoryArchive):
        raise ValueError("a local personal-memory archive is required")
    validate_local_personal_archive(archive.path)

    manifest = _read_manifest(archive)
    entries = manifest["entries"]
    expected_backups = tuple(Path(entry["backup_path"]) for entry in entries)
    resolved_backups = tuple(
        Path(path).expanduser().absolute() for path in backup_paths
    )
    if not resolved_backups:
        raise ValueError("at least one backup from the staged manifest is required")
    if resolved_backups != expected_backups:
        raise ValueError("profile backups must match the trusted staged manifest")
    for backup, entry in zip(resolved_backups, entries, strict=True):
        if backup.is_symlink() or not backup.is_file():
            raise ValueError("all profile backups must be readable files")
        try:
            snapshot_hash = hashlib.sha256(backup.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError("all profile backups must be readable files") from exc
        if snapshot_hash != entry["sha256"]:
            raise ValueError("profile backup snapshot does not match staged manifest")

    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    archive.mark_canonical_profile_active(
        manifest_sha256=hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    )


__all__ = [
    "LegacyProfileMigrator",
    "ProfileMigrationResult",
    "activate_canonical_profile",
    "validate_local_personal_archive",
]
