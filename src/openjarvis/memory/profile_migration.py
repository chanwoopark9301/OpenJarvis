"""Reversible migration from static profile files to canonical memory."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
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
        if backup_dir.is_symlink():
            raise ValueError("profile backup directory must not be a symlink")
        backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        backup_dir.chmod(0o700)
        snapshots: list[tuple[Path, Path, bytes]] = []
        for source in sources:
            if not source.exists():
                continue
            if source.is_symlink():
                raise ValueError(f"legacy profile path must not be a symlink: {source}")
            if not source.is_file():
                raise ValueError(f"legacy profile path is not a file: {source}")
            backup = backup_dir / f"{source.name}.{uuid.uuid4().hex}.backup"
            if backup.exists() or backup.is_symlink():
                raise ValueError("profile backup destination collision")
            shutil.copy2(source, backup)
            if backup.is_symlink() or not backup.is_file():
                raise ValueError("profile backup destination is unsafe")
            backup.chmod(0o600)
            snapshot = backup.read_bytes()
            snapshot.decode("utf-8")
            snapshots.append((source, backup.absolute(), snapshot))

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
