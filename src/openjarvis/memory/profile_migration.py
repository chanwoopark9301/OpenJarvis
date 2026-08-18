"""Reversible migration from static profile files to canonical memory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import uuid
from contextlib import ExitStack
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
_ACTIVE_IDENTITIES_KEY = "canonical_profile_active_backup_identities"


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
        if not _secure_snapshot_primitives_available():
            raise ValueError("secure profile snapshots are not supported")
        sources = tuple(
            Path(path).expanduser().absolute() for path in (user_path, memory_path)
        )
        backup_dir = self.archive.path.parent / ".canonical-profile-backups"
        snapshots: list[tuple[Path, Path, bytes]] = []
        with ExitStack() as backup_stack:
            backup_root_fd: int | None = None
            backup_root_identity: os.stat_result | None = None
            for source in sources:
                with ExitStack() as source_stack:
                    opened_source = _open_regular_source(source, source_stack)
                    if opened_source is None:
                        continue
                    source_fd, source_identity = opened_source
                    if backup_root_fd is None:
                        backup_root_fd, backup_root_identity = (
                            _open_private_backup_root(backup_dir, backup_stack)
                        )
                    assert backup_root_identity is not None
                    backup, snapshot = _copy_private_snapshot(
                        source,
                        source_fd,
                        source_identity,
                        backup_dir,
                        backup_root_fd,
                        backup_root_identity,
                    )
                    snapshot.decode("utf-8")
                    snapshots.append((source, backup, snapshot))

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


def _secure_snapshot_primitives_available() -> bool:
    """Return whether descriptor-only, no-follow snapshot operations exist."""
    return bool(
        os.name == "posix"
        and getattr(os, "O_DIRECTORY", 0)
        and getattr(os, "O_NOFOLLOW", 0)
        and hasattr(os, "fchmod")
        and os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
        and os.utime in os.supports_fd
    )


def _directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _close_fd(fd: int) -> None:
    """Close one descriptor without retrying an ambiguous numeric handle."""
    os.close(fd)


def _stack_fd(stack: ExitStack, fd: int) -> int:
    stack.callback(_close_fd, fd)
    return fd


def _force_mode(fd: int, mode: int) -> None:
    """Apply a private mode, retrying once before preserving the failure."""
    try:
        os.fchmod(fd, mode)
    except OSError:
        try:
            os.fchmod(fd, mode)
        except OSError:
            pass
        raise


def _open_directory_chain(path: Path, stack: ExitStack) -> int:
    """Open every absolute directory component without following symlinks."""
    absolute = path.absolute()
    current_fd = _stack_fd(stack, os.open(absolute.anchor, _directory_open_flags()))
    for component in absolute.parts[1:]:
        current_fd = _stack_fd(
            stack,
            os.open(component, _directory_open_flags(), dir_fd=current_fd),
        )
    return current_fd


def _open_regular_source(
    source: Path,
    stack: ExitStack,
) -> tuple[int, os.stat_result] | None:
    """Open one optional source once, rejecting symlinks and non-files."""
    try:
        parent_fd = _open_directory_chain(source.parent, stack)
        source_fd = _stack_fd(
            stack,
            os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd),
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError("legacy profile path is unsafe or contains a symlink") from exc
    source_identity = os.fstat(source_fd)
    if not stat.S_ISREG(source_identity.st_mode):
        raise ValueError("legacy profile path is not a regular file")
    return source_fd, source_identity


def _open_private_backup_root(
    backup_dir: Path,
    stack: ExitStack,
) -> tuple[int, os.stat_result]:
    """Open or atomically create the non-symlink private backup root."""
    parent_fd = _open_directory_chain(backup_dir.parent, stack)
    try:
        os.mkdir(backup_dir.name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    try:
        root_fd = _stack_fd(
            stack,
            os.open(backup_dir.name, _directory_open_flags(), dir_fd=parent_fd),
        )
    except OSError as exc:
        raise ValueError("profile backup directory is unsafe") from exc
    _force_mode(root_fd, 0o700)
    root_stat = os.fstat(root_fd)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("profile backup directory is unsafe")
    return root_fd, root_stat


def _copy_fd_bytes(source_fd: int, backup_fd: int) -> None:
    """Copy bytes between already-open descriptors without resolving paths."""
    while True:
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            return
        remaining = memoryview(chunk)
        while remaining:
            written = os.write(backup_fd, remaining)
            if written <= 0:
                raise OSError("profile snapshot write made no progress")
            remaining = remaining[written:]


def _verify_backup_path_identity(
    backup: Path,
    expected: os.stat_result,
) -> None:
    """Ensure the recovery path still resolves to the reserved inode."""
    try:
        with ExitStack() as stack:
            parent_fd = _open_directory_chain(backup.parent, stack)
            verification_fd = _stack_fd(
                stack,
                os.open(
                    backup.name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                ),
            )
            observed = os.fstat(verification_fd)
    except OSError as exc:
        raise ValueError("profile backup path identity changed") from exc
    if (observed.st_dev, observed.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ) or not stat.S_ISREG(observed.st_mode):
        raise ValueError("profile backup path identity changed")


def _copy_private_snapshot(
    source: Path,
    source_fd: int,
    source_identity: os.stat_result,
    backup_dir: Path,
    backup_root_fd: int,
    backup_root_identity: os.stat_result,
) -> tuple[Path, bytes]:
    """Copy an opened source into an exclusive destination descriptor."""
    snapshot_name = uuid.uuid4().hex
    try:
        os.mkdir(snapshot_name, mode=0o700, dir_fd=backup_root_fd)
    except FileExistsError as exc:
        raise ValueError("profile backup destination collision") from exc
    backup = (backup_dir / snapshot_name / source.name).absolute()
    with ExitStack() as stack:
        snapshot_fd = _stack_fd(
            stack,
            os.open(snapshot_name, _directory_open_flags(), dir_fd=backup_root_fd),
        )
        _force_mode(snapshot_fd, 0o700)
        file_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        backup_fd = _stack_fd(
            stack,
            os.open(source.name, file_flags, 0o600, dir_fd=snapshot_fd),
        )
        _force_mode(backup_fd, 0o600)
        backup_identity = os.fstat(backup_fd)
        if not stat.S_ISREG(backup_identity.st_mode):
            raise ValueError("profile backup destination is unsafe")
        try:
            _copy_fd_bytes(source_fd, backup_fd)
            os.utime(
                backup_fd,
                ns=(source_identity.st_atime_ns, source_identity.st_mtime_ns),
            )
            os.fsync(backup_fd)
            _verify_backup_path_identity(backup, backup_identity)
            root_observed = os.fstat(backup_root_fd)
            if (root_observed.st_dev, root_observed.st_ino) != (
                backup_root_identity.st_dev,
                backup_root_identity.st_ino,
            ):
                raise ValueError("profile backup root identity changed")
            os.lseek(backup_fd, 0, os.SEEK_SET)
            snapshot = bytearray()
            while chunk := os.read(backup_fd, 1024 * 1024):
                snapshot.extend(chunk)
        finally:
            # The file starts at 0600 and descriptor copying never broadens it.
            _force_mode(backup_fd, 0o600)
    return backup, bytes(snapshot)


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


def _manifest_json(manifest: dict) -> str:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def _manifest_digest(manifest_json: str) -> str:
    return hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()


def _private_identity(
    path: Path,
    observed: os.stat_result,
    *,
    expected_mode: int,
    directory: bool,
) -> dict[str, int | str]:
    expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_kind(observed.st_mode):
        raise ValueError("profile backup identity is not the expected file type")
    if stat.S_IMODE(observed.st_mode) != expected_mode:
        raise ValueError("profile backup permissions are not private")
    if observed.st_uid != os.geteuid():
        raise ValueError("profile backup ownership changed")
    return {
        "path": str(path),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "uid": int(observed.st_uid),
        "mode": int(stat.S_IMODE(observed.st_mode)),
    }


def _hash_open_file(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(fd, 1024 * 1024):
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _validate_staged_backups(
    archive: PersonalMemoryArchive,
    manifest: dict,
    backup_paths: tuple[Path, ...],
) -> tuple[tuple[Path, ...], str]:
    """Validate exact private snapshots without following a pathname twice."""
    if not _secure_snapshot_primitives_available():
        raise ValueError("secure profile backup validation is not supported")
    entries = manifest["entries"]
    expected_backups = tuple(Path(entry["backup_path"]) for entry in entries)
    resolved_backups = tuple(
        Path(path).expanduser().absolute() for path in backup_paths
    )
    if not resolved_backups:
        raise ValueError("at least one backup from the staged manifest is required")
    if resolved_backups != expected_backups:
        raise ValueError("profile backups must match the trusted staged manifest")

    backup_root = (
        archive.path.parent.expanduser().absolute() / ".canonical-profile-backups"
    )
    identities: dict[str, object]
    try:
        with ExitStack() as stack:
            root_fd = _open_directory_chain(backup_root, stack)
            root_identity = os.fstat(root_fd)
            identities = {
                "version": 1,
                "root": _private_identity(
                    backup_root,
                    root_identity,
                    expected_mode=0o700,
                    directory=True,
                ),
                "entries": [],
            }
            identity_entries: list[dict[str, object]] = []
            for backup, entry in zip(resolved_backups, entries, strict=True):
                if backup.parent.parent != backup_root:
                    raise ValueError(
                        "profile backup path is outside the private backup root"
                    )
                snapshot_fd = _stack_fd(
                    stack,
                    os.open(
                        backup.parent.name,
                        _directory_open_flags(),
                        dir_fd=root_fd,
                    ),
                )
                snapshot_identity = os.fstat(snapshot_fd)
                snapshot_record = _private_identity(
                    backup.parent,
                    snapshot_identity,
                    expected_mode=0o700,
                    directory=True,
                )
                backup_fd = _stack_fd(
                    stack,
                    os.open(
                        backup.name,
                        os.O_RDONLY | os.O_NOFOLLOW,
                        dir_fd=snapshot_fd,
                    ),
                )
                backup_identity = os.fstat(backup_fd)
                backup_record = _private_identity(
                    backup,
                    backup_identity,
                    expected_mode=0o600,
                    directory=False,
                )
                if _hash_open_file(backup_fd) != entry["sha256"]:
                    raise ValueError(
                        "profile backup snapshot does not match staged manifest"
                    )
                after_hash = os.fstat(backup_fd)
                if (after_hash.st_dev, after_hash.st_ino) != (
                    backup_identity.st_dev,
                    backup_identity.st_ino,
                ):
                    raise ValueError("profile backup identity changed")
                verify_snapshot_fd = _stack_fd(
                    stack,
                    os.open(
                        backup.parent.name,
                        _directory_open_flags(),
                        dir_fd=root_fd,
                    ),
                )
                verified_snapshot = os.fstat(verify_snapshot_fd)
                if (verified_snapshot.st_dev, verified_snapshot.st_ino) != (
                    snapshot_identity.st_dev,
                    snapshot_identity.st_ino,
                ):
                    raise ValueError("profile backup directory identity changed")
                verify_backup_fd = _stack_fd(
                    stack,
                    os.open(
                        backup.name,
                        os.O_RDONLY | os.O_NOFOLLOW,
                        dir_fd=verify_snapshot_fd,
                    ),
                )
                verified_backup = os.fstat(verify_backup_fd)
                if (verified_backup.st_dev, verified_backup.st_ino) != (
                    backup_identity.st_dev,
                    backup_identity.st_ino,
                ):
                    raise ValueError("profile backup identity changed")
                identity_entries.append(
                    {
                        "backup_path": str(backup),
                        "directory": snapshot_record,
                        "file": backup_record,
                    }
                )
            root_observed = os.fstat(root_fd)
            if (root_observed.st_dev, root_observed.st_ino) != (
                root_identity.st_dev,
                root_identity.st_ino,
            ):
                raise ValueError("profile backup root identity changed")
            identities["entries"] = identity_entries
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError("profile backup path is missing or unsafe") from exc
    return resolved_backups, json.dumps(
        identities,
        sort_keys=True,
        separators=(",", ":"),
    )


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
    _, backup_identities_json = _validate_staged_backups(
        archive,
        manifest,
        backup_paths,
    )
    manifest_json = _manifest_json(manifest)
    archive.mark_canonical_profile_active(
        manifest_json=manifest_json,
        manifest_sha256=_manifest_digest(manifest_json),
        backup_identities_json=backup_identities_json,
    )


def deactivate_canonical_profile(
    archive: PersonalMemoryArchive,
    *,
    backup_paths: tuple[Path, ...],
) -> None:
    """Re-enable current static profile sources without restoring backup bytes."""
    if not isinstance(archive, PersonalMemoryArchive):
        raise ValueError("a local personal-memory archive is required")
    validate_local_personal_archive(archive.path)
    manifest = _read_manifest(archive)
    _, backup_identities_json = _validate_staged_backups(
        archive,
        manifest,
        backup_paths,
    )
    active_identities = archive.get_metadata(_ACTIVE_IDENTITIES_KEY, "")
    if not active_identities or active_identities != backup_identities_json:
        raise ValueError("profile backup identity differs from activation attestation")
    manifest_json = _manifest_json(manifest)
    archive.mark_canonical_profile_inactive(
        manifest_json=manifest_json,
        manifest_sha256=_manifest_digest(manifest_json),
        backup_identities_json=backup_identities_json,
    )


__all__ = [
    "LegacyProfileMigrator",
    "ProfileMigrationResult",
    "activate_canonical_profile",
    "deactivate_canonical_profile",
    "validate_local_personal_archive",
]
