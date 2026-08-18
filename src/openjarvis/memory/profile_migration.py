"""Reversible migration from static profile files to canonical memory."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
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
_ACTIVE_METADATA_KEY = "canonical_profile_active"


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
        """Copy existing files before importing any of their bullet lines."""
        sources = tuple(Path(path).expanduser() for path in (user_path, memory_path))
        backup_paths: list[Path] = []
        existing_sources: list[Path] = []
        run_id = time.time_ns()
        for source in sources:
            if not source.exists():
                continue
            if not source.is_file():
                raise ValueError(f"legacy profile path is not a file: {source}")
            backup = source.with_name(f"{source.name}.canonical-backup-{run_id}")
            shutil.copy2(source, backup)
            backup_paths.append(backup)
            existing_sources.append(source)

        imported = 0
        skipped = 0
        for source in existing_sources:
            for line_number, raw_line in enumerate(
                source.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                match = _BULLET.match(raw_line)
                content = match.group(1).strip() if match is not None else ""
                if not content:
                    continue
                digest_input = f"{source}:{line_number}:{content.casefold()}"
                digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
                exchange_id = f"legacy-profile-{digest}"
                self.archive.record_exchange(
                    exchange_id=exchange_id,
                    user_text="",
                    assistant_text="",
                    source=f"legacy-profile:{source}:{line_number}",
                )
                if self.archive.claim_candidate_job(exchange_id) is None:
                    skipped += 1
                    continue
                self.archive.complete_candidate_job(
                    exchange_id,
                    [
                        CandidateDraft(
                            CandidateKind.FACT,
                            content,
                            0.1,
                            0.1,
                            source=EvidenceSource.LEGACY_IMPORT,
                            temporal_scope="unspecified",
                            subject="legacy_profile",
                        )
                    ],
                    engine_id="legacy-profile-stage",
                    extractor_version="legacy-profile-v1",
                )
                imported += 1

        self.archive.set_metadata(
            _BACKUP_METADATA_KEY,
            json.dumps([str(path) for path in backup_paths], separators=(",", ":")),
        )
        return ProfileMigrationResult(
            backup_paths=tuple(backup_paths),
            imported=imported,
            skipped=skipped,
        )


def activate_canonical_profile(
    archive: PersonalMemoryArchive,
    *,
    backup_paths: tuple[Path, ...],
) -> None:
    """Activate canonical prompt memory only with readable recovery inputs."""
    if not isinstance(archive, PersonalMemoryArchive):
        raise ValueError("a local personal-memory archive is required")
    archive_path = archive.path.expanduser()
    if not archive_path.is_file():
        raise ValueError("local personal-memory archive is not readable")
    try:
        with archive_path.open("rb") as handle:
            handle.read(1)
        if archive.integrity_check() != "ok":
            raise ValueError("local personal-memory archive failed integrity check")
    except OSError as exc:
        raise ValueError("local personal-memory archive is not readable") from exc

    resolved_backups = tuple(Path(path).expanduser() for path in backup_paths)
    if not resolved_backups:
        raise ValueError("at least one readable profile backup is required")
    for backup in resolved_backups:
        if not backup.is_file():
            raise ValueError("all profile backups must be readable files")
        try:
            with backup.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            raise ValueError("all profile backups must be readable files") from exc

    archive.set_metadata(
        _BACKUP_METADATA_KEY,
        json.dumps([str(path) for path in resolved_backups], separators=(",", ":")),
    )
    archive.set_metadata(_ACTIVE_METADATA_KEY, "1")


__all__ = [
    "LegacyProfileMigrator",
    "ProfileMigrationResult",
    "activate_canonical_profile",
]
