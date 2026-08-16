"""Conservative import of legacy JSONL facts as untrusted candidates."""

from __future__ import annotations

import hashlib
import json
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


@dataclass(frozen=True, slots=True)
class LegacyImportResult:
    """Aggregate result that never includes personal text."""

    imported: int
    skipped: int


@dataclass(frozen=True, slots=True)
class RolloutReadiness:
    """Named gates that must pass before legacy prompt injection is disabled."""

    ready: bool
    failing_gates: tuple[str, ...]


class LegacyFactImporter:
    """Stage each distinct legacy fact once without inventing user evidence."""

    def __init__(self, archive: PersonalMemoryArchive) -> None:
        self.archive = archive

    def run(self, path: str | Path) -> LegacyImportResult:
        """Import valid lines as pending low-trust candidates."""
        source_path = Path(path).expanduser()
        imported = 0
        skipped = 0
        if not source_path.exists():
            return LegacyImportResult(imported=0, skipped=0)
        for line_number, raw_line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                payload = json.loads(raw_line)
            except (json.JSONDecodeError, TypeError, ValueError):
                skipped += 1
                continue
            content = str(payload.get("text") or payload.get("content") or "").strip()
            if not content:
                skipped += 1
                continue
            digest = hashlib.sha256(content.casefold().encode("utf-8")).hexdigest()
            exchange_id = f"legacy-fact-{digest}"
            self.archive.record_exchange(
                exchange_id=exchange_id,
                user_text="",
                assistant_text="",
                source=f"legacy:{source_path}:{line_number}",
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
                        0.25,
                        0.25,
                        source=EvidenceSource.LEGACY_IMPORT,
                        temporal_scope="unspecified",
                        subject="legacy_fact",
                    )
                ],
                engine_id="legacy-import",
                extractor_version="legacy-jsonl-v1",
            )
            imported += 1
        return LegacyImportResult(imported=imported, skipped=skipped)

    @staticmethod
    def backup(path: str | Path, destination: str | Path | None = None) -> Path:
        """Create a recoverable copy before any legacy-memory transition."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = (
            Path(destination).expanduser()
            if destination is not None
            else source.with_name(f"{source.name}.backup-{int(time.time())}")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target


def evaluate_rollout_readiness(
    archive: PersonalMemoryArchive,
    *,
    backup_path: str | Path | None,
    manual_override: bool = False,
    now: float | None = None,
) -> RolloutReadiness:
    """Require a backup and seven shadow days unless manually overridden."""
    failures = []
    backup = Path(backup_path).expanduser() if backup_path else None
    if backup is None or not backup.is_file():
        failures.append("legacy_backup_missing")
    started_raw = archive.get_metadata("shadow_started_at", "0")
    try:
        started_at = float(started_raw)
    except ValueError:
        started_at = 0.0
    elapsed = (time.time() if now is None else now) - started_at
    if not manual_override and (started_at <= 0 or elapsed < 7 * 24 * 60 * 60):
        failures.append("seven_shadow_days_incomplete")
    return RolloutReadiness(not failures, tuple(failures))


__all__ = [
    "LegacyFactImporter",
    "LegacyImportResult",
    "RolloutReadiness",
    "evaluate_rollout_readiness",
]
