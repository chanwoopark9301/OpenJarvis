"""``jarvis memory`` — memory management subcommands."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
from rich.console import Console
from rich.progress import track
from rich.table import Table

from openjarvis.core.config import load_config
from openjarvis.core.registry import MemoryRegistry
from openjarvis.tools.storage.chunking import ChunkConfig
from openjarvis.tools.storage.ingest import ingest_path


def _get_backend(backend_key: str | None = None):
    """Instantiate the configured (or overridden) memory backend."""
    config = load_config()
    key = backend_key or config.memory.default_backend

    # Ensure backends are registered
    import openjarvis.tools.storage  # noqa: F401

    if not MemoryRegistry.contains(key):
        raise click.ClickException(
            f"Memory backend '{key}' not found. "
            f"Available: {', '.join(MemoryRegistry.keys())}"
        )

    if key == "sqlite":
        return MemoryRegistry.create(key, db_path=config.memory.db_path)
    return MemoryRegistry.create(key)


@click.group()
def memory() -> None:
    """Manage the memory store."""


@memory.command()
@click.argument("path")
@click.option(
    "--backend",
    "-b",
    default=None,
    help="Override the default memory backend.",
)
@click.option(
    "--chunk-size",
    default=512,
    type=int,
    help="Chunk size in tokens.",
)
@click.option(
    "--chunk-overlap",
    default=64,
    type=int,
    help="Overlap between chunks in tokens.",
)
def index(
    path: str,
    backend: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Index documents from a file or directory."""
    console = Console(stderr=True)
    target = Path(path)

    if not target.exists():
        console.print(f"[red]Path not found:[/red] {path}")
        raise SystemExit(1)

    t0 = time.time()
    cfg = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    console.print(f"[cyan]Indexing[/cyan] {path} ...")
    chunks = ingest_path(target, config=cfg)

    if not chunks:
        console.print("[yellow]No indexable content found.[/yellow]")
        return

    mem = _get_backend(backend)
    try:
        for chunk in track(chunks, description="Storing chunks...", console=console):
            mem.store(
                chunk.content,
                source=chunk.source,
                metadata={
                    "offset": chunk.offset,
                    "index": chunk.index,
                },
            )
    finally:
        if hasattr(mem, "close"):
            mem.close()

    elapsed = time.time() - t0
    sources = {c.source for c in chunks}
    console.print(
        f"[green]Indexed {len(chunks)} chunks "
        f"from {len(sources)} file(s) "
        f"in {elapsed:.1f}s.[/green]"
    )


@memory.command()
@click.argument("query", nargs=-1, required=True)
@click.option(
    "--top-k",
    "-k",
    default=5,
    type=int,
    help="Number of results to return.",
)
@click.option(
    "--backend",
    "-b",
    default=None,
    help="Override the default memory backend.",
)
def search(
    query: tuple[str, ...],
    top_k: int,
    backend: str | None,
) -> None:
    """Search the memory store."""
    console = Console()
    query_text = " ".join(query)

    mem = _get_backend(backend)
    try:
        results = mem.retrieve(query_text, top_k=top_k)
    finally:
        if hasattr(mem, "close"):
            mem.close()

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Search: {query_text}")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=8)
    table.add_column("Source", style="cyan")
    table.add_column("Content")

    for i, r in enumerate(results, 1):
        # Truncate content for display
        preview = r.content[:200]
        if len(r.content) > 200:
            preview += "..."
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            r.source or "-",
            preview,
        )

    console.print(table)


def _get_fact_store():
    """Instantiate the automatic-memory fact store from config."""
    from openjarvis.memory.store import create_fact_store

    config = load_config()
    mem = config.memory
    return create_fact_store(
        getattr(mem, "backend", "local"),
        path=getattr(mem, "facts_path", "~/.openjarvis/memory_facts.jsonl"),
        max_facts=getattr(mem, "max_facts", 1000),
    )


def _get_personal_inspector():
    """Open the configured local personal-memory archive for explicit controls."""
    from openjarvis.memory.archive import PersonalMemoryArchive
    from openjarvis.memory.inspector import PersonalMemoryInspector

    config = load_config()
    return PersonalMemoryInspector(
        PersonalMemoryArchive(config.personal_memory.archive_path)
    )


@memory.command(name="list")
def list_facts() -> None:
    """List durable facts captured by the automatic memory service."""
    console = Console()

    store = _get_fact_store()
    facts = store.list()
    if not facts:
        console.print("[yellow]No memory facts stored yet.[/yellow]")
        return

    table = Table(title=f"Memory Facts ({len(facts)})")
    table.add_column("#", style="dim", width=4)
    table.add_column("Fact")
    table.add_column("Source", style="cyan")
    for i, fact in enumerate(facts, 1):
        table.add_row(str(i), fact.text, fact.source or "-")
    console.print(table)


@memory.command()
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
def clear(yes: bool) -> None:
    """Remove all durable facts captured by the automatic memory service."""
    console = Console()

    store = _get_fact_store()
    count = store.count()
    if count == 0:
        console.print("[yellow]No memory facts to clear.[/yellow]")
        return

    if not yes:
        if not click.confirm(f"Remove all {count} stored memory fact(s)?"):
            console.print("[dim]Aborted.[/dim]")
            return

    removed = store.clear()
    console.print(f"[green]Cleared {removed} memory fact(s).[/green]")


@memory.command(name="personal-list")
def personal_list() -> None:
    """List developmental personal-memory subjects and their states."""
    console = Console()
    subjects = _get_personal_inspector().list_subjects()
    if not subjects:
        console.print("[yellow]No personal memories stored yet.[/yellow]")
        return
    table = Table(title=f"Personal Memory ({len(subjects)})")
    table.add_column("ID", style="dim")
    table.add_column("State", style="cyan")
    table.add_column("Content")
    for subject in subjects:
        table.add_row(subject.id, subject.state, subject.content)
    console.print(table)


@memory.command(name="personal-explain")
@click.argument("subject_id")
def personal_explain(subject_id: str) -> None:
    """Show one memory and only its directly linked evidence."""
    console = Console()
    subject = _get_personal_inspector().explain(subject_id)
    if subject is None:
        raise click.ClickException("Personal memory not found")
    console.print(f"[cyan]State:[/cyan] {subject.state}")
    console.print(f"[cyan]Content:[/cyan] {subject.content}")
    if subject.supporting_evidence:
        console.print("[cyan]Supporting evidence:[/cyan]")
        for evidence in subject.supporting_evidence:
            console.print(f"- {evidence}")


@memory.command(name="personal-suppress")
@click.argument("subject_id")
@click.option("--reason", default="user requested", show_default=True)
def personal_suppress(subject_id: str, reason: str) -> None:
    """Hide one personal memory from future response context."""
    if not _get_personal_inspector().suppress(subject_id, reason):
        raise click.ClickException("Personal memory not found")
    Console().print("[green]Personal memory suppressed.[/green]")


@memory.command(name="personal-restore")
@click.argument("subject_id")
def personal_restore(subject_id: str) -> None:
    """Restore one suppressed personal memory."""
    if not _get_personal_inspector().restore(subject_id):
        raise click.ClickException("Suppressed personal memory not found")
    Console().print("[green]Personal memory restored.[/green]")


@memory.command(name="personal-correct")
@click.argument("subject_id")
@click.argument("replacement_text")
@click.option("--user-text", required=True)
def personal_correct(
    subject_id: str,
    replacement_text: str,
    user_text: str,
) -> None:
    """Supersede one memory with a user-confirmed correction."""
    result = _get_personal_inspector().correct(
        subject_id,
        replacement_text,
        user_text,
    )
    if result is None:
        raise click.ClickException("Personal memory could not be corrected")
    Console().print(f"[green]Created corrected memory {result.id}.[/green]")


@memory.command(name="personal-delete")
@click.argument("subject_id")
@click.option("--include-raw-evidence", is_flag=True, default=False)
def personal_delete(subject_id: str, include_raw_evidence: bool) -> None:
    """Delete one memory, preserving raw evidence unless explicitly requested."""
    try:
        removed = _get_personal_inspector().delete_subject(
            subject_id,
            include_raw_evidence=include_raw_evidence,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not any(removed.values()):
        raise click.ClickException("Personal memory not found")
    Console().print("[green]Personal memory deleted.[/green]")


@memory.command(name="personal-delete-all")
@click.option("--confirm-token", required=True)
def personal_delete_all(confirm_token: str) -> None:
    """Delete all personal memory after exact-token confirmation."""
    inspector = _get_personal_inspector()
    preview = inspector.deletion_preview()
    Console().print(f"Rows scheduled for deletion: {sum(preview.values())}")
    try:
        removed = inspector.delete_all_personal_memory(confirm_token)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    Console().print(f"[green]Deleted {sum(removed.values())} rows.[/green]")


@memory.command(name="personal-import-legacy")
@click.option("--path", "legacy_path", default=None)
@click.option("--backup/--no-backup", default=True)
def personal_import_legacy(legacy_path: str | None, backup: bool) -> None:
    """Stage legacy facts as low-trust candidates, with a backup by default."""
    from openjarvis.memory.legacy_import import LegacyFactImporter

    config = load_config()
    source = Path(legacy_path or config.memory.facts_path).expanduser()
    importer = LegacyFactImporter(_get_personal_inspector().archive)
    backup_path = importer.backup(source) if backup else None
    result = importer.run(source)
    console = Console()
    if backup_path is not None:
        console.print(f"[green]Backup:[/green] {backup_path}")
    console.print(
        f"[green]Imported {result.imported}; skipped {result.skipped}.[/green]"
    )


@memory.command(name="personal-rollout-status")
@click.option("--backup-path", required=True)
@click.option("--manual-override", is_flag=True, default=False)
@click.option("--activate", is_flag=True, default=False)
def personal_rollout_status(
    backup_path: str,
    manual_override: bool,
    activate: bool,
) -> None:
    """Check whether legacy prompt injection may be disabled safely."""
    from openjarvis.memory.legacy_import import (
        activate_rollout,
        evaluate_rollout_readiness,
    )

    inspector = _get_personal_inspector()
    config = load_config()
    check = activate_rollout if activate else evaluate_rollout_readiness
    result = check(
        inspector.archive,
        backup_path=backup_path,
        manual_override=manual_override,
        worker_concurrency=config.personal_memory.worker_concurrency,
    )
    if not result.ready:
        failures = ", ".join(result.failing_gates)
        raise click.ClickException(f"Rollout gates failed: {failures}")
    status = "activated" if activate else "passed"
    Console().print(f"[green]Rollout gates {status}.[/green]")


@memory.command(name="stage-canonical-profile")
def stage_canonical_profile() -> None:
    """Back up and stage legacy profile bullets without activating them."""
    from openjarvis.memory.archive import PersonalMemoryArchive
    from openjarvis.memory.profile_migration import LegacyProfileMigrator

    config = load_config()
    archive = PersonalMemoryArchive(config.personal_memory.archive_path)
    try:
        result = LegacyProfileMigrator(archive).stage(
            Path(config.memory_files.user_path),
            Path(config.memory_files.memory_path),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    Console().print(
        "[green]Canonical profile staged:[/green] "
        f"{len(result.backup_paths)} backup(s), "
        f"{result.imported} candidate(s)."
    )


@memory.command(name="activate-canonical-profile")
@click.option(
    "--backup-path",
    "backup_paths",
    multiple=True,
    type=click.Path(path_type=Path),
    help="Readable recovery backup; may be repeated.",
)
def activate_canonical_profile_command(backup_paths: tuple[Path, ...]) -> None:
    """Stop static USER/MEMORY injection after explicit backup validation."""
    from openjarvis.memory.archive import PersonalMemoryArchive
    from openjarvis.memory.profile_migration import (
        activate_canonical_profile,
        validate_local_personal_archive,
    )

    config = load_config()
    try:
        validate_local_personal_archive(config.personal_memory.archive_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    archive = PersonalMemoryArchive(config.personal_memory.archive_path)
    selected_backups = backup_paths
    if not selected_backups:
        try:
            manifest = json.loads(
                archive.get_metadata("canonical_profile_staging_manifest", "{}")
            )
            entries = manifest.get("entries", []) if isinstance(manifest, dict) else []
            stored = [entry["backup_path"] for entry in entries]
        except (TypeError, ValueError, json.JSONDecodeError):
            stored = []
        if isinstance(stored, list):
            selected_backups = tuple(Path(str(path)) for path in stored)
    try:
        activate_canonical_profile(archive, backup_paths=selected_backups)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    Console().print("[green]Canonical profile activated.[/green]")


@memory.command(name="deactivate-canonical-profile")
@click.option(
    "--backup-path",
    "backup_paths",
    multiple=True,
    type=click.Path(path_type=Path),
    help="Attested recovery backup; may be repeated.",
)
def deactivate_canonical_profile_command(backup_paths: tuple[Path, ...]) -> None:
    """Re-enable current USER/MEMORY files after backup revalidation."""
    from openjarvis.memory.archive import PersonalMemoryArchive
    from openjarvis.memory.profile_migration import (
        deactivate_canonical_profile,
        validate_local_personal_archive,
    )

    config = load_config()
    try:
        validate_local_personal_archive(config.personal_memory.archive_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    archive = PersonalMemoryArchive(config.personal_memory.archive_path)
    selected_backups = backup_paths
    if not selected_backups:
        try:
            manifest = json.loads(
                archive.get_metadata("canonical_profile_staging_manifest", "{}")
            )
            entries = manifest.get("entries", []) if isinstance(manifest, dict) else []
            stored = [entry["backup_path"] for entry in entries]
        except (TypeError, ValueError, json.JSONDecodeError):
            stored = []
        if isinstance(stored, list):
            selected_backups = tuple(Path(str(path)) for path in stored)
    try:
        deactivate_canonical_profile(archive, backup_paths=selected_backups)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    Console().print(
        "[green]Canonical profile deactivated; current static profile files "
        "were not changed.[/green]"
    )


@memory.command()
@click.option(
    "--backend",
    "-b",
    default=None,
    help="Override the default memory backend.",
)
def stats(backend: str | None) -> None:
    """Show memory store statistics."""
    console = Console()

    mem = _get_backend(backend)
    try:
        count = 0
        if hasattr(mem, "count"):
            count = mem.count()

        table = Table(title="Memory Statistics")
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        table.add_row("Backend", mem.backend_id)
        table.add_row("Documents", str(count))

        if hasattr(mem, "_db_path"):
            db_path = Path(mem._db_path)
            if db_path.exists():
                size_kb = db_path.stat().st_size / 1024
                table.add_row(
                    "Database size",
                    f"{size_kb:.1f} KB",
                )
            table.add_row("Database path", str(db_path))

        console.print(table)
    finally:
        if hasattr(mem, "close"):
            mem.close()
