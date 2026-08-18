"""Tests for ``jarvis memory`` CLI commands."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from openjarvis.cli import cli
from openjarvis.core.registry import MemoryRegistry
from openjarvis.memory.store import LocalFactStore
from openjarvis.tools.storage.sqlite import SQLiteMemory


def _register_sqlite():
    """Re-register sqlite backend (conftest clears registries)."""
    if not MemoryRegistry.contains("sqlite"):
        MemoryRegistry.register_value("sqlite", SQLiteMemory)


def test_memory_index_file(tmp_path: Path, monkeypatch):
    """Index a single text file and check success message."""
    _register_sqlite()
    db_path = str(tmp_path / "mem.db")

    # Create a text file with enough content
    doc = tmp_path / "doc.txt"
    doc.write_text(" ".join(f"word{i}" for i in range(100)))

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda b=None: SQLiteMemory(db_path=db_path),
    )

    result = CliRunner().invoke(cli, ["memory", "index", str(doc)])
    assert result.exit_code == 0
    assert "Indexed" in result.output or "chunk" in result.output


def test_memory_index_nonexistent(tmp_path: Path):
    """Indexing a nonexistent path should fail."""
    _register_sqlite()
    result = CliRunner().invoke(cli, ["memory", "index", str(tmp_path / "nope")])
    assert result.exit_code != 0


def test_memory_search_returns_results(tmp_path: Path, monkeypatch):
    """Search returns results from pre-populated backend."""
    _register_sqlite()
    db_path = str(tmp_path / "mem.db")
    backend = SQLiteMemory(db_path=db_path)
    backend.store(
        "Python programming language guide",
        source="guide.md",
    )

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda b=None: SQLiteMemory(db_path=db_path),
    )

    result = CliRunner().invoke(cli, ["memory", "search", "Python"])
    assert result.exit_code == 0
    assert "Python" in result.output
    backend.close()


def test_memory_search_no_results(tmp_path: Path, monkeypatch):
    """Search with no matches shows appropriate message."""
    _register_sqlite()
    db_path = str(tmp_path / "mem.db")
    backend = SQLiteMemory(db_path=db_path)
    backend.store("some unrelated content about cats")

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda b=None: SQLiteMemory(db_path=db_path),
    )

    result = CliRunner().invoke(cli, ["memory", "search", "quantum supercollider"])
    assert result.exit_code == 0
    assert "No results" in result.output
    backend.close()


def test_memory_stats_shows_count(tmp_path: Path, monkeypatch):
    """Stats command shows document count."""
    _register_sqlite()
    db_path = str(tmp_path / "mem.db")
    backend = SQLiteMemory(db_path=db_path)
    backend.store("doc one")
    backend.store("doc two")

    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(
        mod,
        "_get_backend",
        lambda b=None: SQLiteMemory(db_path=db_path),
    )

    result = CliRunner().invoke(cli, ["memory", "stats"])
    assert result.exit_code == 0
    assert "2" in result.output
    backend.close()


def _patch_fact_store(monkeypatch, tmp_path: Path) -> LocalFactStore:
    """Point ``jarvis memory list/clear`` at a temp fact store."""
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    store = LocalFactStore(tmp_path / "facts.jsonl")
    monkeypatch.setattr(mod, "_get_fact_store", lambda: store)
    return store


def test_memory_list_empty(tmp_path: Path, monkeypatch):
    _patch_fact_store(monkeypatch, tmp_path)
    result = CliRunner().invoke(cli, ["memory", "list"])
    assert result.exit_code == 0
    assert "No memory facts" in result.output


def test_memory_list_shows_facts(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("User prefers dark mode")
    store.add("User lives in Berlin")

    result = CliRunner().invoke(cli, ["memory", "list"])
    assert result.exit_code == 0
    assert "dark mode" in result.output
    assert "Berlin" in result.output


def test_memory_clear_with_confirmation(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("fact one")
    store.add("fact two")

    result = CliRunner().invoke(cli, ["memory", "clear"], input="y\n")
    assert result.exit_code == 0
    assert "Cleared 2" in result.output
    assert store.count() == 0


def test_memory_clear_aborted(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("keep me")

    result = CliRunner().invoke(cli, ["memory", "clear"], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert store.count() == 1


def test_memory_clear_yes_flag(tmp_path: Path, monkeypatch):
    store = _patch_fact_store(monkeypatch, tmp_path)
    store.add("fact")

    result = CliRunner().invoke(cli, ["memory", "clear", "--yes"])
    assert result.exit_code == 0
    assert "Cleared 1" in result.output
    assert store.count() == 0


def test_memory_clear_empty(tmp_path: Path, monkeypatch):
    _patch_fact_store(monkeypatch, tmp_path)
    result = CliRunner().invoke(cli, ["memory", "clear"])
    assert result.exit_code == 0
    assert "No memory facts to clear" in result.output


def test_stage_canonical_profile_cli_keeps_personal_text_out_of_output(
    tmp_path: Path,
    monkeypatch,
):
    from openjarvis.core.config import JarvisConfig
    from openjarvis.memory.archive import PersonalMemoryArchive

    config = JarvisConfig()
    config.personal_memory.archive_path = str(tmp_path / "personal.db")
    config.memory_files.user_path = str(tmp_path / "USER.md")
    config.memory_files.memory_path = str(tmp_path / "MEMORY.md")
    Path(config.memory_files.user_path).write_text(
        "- private profile marker",
        encoding="utf-8",
    )
    Path(config.memory_files.memory_path).write_text(
        "- private memory marker",
        encoding="utf-8",
    )
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(mod, "load_config", lambda: config)

    result = CliRunner().invoke(cli, ["memory", "stage-canonical-profile"])

    assert result.exit_code == 0
    assert "private profile marker" not in result.output
    assert "private memory marker" not in result.output
    archive = PersonalMemoryArchive(config.personal_memory.archive_path)
    assert archive.get_metadata("canonical_profile_backup_paths", "")
    assert archive.get_metadata("canonical_profile_active", "0") == "0"


def test_activate_canonical_profile_cli_uses_staged_backups(tmp_path, monkeypatch):
    from openjarvis.core.config import JarvisConfig
    from openjarvis.memory.archive import PersonalMemoryArchive

    config = JarvisConfig()
    config.personal_memory.archive_path = str(tmp_path / "personal.db")
    config.memory_files.user_path = str(tmp_path / "USER.md")
    config.memory_files.memory_path = str(tmp_path / "MEMORY.md")
    Path(config.memory_files.user_path).write_text("- private user", encoding="utf-8")
    Path(config.memory_files.memory_path).write_text(
        "- private memory",
        encoding="utf-8",
    )
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(mod, "load_config", lambda: config)
    runner = CliRunner()
    assert runner.invoke(cli, ["memory", "stage-canonical-profile"]).exit_code == 0

    result = runner.invoke(cli, ["memory", "activate-canonical-profile"])

    assert result.exit_code == 0
    assert "private user" not in result.output
    assert "private memory" not in result.output
    archive = PersonalMemoryArchive(config.personal_memory.archive_path)
    assert archive.get_metadata("canonical_profile_active", "0") == "1"


def test_deactivate_canonical_profile_cli_preserves_live_profile_files(
    tmp_path,
    monkeypatch,
):
    from openjarvis.core.config import JarvisConfig
    from openjarvis.memory.archive import PersonalMemoryArchive

    config = JarvisConfig()
    config.personal_memory.archive_path = str(tmp_path / "personal.db")
    config.memory_files.user_path = str(tmp_path / "USER.md")
    config.memory_files.memory_path = str(tmp_path / "MEMORY.md")
    user_path = Path(config.memory_files.user_path)
    memory_path = Path(config.memory_files.memory_path)
    user_path.write_text("- private original user", encoding="utf-8")
    memory_path.write_text("- private original memory", encoding="utf-8")
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(mod, "load_config", lambda: config)
    runner = CliRunner()
    assert runner.invoke(cli, ["memory", "stage-canonical-profile"]).exit_code == 0
    assert runner.invoke(cli, ["memory", "activate-canonical-profile"]).exit_code == 0
    user_path.write_text("CURRENT PRIVATE USER", encoding="utf-8")
    memory_path.write_text("CURRENT PRIVATE MEMORY", encoding="utf-8")

    result = runner.invoke(cli, ["memory", "deactivate-canonical-profile"])

    assert result.exit_code == 0
    assert "CURRENT PRIVATE USER" not in result.output
    assert "CURRENT PRIVATE MEMORY" not in result.output
    assert user_path.read_text(encoding="utf-8") == "CURRENT PRIVATE USER"
    assert memory_path.read_text(encoding="utf-8") == "CURRENT PRIVATE MEMORY"
    archive = PersonalMemoryArchive(config.personal_memory.archive_path)
    assert archive.get_metadata("canonical_profile_active", "1") == "0"


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
_PRIVATE_MANIFEST_MARKER = "private-cli-manifest-marker"


def _corrupt_staging_manifest(archive, corruption: str) -> None:
    manifest = json.loads(archive.get_metadata("canonical_profile_staging_manifest"))
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
def test_canonical_profile_cli_rejects_corrupt_staging_manifest(
    tmp_path,
    monkeypatch,
    operation,
    corruption,
):
    """The CLI must translate trusted-metadata errors without content leaks."""
    from openjarvis.core.config import JarvisConfig
    from openjarvis.memory.archive import PersonalMemoryArchive

    config = JarvisConfig()
    config.personal_memory.archive_path = str(tmp_path / "personal.db")
    config.memory_files.user_path = str(tmp_path / "USER.md")
    config.memory_files.memory_path = str(tmp_path / "MEMORY.md")
    user_path = Path(config.memory_files.user_path)
    memory_path = Path(config.memory_files.memory_path)
    user_path.write_text("- private original user", encoding="utf-8")
    memory_path.write_text("- private original memory", encoding="utf-8")
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(mod, "load_config", lambda: config)
    runner = CliRunner()
    assert runner.invoke(cli, ["memory", "stage-canonical-profile"]).exit_code == 0
    archive = PersonalMemoryArchive(config.personal_memory.archive_path)
    if operation == "deactivate":
        assert (
            runner.invoke(cli, ["memory", "activate-canonical-profile"]).exit_code == 0
        )
    user_path.write_text("CURRENT PRIVATE USER", encoding="utf-8")
    memory_path.write_text("CURRENT PRIVATE MEMORY", encoding="utf-8")
    active_before = archive.get_metadata("canonical_profile_active", "0")
    decisions_before = archive.decision_count(subject_id="canonical_profile")
    backup_path = json.loads(
        archive.get_metadata("canonical_profile_staging_manifest")
    )["entries"][0]["backup_path"]
    _corrupt_staging_manifest(archive, corruption)

    result = runner.invoke(
        cli,
        ["memory", f"{operation}-canonical-profile"],
    )

    assert result.exit_code == 1
    expected_error = (
        "trusted staged manifest is unreadable"
        if corruption == "pathologically_nested_json"
        else "trusted staged manifest is invalid"
    )
    assert f"Error: {expected_error}" in result.output
    assert _PRIVATE_MANIFEST_MARKER not in result.output
    assert backup_path not in result.output
    assert "CURRENT PRIVATE USER" not in result.output
    assert "CURRENT PRIVATE MEMORY" not in result.output
    assert archive.get_metadata("canonical_profile_active", "0") == active_before
    assert archive.get_metadata("canonical_profile_deactivated_at", "") == ""
    assert archive.decision_count(subject_id="canonical_profile") == decisions_before
    assert user_path.read_text(encoding="utf-8") == "CURRENT PRIVATE USER"
    assert memory_path.read_text(encoding="utf-8") == "CURRENT PRIVATE MEMORY"


def test_activate_canonical_profile_cli_refuses_without_staging(
    tmp_path,
    monkeypatch,
):
    from openjarvis.core.config import JarvisConfig
    from openjarvis.memory.archive import PersonalMemoryArchive

    config = JarvisConfig()
    config.personal_memory.archive_path = str(tmp_path / "personal.db")
    PersonalMemoryArchive(config.personal_memory.archive_path)
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(mod, "load_config", lambda: config)

    result = CliRunner().invoke(cli, ["memory", "activate-canonical-profile"])

    assert result.exit_code != 0
    assert "backup" in result.output.lower()


def test_activate_canonical_profile_cli_missing_archive_does_not_create_it(
    tmp_path,
    monkeypatch,
):
    from openjarvis.core.config import JarvisConfig

    config = JarvisConfig()
    archive_path = tmp_path / "missing.db"
    config.personal_memory.archive_path = str(archive_path)
    mod = importlib.import_module("openjarvis.cli.memory_cmd")
    monkeypatch.setattr(mod, "load_config", lambda: config)

    result = CliRunner().invoke(cli, ["memory", "activate-canonical-profile"])

    assert result.exit_code != 0
    assert "archive" in result.output.lower()
    assert not archive_path.exists()
