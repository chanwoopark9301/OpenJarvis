"""Tests for ``jarvis memory`` CLI commands."""

from __future__ import annotations

import importlib
from pathlib import Path

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
