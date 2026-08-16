"""Run and persist the named personal-memory release checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from openjarvis.memory.archive import PersonalMemoryArchive


def _run(repo: Path, *args: str) -> bool:
    result = subprocess.run(
        [sys.executable, *args],
        cwd=repo,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-path", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    checks = {
        "restart_and_direct_rule": (
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/memory/test_developmental_memory_acceptance.py::"
            "test_direct_rules_and_current_state_survive_restart",
            "tests/memory/test_developmental_memory_acceptance.py::"
            "test_conflicting_old_direct_rule_is_absent_immediately_after_restart",
        ),
        "assistant_only_rejection": (
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/memory/test_developmental_memory_acceptance.py::"
            "test_assistant_text_never_becomes_personal_evidence",
        ),
        "chat_priority_and_concurrency": (
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/memory/test_personal_memory_service.py::"
            "test_chat_busy_defers_reflection_without_running_the_extractor",
            "tests/telemetry/test_instrumented_engine.py::"
            "TestInstrumentedEngine::test_background_and_chat_generation_never_overlap",
        ),
        "replay_idempotency": (
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/memory/test_evaluator.py::"
            "test_evaluator_applies_direct_rule_and_audits_once",
        ),
        "external_isolation": (
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/memory/test_social_reflection.py",
        ),
        "provider_privacy_interception": (
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/memory/test_social_reflection.py::"
            "test_provider_receives_only_a_deidentified_general_query",
        ),
        "user_controls": (
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/memory/test_memory_inspector.py",
        ),
        "benchmark_10000": (
            "scripts/benchmark_personal_memory.py",
            "--exchanges",
            "10000",
        ),
    }
    results = {gate: _run(repo, *command) for gate, command in checks.items()}
    archive = PersonalMemoryArchive(args.archive_path)
    for gate, passed in results.items():
        archive.record_release_gate_attestation(gate, passed=passed)
        print(f"{'PASS' if passed else 'FAIL'}: {gate}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
