"""Fixed-budget performance checks that exclude local model calls."""

from __future__ import annotations

import time

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.context_composer import ContextComposer


def test_context_composition_stays_fixed_with_large_archive(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    for index in range(1000):
        archive.record_exchange(
            exchange_id=f"exchange-{index}",
            user_text=f"ordinary message {index}",
            assistant_text="ordinary reply",
            source="performance",
        )

    durations = []
    for _ in range(30):
        started = time.perf_counter()
        context = ContextComposer(archive).compose("ordinary question")
        durations.append((time.perf_counter() - started) * 1000)

    durations.sort()
    p95 = durations[int(len(durations) * 0.95) - 1]
    assert p95 < 100
    assert len(context.constraints) <= 5
    assert len(context.schemas) <= 8
    assert len(context.episodes) <= 5
    assert len(context.raw_evidence) <= 3
