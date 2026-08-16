"""Run the model-free personal-memory acceptance benchmark."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.context_composer import ContextComposer


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * 0.95) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchanges", type=int, default=10_000)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="openjarvis-memory-benchmark-") as temp:
        path = Path(temp) / "personal.db"
        archive = PersonalMemoryArchive(path)
        for index in range(max(0, args.exchanges)):
            archive.record_exchange(
                exchange_id=f"exchange-{index}",
                user_text=f"synthetic user message {index}",
                assistant_text="synthetic assistant response",
                source="benchmark",
            )
        timings = []
        for _ in range(100):
            started = time.perf_counter()
            context = ContextComposer(archive).compose("benchmark query")
            timings.append((time.perf_counter() - started) * 1000)
        job_timings = []
        for index in range(100):
            job = archive.enqueue_job(
                job_type="benchmark",
                subject_id=f"subject-{index}",
                idempotency_key=f"benchmark:{index}",
            )
            started = time.perf_counter()
            assert archive.claim_job(job.id) is not None
            archive.complete_job(job.id)
            job_timings.append((time.perf_counter() - started) * 1000)
        checks = {
            "context composition p95 under 100 ms": _p95(timings) < 100,
            "job claim and completion p95 under 50 ms": _p95(job_timings) < 50,
            "fixed constraint cap": len(context.constraints) <= 5,
            "fixed schema cap": len(context.schemas) <= 8,
            "fixed episode cap": len(context.episodes) <= 5,
            "fixed raw evidence cap": len(context.raw_evidence) <= 3,
            "database integrity": archive.integrity_check() == "ok",
        }
        for name, passed in checks.items():
            print(f"{'PASS' if passed else 'FAIL'}: {name}")
        print(f"database bytes: {path.stat().st_size}")
        print(f"context p95 ms: {_p95(timings):.3f}")
        print(f"job p95 ms: {_p95(job_timings):.3f}")
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
