"""Integration tests for one-shot schema consolidation jobs."""

from __future__ import annotations

import time

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.developmental_pipeline import DevelopmentalMemoryPipeline
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import (
    AdaptationOperation,
    CandidateDraft,
    CandidateKind,
)
from openjarvis.memory.personal_service import PersonalMemoryService
from openjarvis.memory.reflection import ReflectionProposal


class _Extractor:
    engine_id = "ollama"
    last_error_code = ""

    def extract(self, exchange):
        return [
            CandidateDraft(
                CandidateKind.FACT,
                exchange.user_text,
                0.8,
                0.9,
                subject="running_and_mood",
            )
        ]


class _Reflection:
    def reflect(self, *, evidence, conflict_evidence_ids):
        assert conflict_evidence_ids == set()
        return ReflectionProposal(
            content="Running often improves mood.",
            operation=AdaptationOperation.ACCOMMODATE_CREATE,
            scope="running_and_mood",
            support_evidence_ids=tuple(evidence),
            counter_evidence_ids=(),
            uncertainties=(),
            requires_user_confirmation=False,
        )


def test_three_sessions_flow_through_separate_jobs_into_emerging_schema(tmp_path):
    archive = PersonalMemoryArchive(tmp_path / "personal.db")
    pipeline = DevelopmentalMemoryPipeline(archive, _Reflection())
    service = PersonalMemoryService(
        archive,
        _Extractor(),
        evaluator=MemoryEvaluator(archive),
        job_handlers=pipeline.handlers(),
        idle_before_reflection_seconds=0,
    )
    service.start()
    try:
        for index in range(3):
            exchange = service.archive_exchange(
                user_text=f"Running improved my mood on day {index}.",
                assistant_text="",
                source="test",
                session_id=f"session-{index}",
            )
            assert service.enqueue_exchange(exchange.id)
        deadline = time.time() + 3
        while time.time() < deadline and archive.active_schema_count() == 0:
            time.sleep(0.01)

        assert archive.active_schema_count() == 1
        assert archive.get_active_schemas() == []
    finally:
        service.stop()
