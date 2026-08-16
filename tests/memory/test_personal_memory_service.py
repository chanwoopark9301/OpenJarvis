"""Tests for durable local personal-memory background processing."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from openjarvis.core.config import JarvisConfig, PersonalMemoryConfig
from openjarvis.core.events import EventBus, EventType


class _FakeExtractor:
    def __init__(self, drafts=None, gate=None):
        self.drafts = drafts or []
        self.gate = gate
        self.started = threading.Event()

    def extract(self, exchange):
        self.started.set()
        if self.gate is not None:
            self.gate.wait(timeout=2.0)
        return list(self.drafts)


class _InvalidOutputExtractor:
    """A local extractor that reports malformed output without exposing text."""

    engine_id = "ollama"

    def __init__(self) -> None:
        self.calls = 0
        self.second_call = threading.Event()
        self.last_error_code = "invalid_output"

    def extract(self, exchange):
        self.calls += 1
        if self.calls > 1:
            self.second_call.set()
        return []


def _candidate_job_state(archive, exchange_id: str) -> str:
    with sqlite3.connect(archive.path) as connection:
        row = connection.execute(
            "SELECT candidate_state FROM conversation_exchanges WHERE id = ?",
            (exchange_id,),
        ).fetchone()
    return row[0] if row is not None else ""


def _memory_job_state(archive, subject_id: str) -> str:
    with sqlite3.connect(archive.path) as connection:
        row = connection.execute(
            "SELECT state FROM memory_jobs WHERE subject_id = ?",
            (subject_id,),
        ).fetchone()
    return row[0] if row is not None else ""


def _service_api():
    try:
        from openjarvis.memory.archive import PersonalMemoryArchive
        from openjarvis.memory.personal_models import CandidateDraft
        from openjarvis.memory.personal_service import (
            PersonalMemoryService,
            build_personal_memory_service,
            record_and_publish_completed_exchange,
        )
    except ImportError:
        pytest.fail("personal memory service API is missing")
    return (
        PersonalMemoryArchive,
        CandidateDraft,
        PersonalMemoryService,
        build_personal_memory_service,
        record_and_publish_completed_exchange,
    )


def _service(
    tmp_path: Path,
    bus: EventBus,
    extractor,
    *,
    max_queue: int = 8,
    **kwargs,
):
    PersonalMemoryArchive, _, PersonalMemoryService, _, _ = _service_api()
    return PersonalMemoryService(
        PersonalMemoryArchive(tmp_path / "personal.db"),
        extractor,
        event_bus=bus,
        max_queue=max_queue,
        **kwargs,
    )


def test_archive_is_written_before_the_completed_exchange_event(tmp_path):
    """Event subscribers must always receive an ID that already exists on disk."""
    _, CandidateDraft, _, _, record_and_publish_completed_exchange = _service_api()
    bus = EventBus(record_history=True)
    service = _service(
        tmp_path,
        bus,
        _FakeExtractor(
            [CandidateDraft("fact", "User is preparing for an exam.", 0.8, 0.9)]
        ),
    )
    service.start()
    try:
        assert record_and_publish_completed_exchange(
            bus,
            service,
            "I am preparing for an exam.",
            "I can help.",
            source="cli.chat",
        )
        event = next(
            event
            for event in bus.history
            if event.event_type is EventType.CHAT_EXCHANGE_COMPLETED
        )
        exchange = service.archive.get_exchange(event.data["exchange_id"])
        assert exchange is not None
        assert exchange.user_text == "I am preparing for an exam."
    finally:
        service.stop()


def test_archive_write_failure_does_not_publish_a_completion_event():
    """A personal archive failure must stop the event rather than claim durability."""
    _, _, _, _, record_and_publish_completed_exchange = _service_api()

    class _FailingArchiveService:
        def archive_exchange(self, **kwargs):
            raise OSError("disk unavailable")

    bus = EventBus(record_history=True)
    published = record_and_publish_completed_exchange(
        bus,
        _FailingArchiveService(),
        "keep this safe",
        "reply",
        source="cli.chat",
    )

    assert published is False
    assert bus.history == []


def test_full_queue_leaves_an_archived_exchange_available_for_retry(tmp_path):
    """Queue pressure may delay work but must not erase the archived exchange."""
    bus = EventBus()
    gate = threading.Event()
    extractor = _FakeExtractor(gate=gate)
    service = _service(tmp_path, bus, extractor, max_queue=1)
    service.start()
    try:
        first = service.archive_exchange(
            user_text="one", assistant_text="a", source="test"
        )
        assert service.enqueue_exchange(first.id) is True
        assert extractor.started.wait(timeout=1.0)

        second = service.archive_exchange(
            user_text="two", assistant_text="b", source="test"
        )
        third = service.archive_exchange(
            user_text="three", assistant_text="c", source="test"
        )
        assert service.enqueue_exchange(second.id) is True
        assert service.enqueue_exchange(third.id) is False
        assert third.id in service.archive.pending_exchange_ids(limit=10)
    finally:
        gate.set()
        service.stop()


def test_invalid_candidate_output_remains_failed_without_a_hot_retry_loop(tmp_path):
    """Malformed local-model output stays retryable instead of becoming complete."""
    extractor = _InvalidOutputExtractor()
    service = _service(tmp_path, EventBus(), extractor)
    service.start()
    try:
        exchange = service.archive_exchange(
            user_text="remember this preference",
            assistant_text="noted",
            source="test",
        )
        assert service.enqueue_exchange(exchange.id) is True
        deadline = time.time() + 1.0
        while (
            time.time() < deadline
            and _candidate_job_state(service.archive, exchange.id) != "failed"
        ):
            time.sleep(0.01)

        assert _candidate_job_state(service.archive, exchange.id) == "failed"
        assert _memory_job_state(service.archive, exchange.id) == "failed"
        assert extractor.second_call.wait(timeout=0.1) is False
    finally:
        service.stop()


def test_cloud_engine_cannot_build_a_personal_memory_service(tmp_path):
    """The service factory must fail closed before a cloud engine can be invoked."""
    _, _, _, build_personal_memory_service, _ = _service_api()
    config = JarvisConfig()
    config.personal_memory = PersonalMemoryConfig(
        enabled=True,
        archive_path=str(tmp_path / "personal.db"),
        extraction_model="qwen3:8b",
    )
    assert (
        build_personal_memory_service(
            config,
            object(),
            engine_key="cloud",
            default_model="qwen3:8b",
            event_bus=EventBus(),
        )
        is None
    )


def test_local_factory_wires_developmental_job_handlers(tmp_path):
    """Production construction must not stop at extraction and evaluation."""
    _, _, _, build_personal_memory_service, _ = _service_api()
    config = JarvisConfig()
    config.personal_memory = PersonalMemoryConfig(
        enabled=True,
        mode="shadow",
        archive_path=str(tmp_path / "personal.db"),
        extraction_model="qwen3:8b",
    )

    service = build_personal_memory_service(
        config,
        object(),
        engine_key="ollama",
        default_model="qwen3:8b",
        event_bus=EventBus(),
    )

    assert service is not None
    assert set(service._job_handlers) == {
        "check_consolidation",
        "reflect_conflicts",
        "validate_insight",
    }


def test_extraction_is_followed_by_durable_candidate_evaluation(tmp_path):
    """Stopping after extraction would leave an explicit rule unusable forever."""
    _, CandidateDraft, _, _, _ = _service_api()
    bus = EventBus()
    service = _service(
        tmp_path,
        bus,
        _FakeExtractor(
            [
                CandidateDraft(
                    "constraint",
                    "Do not mention timers unless I ask.",
                    1.0,
                    1.0,
                    temporal_scope="until_changed",
                    subject="topic:timers",
                )
            ]
        ),
    )
    service.start()
    try:
        exchange = service.archive_exchange(
            user_text="Do not mention timers unless I ask.",
            assistant_text="Understood.",
            source="test",
        )
        assert service.enqueue_exchange(exchange.id) is True
        deadline = time.time() + 2.0
        claims = []
        while time.time() < deadline:
            claims = service.archive.get_active_claims()
            if claims:
                break
            time.sleep(0.01)

        assert len(claims) == 1
        assert claims[0].kind == "constraint"
        assert claims[0].content == "Do not mention timers unless I ask."
    finally:
        service.stop()


def test_stopping_during_a_slow_job_does_not_allow_a_second_worker(tmp_path):
    """A timed-out stop must not permit overlapping local model calls on restart."""
    gate = threading.Event()
    extractor = _FakeExtractor(gate=gate)
    service = _service(tmp_path, EventBus(), extractor)
    service.start()
    exchange = service.archive_exchange(
        user_text="slow memory",
        assistant_text="reply",
        source="test",
    )
    assert service.enqueue_exchange(exchange.id) is True
    assert extractor.started.wait(timeout=1.0)

    service.stop(timeout=0.01)
    original_thread = service._thread
    service.start()

    assert service._thread is original_thread
    gate.set()
    original_thread.join(timeout=1.0)


def test_chat_busy_defers_reflection_without_running_the_extractor(tmp_path):
    """Heavy reflection work must yield while the user is actively chatting."""
    extractor = _FakeExtractor()
    service = _service(
        tmp_path,
        EventBus(),
        extractor,
        chat_is_busy=lambda: True,
        last_user_activity=lambda: time.time(),
        idle_before_reflection_seconds=30.0,
    )
    job = service.archive.enqueue_job(
        job_type="reflect_conflicts",
        subject_id="schema-1",
        idempotency_key="reflect_conflicts:schema-1:v1",
    )

    service.start()
    try:
        deadline = time.time() + 1.0
        while time.time() < deadline:
            state = service.archive.get_job(job.id).state
            if state == "deferred":
                break
            time.sleep(0.01)

        assert service.archive.get_job(job.id).state == "deferred"
        assert extractor.started.is_set() is False
    finally:
        service.stop()


def test_evaluation_schedules_separate_consolidation_job(tmp_path):
    """Schema work must run as a distinct durable job after claim evaluation."""
    _, CandidateDraft, _, _, _ = _service_api()
    handled = threading.Event()
    observed = {}

    def handle_consolidation(job):
        observed["job_type"] = job.job_type
        observed["claim_count"] = len(service.archive.get_active_claims())
        handled.set()

    service = _service(
        tmp_path,
        EventBus(),
        _FakeExtractor(
            [CandidateDraft("fact", "Running improved my mood today.", 0.8, 0.9)]
        ),
        job_handlers={"check_consolidation": handle_consolidation},
    )
    service.start()
    try:
        exchange = service.archive_exchange(
            user_text="Running improved my mood today.",
            assistant_text="",
            source="test",
        )
        assert service.enqueue_exchange(exchange.id)
        assert handled.wait(timeout=2.0)
        assert observed == {"job_type": "check_consolidation", "claim_count": 1}
    finally:
        service.stop()
