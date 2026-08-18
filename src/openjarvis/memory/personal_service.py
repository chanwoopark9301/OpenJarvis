"""Durable, local-only background processing for personal memory candidates."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable, Mapping, Optional

from openjarvis.core.events import Event, EventBus, EventType
from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.candidate_extractor import (
    PersonalCandidateExtractor,
    is_local_personal_memory_engine,
)
from openjarvis.memory.evaluator import MemoryEvaluator
from openjarvis.memory.personal_models import ConversationExchange, MemoryJob
from openjarvis.memory.service import publish_completed_exchange
from openjarvis.memory.social_reflection import (
    ExternalAction,
    SocialReflectionCoordinator,
)
from openjarvis.memory.understanding_gap import (
    GapKind,
    GapRoute,
    UnderstandingGap,
    UnderstandingGapGate,
)

logger = logging.getLogger(__name__)
_STOP = object()
EXTRACT_CANDIDATES = "extract_candidates"
EVALUATE_CANDIDATE = "evaluate_candidate"
CHECK_CONSOLIDATION = "check_consolidation"
REFLECT_CONFLICTS = "reflect_conflicts"
VALIDATE_INSIGHT = "validate_insight"
IMPORT_LEGACY = "import_legacy"
PERSONAL_CANDIDATE_EXTRACTOR_VERSION = "personal-memory-v8"


class PersonalMemoryService:
    """Archive exchanges first, then create provisional candidates asynchronously."""

    def __init__(
        self,
        archive: PersonalMemoryArchive,
        extractor: PersonalCandidateExtractor,
        *,
        event_bus: EventBus | None = None,
        max_queue: int = 256,
        evaluator: MemoryEvaluator | None = None,
        chat_is_busy: Callable[[], bool] | None = None,
        last_user_activity: Callable[[], float] | None = None,
        idle_before_reflection_seconds: float = 30.0,
        job_handlers: Mapping[str, Callable[[MemoryJob], None]] | None = None,
        external_reflection_mode: str = "ask",
        question_answer_handler: Callable[..., Any] | None = None,
    ) -> None:
        self._archive = archive
        self._extractor = extractor
        self._event_bus = event_bus
        self._evaluator = evaluator or MemoryEvaluator(archive)
        self._chat_is_busy = chat_is_busy or (lambda: False)
        self._last_user_activity = last_user_activity or (lambda: 0.0)
        self._idle_before_reflection_seconds = max(
            0.0,
            float(idle_before_reflection_seconds),
        )
        self._job_handlers = dict(job_handlers or {})
        self._gap_gate = UnderstandingGapGate(external_mode=external_reflection_mode)
        self._social_reflection = SocialReflectionCoordinator(
            archive,
            mode=external_reflection_mode,
        )
        self._question_answer_handler = question_answer_handler
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, max_queue))
        self._max_queue = max(1, max_queue)
        self._queued_ids: set[str] = set()
        self._queue_lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._subscribed = False

    @property
    def archive(self) -> PersonalMemoryArchive:
        """The service's durable local archive."""
        return self._archive

    @property
    def is_running(self) -> bool:
        """Whether the service worker is accepting and processing jobs."""
        return self._running.is_set()

    def start(self) -> None:
        """Start one worker and recover work interrupted by a prior shutdown."""
        if self._running.is_set():
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._archive.release_in_progress_jobs()
        self._archive.recover_stale_candidate_jobs(PERSONAL_CANDIDATE_EXTRACTOR_VERSION)
        self._archive.recover_pending_candidate_jobs()
        self._subscribe_events()
        self._thread = threading.Thread(
            target=self._loop,
            name="personal-memory-service",
            daemon=True,
        )
        self._thread.start()
        self._refill_queue()
        logger.debug("Personal memory service started")

    def stop(self, timeout: float = 2.0) -> None:
        """Stop accepting work; archived incomplete jobs remain retryable."""
        if not self._running.is_set():
            return
        self._running.clear()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        if thread is None or not thread.is_alive():
            with self._queue_lock:
                self._queued_ids.clear()
            self._thread = None
        self._unsubscribe_events()
        logger.debug("Personal memory service stopped")

    def archive_exchange(
        self,
        *,
        user_text: str,
        assistant_text: str,
        source: str,
        agent_id: str = "",
        session_id: str = "",
        exchange_id: str | None = None,
    ) -> ConversationExchange:
        """Write raw evidence durably before any best-effort candidate work."""
        return self._archive.record_exchange(
            exchange_id=exchange_id or str(uuid.uuid4()),
            user_text=user_text,
            assistant_text=assistant_text,
            source=source,
            agent_id=agent_id,
            session_id=session_id,
        )

    def enqueue_exchange(self, exchange_id: str) -> bool:
        """Schedule a durable exchange without blocking the chat response path."""
        if not self._running.is_set() or not exchange_id:
            return False
        job = self._archive.enqueue_job(
            job_type=EXTRACT_CANDIDATES,
            subject_id=exchange_id,
            idempotency_key=(
                f"{EXTRACT_CANDIDATES}:"
                f"{PERSONAL_CANDIDATE_EXTRACTOR_VERSION}:{exchange_id}"
            ),
            priority=100,
        )
        return self._enqueue_job(job.id)

    def route_understanding_gap(
        self,
        kind: GapKind | str,
        *,
        user_requested_analysis: bool = False,
        sensitive: bool = False,
    ) -> GapRoute:
        """Choose user clarification before any possible outside lookup."""
        return self._gap_gate.route(
            UnderstandingGap(
                GapKind(kind),
                user_requested_analysis=user_requested_analysis,
                sensitive=sensitive,
            )
        )

    def prepare_external_reflection(
        self,
        query: str,
        *,
        user_requested_analysis: bool,
        consent_granted: bool,
        sensitive: bool = False,
    ) -> ExternalAction:
        """Apply request, consent, and deidentification gates without searching."""
        return self._social_reflection.prepare_request(
            query,
            user_requested_analysis=user_requested_analysis,
            consent_granted=consent_granted,
            sensitive=sensitive,
        )

    def answer_personal_question(
        self,
        question_id: str,
        answer_text: str,
        *,
        confirms: bool,
    ) -> Any:
        """Send an explicit user answer back through validated accommodation."""
        if self._question_answer_handler is None:
            raise RuntimeError("question answering is unavailable")
        return self._question_answer_handler(
            question_id,
            answer_text,
            confirms=confirms,
        )

    def _enqueue_job(self, job_id: str) -> bool:
        """Place a durable job ID in the bounded in-memory wake-up queue."""
        with self._queue_lock:
            if job_id in self._queued_ids:
                return True
            try:
                self._queue.put_nowait(job_id)
            except queue.Full:
                return False
            self._queued_ids.add(job_id)
        return True

    def _subscribe_events(self) -> None:
        if self._event_bus is None or self._subscribed:
            return
        self._event_bus.subscribe(
            EventType.CHAT_EXCHANGE_COMPLETED,
            self._on_completed_exchange,
        )
        self._subscribed = True

    def _unsubscribe_events(self) -> None:
        if self._event_bus is None or not self._subscribed:
            return
        self._event_bus.unsubscribe(
            EventType.CHAT_EXCHANGE_COMPLETED,
            self._on_completed_exchange,
        )
        self._subscribed = False

    def _on_completed_exchange(self, event: Event) -> None:
        """Schedule by durable ID only; raw conversation text stays in SQLite."""
        exchange_id = str((event.data or {}).get("exchange_id", "") or "")
        if exchange_id:
            self.enqueue_exchange(exchange_id)

    def _refill_queue(self) -> None:
        """Let durable pending rows catch up after a full in-memory queue."""
        if not self._running.is_set():
            return
        for exchange_id in self._archive.pending_exchange_ids(limit=self._max_queue):
            self._archive.enqueue_job(
                job_type=EXTRACT_CANDIDATES,
                subject_id=exchange_id,
                idempotency_key=(
                    f"{EXTRACT_CANDIDATES}:"
                    f"{PERSONAL_CANDIDATE_EXTRACTOR_VERSION}:{exchange_id}"
                ),
                priority=100,
            )
        for job_id in self._archive.pending_job_ids(limit=self._max_queue):
            if not self._enqueue_job(job_id):
                break

    def _loop(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                if not self._running.is_set():
                    return
                self._refill_queue()
                continue
            if job is _STOP:
                self._queue.task_done()
                return
            job_id = str(job)
            with self._queue_lock:
                self._queued_ids.discard(job_id)
            claimed_job = None
            try:
                claimed_job = self._archive.claim_job(job_id)
                if claimed_job is None:
                    continue
                if self._should_defer(claimed_job.job_type):
                    self._archive.defer_job(claimed_job.id, retry_after=1.0)
                    continue
                if claimed_job.job_type == EXTRACT_CANDIDATES:
                    exchange = self._archive.claim_candidate_job(claimed_job.subject_id)
                    if exchange is None:
                        self._archive.complete_job(claimed_job.id)
                        continue
                    recent = self._archive.recent_exchanges(exchange.id, limit=6)
                    drafts = self._extractor.extract(
                        exchange,
                        recent_exchanges=tuple(recent),
                    )
                    error_code = str(
                        getattr(self._extractor, "last_error_code", "") or ""
                    )
                    if error_code:
                        self._archive.fail_candidate_job(
                            exchange.id,
                            error_code=error_code,
                            retry_after=None,
                        )
                        self._archive.fail_job(
                            claimed_job.id,
                            error_code=error_code,
                            retry_after=None,
                        )
                        logger.warning(
                            "Personal memory candidate extraction failed",
                            extra={
                                "exchange_id": exchange.id,
                                "error_code": error_code,
                            },
                        )
                    else:
                        candidates = self._archive.complete_candidate_job(
                            exchange.id,
                            drafts,
                            engine_id=getattr(self._extractor, "engine_id", "local"),
                            extractor_version=PERSONAL_CANDIDATE_EXTRACTOR_VERSION,
                        )
                        for candidate in candidates:
                            evaluation_job = self._archive.enqueue_job(
                                job_type=EVALUATE_CANDIDATE,
                                subject_id=candidate.id,
                                idempotency_key=(
                                    f"{EVALUATE_CANDIDATE}:{candidate.id}"
                                ),
                                priority=90,
                            )
                            self._enqueue_job(evaluation_job.id)
                        self._archive.complete_job(claimed_job.id)
                elif claimed_job.job_type == EVALUATE_CANDIDATE:
                    result = self._evaluator.evaluate(claimed_job.subject_id)
                    if result.applied:
                        consolidation_job = self._archive.enqueue_job(
                            job_type=CHECK_CONSOLIDATION,
                            subject_id=claimed_job.subject_id,
                            idempotency_key=(
                                f"{CHECK_CONSOLIDATION}:{claimed_job.subject_id}"
                            ),
                            priority=50,
                        )
                        self._enqueue_job(consolidation_job.id)
                    self._archive.complete_job(claimed_job.id)
                elif claimed_job.job_type in {
                    CHECK_CONSOLIDATION,
                    REFLECT_CONFLICTS,
                    VALIDATE_INSIGHT,
                    IMPORT_LEGACY,
                }:
                    handler = self._job_handlers.get(claimed_job.job_type)
                    if handler is not None:
                        handler(claimed_job)
                    self._archive.complete_job(claimed_job.id)
                else:
                    self._archive.fail_job(
                        claimed_job.id,
                        error_code="unsupported_job_type",
                        retry_after=60.0,
                    )
            except Exception:  # noqa: BLE001 - a failed job must remain retryable
                if claimed_job is not None:
                    if claimed_job.job_type == EXTRACT_CANDIDATES:
                        self._archive.fail_candidate_job(
                            claimed_job.subject_id,
                            error_code="extract_failed",
                            retry_after=None,
                        )
                    self._archive.fail_job(
                        claimed_job.id,
                        error_code="job_failed",
                        retry_after=None,
                    )
                logger.warning(
                    "Personal memory job failed",
                    extra={"job_id": job_id},
                )
            finally:
                self._queue.task_done()
            self._refill_queue()
            if not self._running.is_set() and self._queue.empty():
                return

    def _should_defer(self, job_type: str) -> bool:
        heavy_jobs = {CHECK_CONSOLIDATION, REFLECT_CONFLICTS, VALIDATE_INSIGHT}
        if job_type not in heavy_jobs:
            return False
        if self._chat_is_busy():
            return True
        idle_for = time.time() - float(self._last_user_activity())
        return idle_for < self._idle_before_reflection_seconds


def build_personal_memory_service(
    config: Any,
    engine: Any,
    *,
    engine_key: str,
    default_model: str = "",
    event_bus: EventBus | None = None,
) -> PersonalMemoryService | None:
    """Build the service only when its engine is explicitly loopback-local."""
    personal = getattr(config, "personal_memory", None)
    if personal is None or not getattr(personal, "enabled", False) or engine is None:
        return None
    if getattr(personal, "mode", "active") == "off":
        return None
    if not is_local_personal_memory_engine(config, engine_key):
        logger.warning(
            "Personal memory service disabled because engine is not local",
            extra={"engine_key": engine_key},
        )
        return None
    if not hasattr(engine, "is_generating"):
        logger.warning(
            "Personal memory service disabled because shared generation "
            "arbitration is unavailable"
        )
        return None
    model = str(getattr(personal, "extraction_model", "") or default_model).strip()
    if not model:
        return None
    archive = PersonalMemoryArchive(getattr(personal, "archive_path", ""))
    if (
        getattr(personal, "mode", "active") == "active"
        and not archive.rollout_is_active()
    ):
        logger.warning(
            "Personal memory active mode refused: rollout gate not activated"
        )
        return None
    extractor = PersonalCandidateExtractor(engine, model, engine_id=engine_key)
    from openjarvis.memory.developmental_pipeline import DevelopmentalMemoryPipeline
    from openjarvis.memory.reflection import ReflectionEngine
    from openjarvis.memory.relation_classifier import RelationClassifier

    evaluator = MemoryEvaluator(
        archive,
        relation_classifier=RelationClassifier(engine, model),
    )
    pipeline = DevelopmentalMemoryPipeline(
        archive,
        ReflectionEngine(
            engine,
            model,
            max_evidence=getattr(personal, "max_evidence_per_job", 12),
        ),
        max_evidence=getattr(personal, "max_evidence_per_job", 12),
    )
    return PersonalMemoryService(
        archive,
        extractor,
        event_bus=event_bus,
        evaluator=evaluator,
        job_handlers=pipeline.handlers(),
        max_queue=getattr(personal, "max_queue", 256),
        chat_is_busy=lambda: bool(getattr(engine, "is_generating", False)),
        idle_before_reflection_seconds=getattr(
            personal,
            "idle_before_reflection_seconds",
            30.0,
        ),
        external_reflection_mode=getattr(
            personal,
            "external_reflection_mode",
            "ask",
        ),
        question_answer_handler=pipeline.answer_question,
    )


def record_and_publish_completed_exchange(
    bus: EventBus | None,
    personal_memory_service: PersonalMemoryService | None,
    user_text: str,
    assistant_text: str = "",
    *,
    source: str = "",
    agent_id: str = "",
    session_id: str = "",
) -> bool:
    """Archive before publishing while preserving the legacy event contract."""
    exchange_id = ""
    archive_succeeded = True
    if personal_memory_service is not None:
        try:
            exchange_id = personal_memory_service.archive_exchange(
                user_text=user_text,
                assistant_text=assistant_text,
                source=source,
                agent_id=agent_id,
                session_id=session_id,
            ).id
        except Exception:  # noqa: BLE001 - archive failure must not break chat
            archive_succeeded = False
            logger.warning("Personal archive write failed", extra={"source": source})
    if not archive_succeeded:
        return False
    if bus is None:
        return bool(
            exchange_id
            and personal_memory_service is not None
            and personal_memory_service.enqueue_exchange(exchange_id)
        )
    return publish_completed_exchange(
        bus,
        user_text,
        assistant_text,
        source=source,
        exchange_id=exchange_id,
        agent_id=agent_id,
        session_id=session_id,
    )


__all__ = [
    "CHECK_CONSOLIDATION",
    "EVALUATE_CANDIDATE",
    "EXTRACT_CANDIDATES",
    "IMPORT_LEGACY",
    "PersonalMemoryService",
    "REFLECT_CONFLICTS",
    "VALIDATE_INSIGHT",
    "build_personal_memory_service",
    "record_and_publish_completed_exchange",
]
