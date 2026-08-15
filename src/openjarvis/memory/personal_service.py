"""Durable, local-only background processing for personal memory candidates."""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from typing import Any, Optional

from openjarvis.core.events import Event, EventBus, EventType
from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.candidate_extractor import (
    PersonalCandidateExtractor,
    is_local_personal_memory_engine,
)
from openjarvis.memory.personal_models import ConversationExchange
from openjarvis.memory.service import publish_completed_exchange

logger = logging.getLogger(__name__)
_STOP = object()


class PersonalMemoryService:
    """Archive exchanges first, then create provisional candidates asynchronously."""

    def __init__(
        self,
        archive: PersonalMemoryArchive,
        extractor: PersonalCandidateExtractor,
        *,
        event_bus: EventBus | None = None,
        max_queue: int = 256,
    ) -> None:
        self._archive = archive
        self._extractor = extractor
        self._event_bus = event_bus
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
        self._running.set()
        self._archive.release_in_progress_jobs()
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
        with self._queue_lock:
            if exchange_id in self._queued_ids:
                return True
            try:
                self._queue.put_nowait(exchange_id)
            except queue.Full:
                return False
            self._queued_ids.add(exchange_id)
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
            if not self.enqueue_exchange(exchange_id):
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
            exchange_id = str(job)
            with self._queue_lock:
                self._queued_ids.discard(exchange_id)
            try:
                exchange = self._archive.claim_candidate_job(exchange_id)
                if exchange is not None:
                    drafts = self._extractor.extract(exchange)
                    error_code = str(
                        getattr(self._extractor, "last_error_code", "") or ""
                    )
                    if error_code:
                        self._archive.fail_candidate_job(
                            exchange.id,
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
                        self._archive.complete_candidate_job(
                            exchange.id,
                            drafts,
                            engine_id=getattr(self._extractor, "engine_id", "local"),
                            extractor_version="personal-memory-v1",
                        )
            except Exception:  # noqa: BLE001 - a failed job must remain retryable
                self._archive.fail_candidate_job(
                    exchange_id,
                    error_code="extract_failed",
                    retry_after=None,
                )
                logger.warning(
                    "Personal memory candidate job failed",
                    extra={"exchange_id": exchange_id},
                )
            finally:
                self._queue.task_done()
            self._refill_queue()
            if not self._running.is_set() and self._queue.empty():
                return


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
    if not is_local_personal_memory_engine(config, engine_key):
        logger.warning(
            "Personal memory service disabled because engine is not local",
            extra={"engine_key": engine_key},
        )
        return None
    model = str(getattr(personal, "extraction_model", "") or default_model).strip()
    if not model:
        return None
    archive = PersonalMemoryArchive(getattr(personal, "archive_path", ""))
    extractor = PersonalCandidateExtractor(engine, model, engine_id=engine_key)
    return PersonalMemoryService(
        archive,
        extractor,
        event_bus=event_bus,
        max_queue=getattr(personal, "max_queue", 256),
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
    "PersonalMemoryService",
    "build_personal_memory_service",
    "record_and_publish_completed_exchange",
]
