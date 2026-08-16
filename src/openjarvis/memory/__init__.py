"""Native persistent long-term memory for OpenJarvis.

This package provides the automatic memory service that extracts durable facts
from conversations in the background and persists them across sessions. It is
started and stopped as part of the ``jarvis serve`` / ``jarvis chat`` lifecycle
and configured via the ``[memory]`` section of ``config.toml``.
"""

from __future__ import annotations

from openjarvis.memory.archive import PersonalMemoryArchive
from openjarvis.memory.extractor import FactExtractor
from openjarvis.memory.personal_models import (
    CandidateDraft,
    ConversationExchange,
    MemoryCandidate,
)
from openjarvis.memory.personal_service import (
    PersonalMemoryService,
    build_personal_memory_service,
    record_and_publish_completed_exchange,
)
from openjarvis.memory.service import (
    MemoryService,
    build_memory_service,
    publish_completed_exchange,
)
from openjarvis.memory.store import (
    Fact,
    FactStore,
    LocalFactStore,
    create_fact_store,
    load_configured_facts,
)

__all__ = [
    "Fact",
    "FactStore",
    "FactExtractor",
    "CandidateDraft",
    "ConversationExchange",
    "LocalFactStore",
    "MemoryCandidate",
    "MemoryService",
    "PersonalMemoryArchive",
    "PersonalMemoryService",
    "build_memory_service",
    "build_personal_memory_service",
    "create_fact_store",
    "load_configured_facts",
    "publish_completed_exchange",
    "record_and_publish_completed_exchange",
]
