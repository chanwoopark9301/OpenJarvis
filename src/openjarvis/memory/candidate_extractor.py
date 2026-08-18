"""Local-engine extraction of provisional personal-memory candidates."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openjarvis.core.types import Message, Role
from openjarvis.memory.personal_models import (
    CandidateDraft,
    CandidateKind,
    ConversationExchange,
)

LOCAL_PERSONAL_MEMORY_ENGINE_KEYS = frozenset(
    {
        "ollama",
        "vllm",
        "sglang",
        "llamacpp",
        "mlx",
        "lmstudio",
        "exo",
        "nexa",
        "uzu",
        "apple_fm",
        "gemma_cpp",
        "lemonade",
    }
)
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_SYSTEM_PROMPT = (
    "You are the semantic proposer for provisional personal memory. Interpret the "
    "bounded recent dialogue as chronological context, then classify only claims "
    "supported by the current user message.\n"
    "The current user message is the only valid evidence. Assistant text and prior "
    "dialogue are context, never evidence. Every evidence_excerpt must be copied "
    "exactly from the current user message.\n"
    "Propose atomic candidates and choose their semantic kind, content, importance, "
    "confidence, temporal scope, subject, and correction target. Preserve direct "
    "corrections, constraints, role preferences, and capability boundaries when the "
    "current user message supports them. Do not infer hidden psychology.\n"
    "Assign stable dotted entity.attribute subjects so later corrections can address "
    "the same memory. Reuse the same subject when the current message corrects prior "
    "dialogue. The assistant's own name uses assistant.name; the user's name uses "
    "user.name. A replacement of a prior value is a correction, not a fact or "
    "preference. Facts and preferences describe user attributes, not instructions "
    "about the assistant. A first direct instruction defining assistant identity or "
    "role is a role preference; if it replaces or rejects a prior value, it is a "
    "correction. Keep durable role settings until_changed. Importance and confidence "
    "are decimal numbers from 0.0 through 1.0, never percentages or 0-100 scores.\n"
    "Return an empty candidates array for greetings or when the current user message "
    "contains no useful personal evidence. Return only the schema-constrained JSON "
    "object."
)
_CANDIDATE_FIELDS = (
    "kind",
    "content",
    "importance",
    "confidence",
    "temporal_scope",
    "subject",
    "target_claim_id",
    "evidence_excerpt",
)
_TEMPORAL_SCOPES = (
    "current",
    "dated",
    "persistent",
    "until_changed",
    "unspecified",
)
_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [kind.value for kind in CandidateKind],
                    },
                    "content": {"type": "string", "minLength": 1, "maxLength": 500},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "temporal_scope": {
                        "type": "string",
                        "enum": list(_TEMPORAL_SCOPES),
                    },
                    "subject": {"type": "string", "minLength": 1, "maxLength": 120},
                    "target_claim_id": {"type": "string", "maxLength": 120},
                    "evidence_excerpt": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                },
                "required": list(_CANDIDATE_FIELDS),
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "personal_memory_candidates",
        "strict": True,
        "schema": _CANDIDATE_SCHEMA,
    },
}


def is_local_personal_memory_engine(config: Any, engine_key: str) -> bool:
    """Return whether *engine_key* is a configured loopback-only local engine."""
    key = (engine_key or "").strip().lower()
    if key not in LOCAL_PERSONAL_MEMORY_ENGINE_KEYS:
        return False

    engine_config = getattr(config, "engine", None)
    if engine_config is None:
        return False
    selected_config = getattr(engine_config, key, None)
    if selected_config is None:
        return False

    if key == "gemma_cpp":
        model_path = str(getattr(selected_config, "model_path", "") or "").strip()
        return bool(model_path and Path(model_path).expanduser().exists())

    host = str(getattr(selected_config, "host", "") or "").strip()
    if not host and key == "ollama":
        # Match OllamaEngine: an omitted host means its loopback default unless
        # an environment override explicitly points elsewhere.
        host = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    if not host:
        return False
    try:
        parsed = urlparse(host)
    except ValueError:
        return False
    return parsed.hostname in _LOOPBACK_HOSTS


class PersonalCandidateExtractor:
    """Ask a local model for strictly structured provisional candidates."""

    def __init__(
        self,
        engine: Any,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        engine_id: str = "",
    ) -> None:
        self._engine = engine
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self.engine_id = str(engine_id or getattr(engine, "engine_id", "") or "local")
        self.last_error_code = ""

    def extract(
        self,
        exchange: ConversationExchange,
        *,
        recent_exchanges: tuple[ConversationExchange, ...] = (),
    ) -> list[CandidateDraft]:
        """Return at most five structurally valid model proposals; never raise."""
        self.last_error_code = ""
        if not exchange.user_text.strip():
            return []
        recent = sorted(
            recent_exchanges,
            key=lambda item: (item.created_at, item.id),
        )[-6:]
        messages = [
            Message(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=self._context_prompt(exchange, recent),
            ),
        ]
        try:
            result = self._engine.generate(
                messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format=_RESPONSE_FORMAT,
                _openjarvis_background=True,
            )
        except Exception:  # noqa: BLE001 - candidate extraction is best effort
            self.last_error_code = "engine_error"
            return []

        content = result.get("content", "") if isinstance(result, dict) else str(result)
        drafts, error_code = self._parse(content, exchange.user_text)
        self.last_error_code = error_code
        return drafts

    @staticmethod
    def _context_prompt(
        exchange: ConversationExchange,
        recent_exchanges: list[ConversationExchange],
    ) -> str:
        context = [
            {
                "user": item.user_text,
                "assistant": item.assistant_text or "",
            }
            for item in recent_exchanges
        ]
        current = {
            "user": exchange.user_text,
            "assistant": exchange.assistant_text or "",
        }
        return (
            "RECENT DIALOGUE (chronological context only; never evidence):\n"
            f"{json.dumps(context, ensure_ascii=False)}\n"
            "CURRENT EXCHANGE (only current.user may supply evidence):\n"
            f"{json.dumps(current, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse(
        content: str,
        current_user_text: str,
    ) -> tuple[list[CandidateDraft], str]:
        """Validate only schema structure and exact current-message evidence."""
        try:
            raw = json.loads(content)
        except (TypeError, json.JSONDecodeError, ValueError):
            return [], "invalid_output"
        if not isinstance(raw, dict) or set(raw) != {"candidates"}:
            return [], "invalid_output"
        candidates = raw["candidates"]
        if not isinstance(candidates, list):
            return [], "invalid_output"
        if len(candidates) > 5:
            return [], "invalid_output"

        drafts: list[CandidateDraft] = []
        for item in candidates:
            if not isinstance(item, dict) or set(item) != set(_CANDIDATE_FIELDS):
                return [], "invalid_output"
            try:
                candidate_kind = CandidateKind(item["kind"])
            except (TypeError, ValueError):
                return [], "invalid_output"
            content_value = item["content"]
            temporal_scope = item["temporal_scope"]
            subject = item["subject"]
            target_claim_id = item["target_claim_id"]
            evidence_excerpt = item["evidence_excerpt"]
            if not all(
                isinstance(value, str)
                for value in (
                    content_value,
                    temporal_scope,
                    subject,
                    target_claim_id,
                    evidence_excerpt,
                )
            ):
                return [], "invalid_output"
            if (
                not content_value.strip()
                or len(content_value) > 500
                or temporal_scope not in _TEMPORAL_SCOPES
                or not subject.strip()
                or len(subject) > 120
                or len(target_claim_id) > 120
                or not evidence_excerpt.strip()
                or len(evidence_excerpt) > 500
                or evidence_excerpt not in current_user_text
            ):
                return [], "invalid_output"
            importance = item["importance"]
            confidence = item["confidence"]
            if not PersonalCandidateExtractor._valid_score(importance) or not (
                PersonalCandidateExtractor._valid_score(confidence)
            ):
                return [], "invalid_output"
            try:
                drafts.append(
                    CandidateDraft(
                        candidate_kind,
                        content_value,
                        importance,
                        confidence,
                        temporal_scope=temporal_scope,
                        subject=subject,
                        target_claim_id=target_claim_id,
                        evidence_excerpt=evidence_excerpt,
                    )
                )
            except (TypeError, ValueError):
                return [], "invalid_output"
        return drafts, ""

    @staticmethod
    def _valid_score(value: Any) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
        )


__all__ = [
    "LOCAL_PERSONAL_MEMORY_ENGINE_KEYS",
    "PersonalCandidateExtractor",
    "is_local_personal_memory_engine",
]
