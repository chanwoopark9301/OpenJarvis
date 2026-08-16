"""Local-engine extraction of provisional personal-memory candidates."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openjarvis.core.types import Message, Role
from openjarvis.memory.personal_models import (
    DIRECT_RULE_KINDS,
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
    "Extract atomic provisional personal-memory candidates from one conversation.\n"
    "Only the user's own words are personal evidence. Never treat assistant text as "
    "evidence, even when it contains a confident claim or suggestion.\n"
    "Return ONLY a JSON array. Allowed fields are kind, content, importance, "
    "confidence, temporal_scope, subject, and target_claim_id.\n"
    "kind must be fact, episode, preference, constraint, correction, "
    "role_preference, capability_boundary, or hypothesis.\n"
    "Use current for a temporary state, dated for a specific event, persistent for "
    "a recurring preference, until_changed for a direct rule, and unspecified only "
    "when the user's time scope is genuinely absent.\n"
    "Do not infer hidden psychology. Return [] when the user supplied no useful "
    "personal evidence."
)
_ALLOWED_FIELDS = frozenset(
    {
        "kind",
        "content",
        "importance",
        "confidence",
        "temporal_scope",
        "subject",
        "target_claim_id",
    }
)
_EXPLICIT_RULE_SIGNALS = (
    "하지 마",
    "기억해",
    "틀렸어",
    "할 수 없어",
    "역할은",
    "do not",
    "don't",
    "remember this",
    "you cannot",
    "your role is",
)


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

    def extract(self, exchange: ConversationExchange) -> list[CandidateDraft]:
        """Return at most five valid provisional candidates; never raise."""
        self.last_error_code = ""
        if not exchange.user_text.strip():
            return []
        messages = [
            Message(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
            Message(
                role=Role.USER,
                content=(
                    f"User: {exchange.user_text}\n"
                    f"Assistant: {exchange.assistant_text or ''}"
                ),
            ),
        ]
        try:
            result = self._engine.generate(
                messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception:  # noqa: BLE001 - candidate extraction is best effort
            self.last_error_code = "engine_error"
            return []

        content = result.get("content", "") if isinstance(result, dict) else str(result)
        drafts, error_code = self._parse(content)
        if not error_code and self._has_explicit_rule_signal(exchange.user_text):
            drafts = [
                replace(draft, importance=1.0)
                if draft.kind in DIRECT_RULE_KINDS
                else draft
                for draft in drafts
            ]
        self.last_error_code = error_code
        return drafts

    @staticmethod
    def _has_explicit_rule_signal(user_text: str) -> bool:
        normalized = user_text.casefold()
        return any(signal in normalized for signal in _EXPLICIT_RULE_SIGNALS)

    @staticmethod
    def _parse(content: str) -> tuple[list[CandidateDraft], str]:
        """Accept only a complete JSON array matching the candidate contract."""
        try:
            raw = json.loads(content)
        except (TypeError, json.JSONDecodeError, ValueError):
            return [], "invalid_output"
        if not isinstance(raw, list):
            return [], "invalid_output"

        drafts: list[CandidateDraft] = []
        seen: set[tuple[str, str]] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            if not set(item).issubset(_ALLOWED_FIELDS):
                continue
            kind = item.get("kind")
            content_value = item.get("content")
            importance = item.get("importance")
            confidence = item.get("confidence")
            try:
                candidate_kind = CandidateKind(kind)
            except (TypeError, ValueError):
                continue
            if not isinstance(content_value, str):
                continue
            clean_content = content_value.strip()
            if not clean_content or len(clean_content) > 500:
                continue
            temporal_scope = item.get("temporal_scope", "unspecified")
            subject = item.get("subject", "user")
            target_claim_id = item.get("target_claim_id", "")
            if not all(
                isinstance(value, str)
                for value in (temporal_scope, subject, target_claim_id)
            ):
                continue
            if (
                len(temporal_scope) > 80
                or len(subject) > 120
                or len(target_claim_id) > 120
            ):
                continue
            if (
                isinstance(importance, bool)
                or isinstance(confidence, bool)
                or not isinstance(importance, (int, float))
                or not isinstance(confidence, (int, float))
            ):
                continue
            key = (candidate_kind.value, clean_content.casefold())
            if key in seen:
                continue
            try:
                draft = CandidateDraft(
                    candidate_kind,
                    clean_content,
                    importance,
                    confidence,
                    temporal_scope=temporal_scope,
                    subject=subject,
                    target_claim_id=target_claim_id,
                )
            except (TypeError, ValueError):
                continue
            drafts.append(draft)
            seen.add(key)
            if len(drafts) == 5:
                break
        if raw and not drafts:
            return [], "invalid_output"
        return drafts, ""


__all__ = [
    "LOCAL_PERSONAL_MEMORY_ENGINE_KEYS",
    "PersonalCandidateExtractor",
    "is_local_personal_memory_engine",
]
