"""Local-engine extraction of provisional personal-memory candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openjarvis.core.types import Message, Role
from openjarvis.memory.personal_models import CandidateDraft, ConversationExchange

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
    "Extract only provisional personal-memory candidates from one conversation.\n"
    "Return ONLY a JSON array. Each item must be an object with exactly these useful\n"
    "fields: kind (fact or episode), content, importance (0 to 1),\n"
    "confidence (0 to 1).\n"
    "Include only durable user-related information. Return [] when nothing is useful."
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
        self.last_error_code = error_code
        return drafts

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
            kind = item.get("kind")
            content_value = item.get("content")
            importance = item.get("importance")
            confidence = item.get("confidence")
            if kind not in ("fact", "episode") or not isinstance(content_value, str):
                continue
            clean_content = content_value.strip()
            if not clean_content or len(clean_content) > 500:
                continue
            if (
                isinstance(importance, bool)
                or isinstance(confidence, bool)
                or not isinstance(importance, (int, float))
                or not isinstance(confidence, (int, float))
            ):
                continue
            key = (kind, clean_content.casefold())
            if key in seen:
                continue
            try:
                draft = CandidateDraft(kind, clean_content, importance, confidence)
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
