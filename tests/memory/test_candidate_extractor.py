"""Tests for local-only provisional personal-memory extraction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakeEngine:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.response


def _extractor_api():
    try:
        from openjarvis.memory.candidate_extractor import (
            PersonalCandidateExtractor,
            is_local_personal_memory_engine,
        )
        from openjarvis.memory.personal_models import ConversationExchange
    except ImportError:
        pytest.fail("personal candidate extractor API is missing")
    return (
        PersonalCandidateExtractor,
        is_local_personal_memory_engine,
        ConversationExchange,
    )


def _exchange():
    _, _, ConversationExchange = _extractor_api()
    return ConversationExchange(
        id="exchange-1",
        created_at=1.0,
        archived_at=1.0,
        source="test",
        user_text="Please keep answers short.",
        assistant_text="I will.",
        content_hash="hash",
    )


def test_extractor_keeps_only_supported_provisional_candidate_kinds():
    """A model's unsupported candidate type must never reach the archive."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    engine = _FakeEngine(
        '[{"kind":"fact","content":"User prefers concise answers.",'
        '"importance":0.8,"confidence":0.9},'
        '{"kind":"pattern","content":"ignore","importance":1,"confidence":1}]'
    )

    drafts = PersonalCandidateExtractor(engine, "qwen3:8b").extract(_exchange())

    assert [(draft.kind, draft.content) for draft in drafts] == [
        ("fact", "User prefers concise answers.")
    ]


def test_extractor_rejects_malformed_model_output_without_raising():
    """A broken local-model answer must produce no provisional memory."""
    PersonalCandidateExtractor, _, _ = _extractor_api()

    extractor = PersonalCandidateExtractor(_FakeEngine("not json"), "qwen3:8b")
    drafts = extractor.extract(_exchange())

    assert drafts == []
    assert extractor.last_error_code == "invalid_output"


def test_extractor_marks_schema_invalid_json_array_for_retry():
    """A non-empty array with no valid candidate is not a legitimate empty result."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[{}]"), "qwen3:8b")

    drafts = extractor.extract(_exchange())

    assert drafts == []
    assert extractor.last_error_code == "invalid_output"


def test_extractor_keeps_the_verified_engine_name_for_candidate_provenance():
    """Persisted candidates must name the validated engine that proposed them."""
    PersonalCandidateExtractor, _, _ = _extractor_api()

    extractor = PersonalCandidateExtractor(
        _FakeEngine("[]"),
        "qwen3:8b",
        engine_id="ollama",
    )

    assert extractor.engine_id == "ollama"


@pytest.mark.parametrize(
    ("engine_key", "host"),
    [
        ("cloud", ""),
        ("litellm", ""),
        ("multi", ""),
        ("ollama", "http://192.168.1.10:11434"),
    ],
)
def test_local_engine_guard_rejects_a_cloud_or_network_address(engine_key, host):
    """Any engine that can leave this computer is forbidden for personal memory."""
    _, is_local_personal_memory_engine, _ = _extractor_api()
    config = SimpleNamespace(
        engine=SimpleNamespace(**{engine_key: SimpleNamespace(host=host)})
    )

    assert is_local_personal_memory_engine(config, engine_key) is False


def test_local_engine_guard_accepts_a_loopback_engine():
    """A known engine listening only on this computer is allowed."""
    _, is_local_personal_memory_engine, _ = _extractor_api()
    config = SimpleNamespace(
        engine=SimpleNamespace(ollama=SimpleNamespace(host="http://127.0.0.1:11434"))
    )

    assert is_local_personal_memory_engine(config, "ollama") is True


def test_local_engine_guard_accepts_ollamas_default_loopback_host(monkeypatch):
    """An unconfigured Ollama engine uses its built-in local address."""
    _, is_local_personal_memory_engine, _ = _extractor_api()
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    config = SimpleNamespace(engine=SimpleNamespace(ollama=SimpleNamespace(host="")))

    assert is_local_personal_memory_engine(config, "ollama") is True


def test_local_engine_guard_rejects_a_remote_ollama_environment_host(monkeypatch):
    """Ollama's environment override must remain subject to the local-only rule."""
    _, is_local_personal_memory_engine, _ = _extractor_api()
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.10:11434")
    config = SimpleNamespace(engine=SimpleNamespace(ollama=SimpleNamespace(host="")))

    assert is_local_personal_memory_engine(config, "ollama") is False
