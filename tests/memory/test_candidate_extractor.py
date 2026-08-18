"""Tests for local-only provisional personal-memory extraction."""

from __future__ import annotations

import json
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


def _exchange(
    user_text="I enjoy tea.",
    assistant_text="I will.",
    *,
    exchange_id="exchange-1",
    created_at=1.0,
):
    _, _, ConversationExchange = _extractor_api()
    return ConversationExchange(
        id=exchange_id,
        created_at=created_at,
        archived_at=created_at,
        source="test",
        user_text=user_text,
        assistant_text=assistant_text,
        content_hash=f"hash-{exchange_id}",
    )


def _response(*candidates):
    return json.dumps({"candidates": list(candidates)}, ensure_ascii=False)


def _candidate(**overrides):
    candidate = {
        "kind": "fact",
        "content": "The user enjoys tea.",
        "importance": 0.8,
        "confidence": 0.9,
        "temporal_scope": "persistent",
        "subject": "user.preference.tea",
        "target_claim_id": "",
        "evidence_excerpt": "enjoy tea",
    }
    candidate.update(overrides)
    return candidate


def test_bare_name_correction_uses_recent_dialogue():
    """Removing recent context would make the correction impossible to resolve."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    engine = _FakeEngine(
        _response(
            _candidate(
                kind="correction",
                content="The assistant name is 공박사.",
                importance=1.0,
                confidence=1.0,
                temporal_scope="until_changed",
                subject="assistant.name",
                evidence_excerpt="공박사라고.",
            )
        )
    )
    drafts = PersonalCandidateExtractor(engine, "qwen3.5:9b").extract(
        _exchange("공박사라고.", "네, 공박사입니다.", created_at=3.0),
        recent_exchanges=(
            _exchange(
                "너의 이름은 조visor가 아니야. 공박사야.",
                "알겠습니다.",
                exchange_id="exchange-0",
                created_at=2.0,
            ),
        ),
    )

    assert len(drafts) == 1
    assert drafts[0].content == "The assistant name is 공박사."
    assert drafts[0].subject == "assistant.name"
    assert drafts[0].evidence_excerpt == "공박사라고."
    messages, _ = engine.calls[0]
    prompt = messages[-1].content
    assert prompt.index("너의 이름은 조visor가 아니야. 공박사야.") < prompt.index(
        "공박사라고."
    )


def test_evidence_excerpt_copied_only_from_assistant_text_is_rejected():
    """Assistant text must never become evidence for a proposed memory."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(
        _FakeEngine(
            _response(
                _candidate(
                    content="The user lives in Busan.",
                    evidence_excerpt="You live in Busan.",
                )
            )
        ),
        "qwen3.5:9b",
    )

    drafts = extractor.extract(_exchange("Where do I live?", "You live in Busan."))

    assert drafts == []
    assert extractor.last_error_code == "invalid_output"


def test_evidence_excerpt_absent_from_current_user_message_is_rejected():
    """Prior dialogue is context, never evidence for the current proposal."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(
        _FakeEngine(
            _response(
                _candidate(
                    content="The user lives in Busan.",
                    evidence_excerpt="I live in Busan.",
                )
            )
        ),
        "qwen3.5:9b",
    )

    drafts = extractor.extract(
        _exchange("What did I just say?", "You mentioned your home."),
        recent_exchanges=(
            _exchange(
                "I live in Busan.",
                "Thanks for telling me.",
                exchange_id="exchange-0",
                created_at=0.0,
            ),
        ),
    )

    assert drafts == []
    assert extractor.last_error_code == "invalid_output"


def test_model_produced_constraint_is_preserved():
    """Deleting model-produced direct kinds would lose a valid user constraint."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    engine = _FakeEngine(
        _response(
            _candidate(
                kind="constraint",
                content="Do not mention timers unless the user asks.",
                importance=1.0,
                confidence=1.0,
                temporal_scope="until_changed",
                subject="topic:timers",
                evidence_excerpt="Do not mention timers unless I ask.",
            )
        )
    )

    drafts = PersonalCandidateExtractor(engine, "qwen3.5:9b").extract(
        _exchange("Do not mention timers unless I ask.")
    )

    assert len(drafts) == 1
    assert drafts[0].kind == "constraint"
    assert drafts[0].content == "Do not mention timers unless the user asks."
    assert drafts[0].importance == 1.0


def test_greeting_returns_no_candidates():
    """A legitimate empty semantic result must complete without a retry error."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine(_response()), "qwen3.5:9b")

    assert extractor.extract(_exchange("안녕?")) == []
    assert extractor.last_error_code == ""


def test_extractor_passes_one_strict_candidates_schema():
    """Dropping structured output would reopen the free-form parsing boundary."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    engine = _FakeEngine(_response())

    PersonalCandidateExtractor(engine, "qwen3.5:9b").extract(_exchange())

    messages, kwargs = engine.calls[0]
    response_format = kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["candidates"]
    assert schema["additionalProperties"] is False
    item_schema = schema["properties"]["candidates"]["items"]
    assert item_schema["additionalProperties"] is False
    assert set(item_schema["required"]) == {
        "kind",
        "content",
        "importance",
        "confidence",
        "temporal_scope",
        "subject",
        "target_claim_id",
        "evidence_excerpt",
    }
    assert "current user message" in messages[0].content.lower()


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "[]",
        json.dumps({"candidates": [], "action": "delete_all"}),
        json.dumps({"candidates": [{}]}),
    ],
)
def test_extractor_rejects_malformed_or_schema_invalid_output(response):
    """Malformed output must produce no proposal and a durable retry signal."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine(response), "qwen3.5:9b")

    assert extractor.extract(_exchange()) == []
    assert extractor.last_error_code == "invalid_output"


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "pattern"},
        {"importance": -0.01},
        {"confidence": 1.01},
        {"temporal_scope": "forever"},
        {"content": "x" * 501},
        {"subject": "x" * 121},
        {"target_claim_id": "x" * 121},
        {"evidence_excerpt": "x" * 501},
        {"action": "delete_all"},
    ],
)
def test_extractor_rejects_invalid_enums_bounds_lengths_and_extra_fields(overrides):
    """Only structurally bounded candidates may cross the model boundary."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(
        _FakeEngine(_response(_candidate(**overrides))), "qwen3.5:9b"
    )

    assert extractor.extract(_exchange("I enjoy tea. " + "x" * 600)) == []
    assert extractor.last_error_code == "invalid_output"


def test_extractor_caps_valid_candidates_at_five():
    """A model response cannot create more than five proposals per exchange."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    candidates = [
        _candidate(
            content=f"Tea preference {index}",
            subject=f"user.preference.tea.{index}",
        )
        for index in range(6)
    ]

    drafts = PersonalCandidateExtractor(
        _FakeEngine(_response(*candidates)), "qwen3.5:9b"
    ).extract(_exchange())

    assert len(drafts) == 5


def test_extractor_keeps_the_verified_engine_name_for_candidate_provenance():
    """Persisted candidates must name the validated engine that proposed them."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(
        _FakeEngine(_response()),
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
