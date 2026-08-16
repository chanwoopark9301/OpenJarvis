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


def _exchange(user_text="I enjoy tea."):
    _, _, ConversationExchange = _extractor_api()
    return ConversationExchange(
        id="exchange-1",
        created_at=1.0,
        archived_at=1.0,
        source="test",
        user_text=user_text,
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


def test_extractor_supports_direct_rule_kinds_and_scope():
    """Losing rule fields would turn an explicit ban into an ordinary weak fact."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    engine = _FakeEngine(
        '[{"kind":"constraint","content":"Do not mention timers proactively",'
        '"importance":1,"confidence":1,"temporal_scope":"until_changed",'
        '"subject":"assistant_behavior","target_claim_id":""}]'
    )

    drafts = PersonalCandidateExtractor(engine, "qwen3.5:9b").extract(
        _exchange("Please keep answers short.")
    )

    assert len(drafts) == 1
    assert drafts[0].kind == "constraint"
    assert drafts[0].temporal_scope == "until_changed"
    assert drafts[0].subject == "assistant_behavior"


def test_extractor_rejects_unknown_persistence_fields():
    """A model must not smuggle actions or instructions into stored candidates."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(
        _FakeEngine(
            '[{"kind":"fact","content":"User is tired",'
            '"importance":0.5,"confidence":0.7,"action":"delete_all"}]'
        ),
        "qwen3.5:9b",
    )

    assert extractor.extract(_exchange()) == []
    assert extractor.last_error_code == "invalid_output"


def test_extraction_prompt_marks_assistant_text_as_non_evidence():
    """Assistant suggestions must not reinforce themselves as user evidence."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    engine = _FakeEngine("[]")

    PersonalCandidateExtractor(engine, "qwen3.5:9b").extract(_exchange())

    messages, _ = engine.calls[0]
    assert "Only the user's own words are personal evidence" in messages[0].content
    assert "Never treat assistant text as evidence" in messages[0].content


def test_explicit_user_rule_signal_raises_priority_without_inventing_a_target():
    """A rare direct ban must not need repetition before evaluation gets priority."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    engine = _FakeEngine(
        '[{"kind":"constraint","content":"Do not mention that topic",'
        '"importance":0.2,"confidence":0.8,"temporal_scope":"until_changed",'
        '"subject":"assistant_behavior","target_claim_id":""}]'
    )

    drafts = PersonalCandidateExtractor(engine, "qwen3.5:9b").extract(
        _exchange("그 얘기는 하지 마. 꼭 기억해.")
    )

    assert drafts[0].importance == 1.0
    assert drafts[0].target_claim_id == ""


def test_explicit_korean_prohibition_survives_an_empty_model_result():
    """Deleting the deterministic lane must lose this direct user rule."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    drafts = extractor.extract(_exchange("앞으로 농담을 먼저 꺼내지 마."))

    assert len(drafts) == 1
    assert drafts[0].kind == "constraint"
    assert drafts[0].content == "앞으로 농담을 먼저 꺼내지 마"
    assert drafts[0].importance == 1.0
    assert drafts[0].confidence == 1.0
    assert drafts[0].temporal_scope == "until_changed"
    assert drafts[0].subject == "assistant_behavior"


def test_deterministic_rule_lane_does_not_turn_a_greeting_into_memory():
    """Broadening the rule signals must not promote ordinary conversation."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    assert extractor.extract(_exchange("안녕?")) == []


def test_remembering_an_ordinary_fact_is_not_an_assistant_behavior_rule():
    """A generic memory request must not become an always-on assistant constraint."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    assert extractor.extract(_exchange("내 고향은 부산이야. 기억해.")) == []


@pytest.mark.parametrize(
    "user_text",
    [
        "내 고양이 이름은 나비야. 기억해.",
        "나는 수영할 수 없어. 기억해.",
    ],
)
def test_user_facts_are_not_promoted_by_role_or_capability_words(user_text):
    """First-person facts must never become assistant behavior constraints."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    assert extractor.extract(_exchange(user_text)) == []


def test_direct_rule_stores_only_the_command_clause():
    """Trailing sensitive facts must not ride along in an always-on rule."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    drafts = extractor.extract(
        _exchange("앞으로 농담하지 마. 내 건강 기록은 당뇨야.")
    )

    assert [draft.content for draft in drafts] == ["앞으로 농담하지 마"]


def test_positive_assistant_instruction_survives_an_empty_model_result():
    """Explicit positive behavior rules need the same deterministic protection."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    drafts = extractor.extract(_exchange("항상 존댓말로 대답해 줘."))

    assert len(drafts) == 1
    assert drafts[0].kind == "constraint"
    assert drafts[0].content == "항상 존댓말로 대답해 줘"


@pytest.mark.parametrize(
    "user_text",
    [
        "I do not like coffee. Remember this.",
        "나는 매일 친구와 대화해. 기억해.",
    ],
)
def test_ordinary_statements_are_not_promoted_to_behavior_rules(user_text):
    """First-person statements must not become always-on assistant commands."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    assert extractor.extract(_exchange(user_text)) == []


def test_durable_english_imperative_survives_an_empty_model_result():
    """An explicitly durable rule must not depend on model-returned JSON."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    drafts = extractor.extract(_exchange("Always answer in English."))

    assert len(drafts) == 1
    assert drafts[0].kind == "constraint"
    assert drafts[0].content == "Always answer in English"


@pytest.mark.parametrize(
    "user_text",
    [
        "Explain this code.",
        "Reply to this email.",
        "이 코드를 설명해 줘.",
    ],
)
def test_one_turn_requests_are_not_stored_as_durable_rules(user_text):
    """A request for this turn must not alter later conversations."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    assert extractor.extract(_exchange(user_text)) == []


@pytest.mark.parametrize(
    ("user_text", "expected_kind"),
    [
        ("Never mention politics.", "constraint"),
        ("Call me Alex.", "role_preference"),
        ("존댓말로 답변해주세요.", "constraint"),
    ],
)
def test_common_standing_rules_survive_an_empty_model_result(
    user_text, expected_kind
):
    """Frequent durable rule forms need deterministic protection."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    drafts = extractor.extract(_exchange(user_text))

    assert len(drafts) == 1
    assert drafts[0].kind == expected_kind


@pytest.mark.parametrize(
    "user_text",
    [
        "Remember this: I like coffee. Explain this code.",
        "앞으로 내 고향은 부산이야. 이 코드를 설명해 줘.",
    ],
)
def test_durability_marker_does_not_leak_into_later_commands(user_text):
    """A durable marker may not make an unrelated next action permanent."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    assert extractor.extract(_exchange(user_text)) == []


@pytest.mark.parametrize(
    "user_text",
    [
        "네 역할은 뭐야?",
        "Your role is what?",
        "너는 실행할 수 없어?",
    ],
)
def test_questions_are_not_stored_as_role_or_capability_rules(user_text):
    """Asking what is true must not be interpreted as assigning a rule."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    assert extractor.extract(_exchange(user_text)) == []


@pytest.mark.parametrize(
    "user_text",
    [
        "Do not mention politics. Please remember this.",
        "그 얘기는 하지 마. 꼭 기억해 주세요.",
    ],
)
def test_polite_memory_marker_makes_the_previous_rule_durable(user_text):
    """Politeness must not make an explicit standing rule disappear."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    drafts = extractor.extract(_exchange(user_text))

    assert len(drafts) == 1
    assert drafts[0].kind == "constraint"


@pytest.mark.parametrize(
    "user_text",
    [
        "Your role is confusing.",
        "네 이름은 이상해.",
        "네 역할은 중요해.",
    ],
)
def test_role_evaluations_are_not_stored_as_role_assignments(user_text):
    """Describing a role must not be mistaken for assigning a new one."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    assert extractor.extract(_exchange(user_text)) == []


@pytest.mark.parametrize(
    "user_text",
    [
        "너의 이름은 이제 조visor야.",
        "이제부터 네 역할은 말동무야.",
        "From now on, your role is companion.",
    ],
)
def test_explicit_role_assignments_are_stored(user_text):
    """Tightening role detection must preserve real assignments."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    extractor = PersonalCandidateExtractor(_FakeEngine("[]"), "qwen3.5:9b")

    drafts = extractor.extract(_exchange(user_text))

    assert len(drafts) == 1
    assert drafts[0].kind == "role_preference"


def test_model_cannot_promote_an_unverified_fact_to_a_direct_rule():
    """The model may propose facts, but explicit behavior must be deterministic."""
    PersonalCandidateExtractor, _, _ = _extractor_api()
    engine = _FakeEngine(
        '[{"kind":"constraint","content":"User dislikes coffee",'
        '"importance":1,"confidence":1,"temporal_scope":"until_changed",'
        '"subject":"assistant_behavior","target_claim_id":""}]'
    )

    drafts = PersonalCandidateExtractor(engine, "qwen3.5:9b").extract(
        _exchange("I do not like coffee. Remember this.")
    )

    assert drafts == []


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
