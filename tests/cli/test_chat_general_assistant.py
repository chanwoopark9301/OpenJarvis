"""Integration tests for the generic assistant path in ``jarvis chat``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from openjarvis.cli._assistant_preflight import AssistantPreflightResult
from openjarvis.cli._search_evidence import EvidenceSource
from openjarvis.cli.chat_cmd import chat
from openjarvis.core.config import JarvisConfig
from openjarvis.core.types import Message, Role


class _FixedPreflight:
    def __init__(self, *results: AssistantPreflightResult):
        self.results = list(results)
        self.calls = []

    def prepare(
        self,
        user_text,
        recent_messages=(),
        *,
        pending_clarification=False,
    ):
        self.calls.append((user_text, tuple(recent_messages), pending_clarification))
        return self.results.pop(0)


def _config() -> JarvisConfig:
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.agent.default_agent = "none"
    config.agent.context_from_memory = False
    config.memory.enabled = False
    config.personal_memory.enabled = False
    return config


def _weather_result() -> AssistantPreflightResult:
    evidence = EvidenceSource(
        "S1",
        "R1",
        "과천시 오늘 날씨",
        "https://weather.example/gwacheon",
        "현재 기온 24도, 저녁에는 흐림",
        "official",
    )
    return AssistantPreflightResult(
        triggered=True,
        mode="search",
        response_style="direct",
        goal="과천시 오늘 날씨 안내",
        success=True,
        queries=("과천시 오늘 날씨",),
        sources=(evidence.url,),
        evidence=(evidence,),
        context="UNTRUSTED WEB SEARCH RESULTS\n[S1] 현재 기온 24도",
        request_text="과천시야.",
    )


def _run(engine, preflight, user_input, *, config=None):
    active_config = config or _config()
    with (
        patch("openjarvis.cli.chat_cmd.load_config", return_value=active_config),
        patch("openjarvis.engine.get_engine", return_value=("mock", engine)),
        patch("openjarvis.intelligence.register_builtin_models"),
        patch(
            "openjarvis.cli.chat_cmd._build_assistant_preflight",
            return_value=preflight,
        ),
    ):
        return CliRunner().invoke(
            chat,
            ["--model", "test-model"],
            input=user_input,
        )


def test_ordinary_chat_keeps_the_single_generation_fast_path():
    engine = MagicMock()
    engine.generate.return_value = {"content": "안녕, 오늘은 어땠어?"}
    preflight = _FixedPreflight(AssistantPreflightResult(triggered=False))

    result = _run(engine, preflight, "안녕?\n/quit\n")

    assert result.exit_code == 0
    assert "오늘은 어땠어?" in result.output
    assert engine.generate.call_count == 1
    assert preflight.calls[0][2] is False


def test_clarification_is_direct_and_only_the_next_turn_is_pending():
    engine = MagicMock()
    engine.generate.return_value = {
        "content": (
            '{"lead":"과천시 날씨를 확인했어.","blocks":['
            '{"kind":"fact","text":"현재 기온은 24도야.","supports":['
            '{"source_id":"S1","excerpt":"현재 기온 24도"}]}],'
            '"follow_up":""}'
        )
    }
    preflight = _FixedPreflight(
        AssistantPreflightResult(
            triggered=True,
            mode="clarify",
            clarifying_question="어느 지역의 날씨를 볼까?",
        ),
        _weather_result(),
    )

    result = _run(
        engine,
        preflight,
        "오늘 날씨를 검색해 줘.\n경기도 과천시야.\n/quit\n",
    )

    assert result.exit_code == 0
    assert "어느 지역의 날씨를 볼까?" in result.output
    assert "현재 기온은 24도야." in result.output
    assert engine.generate.call_count == 1
    assert preflight.calls[0][2] is False
    assert preflight.calls[1][2] is True
    recent = preflight.calls[1][1]
    assert any("어느 지역의 날씨를 볼까?" in message.content for message in recent)


def test_search_uses_generic_prompt_and_renders_only_used_source():
    engine = MagicMock()
    engine.generate.return_value = {
        "content": (
            '{"lead":"확인했어.","blocks":['
            '{"kind":"fact","text":"현재 기온은 24도야.","supports":['
            '{"source_id":"S1","excerpt":"현재 기온 24도"}]},'
            '{"kind":"advice","text":"얇은 겉옷을 챙겨.","supports":[]}],'
            '"follow_up":""}'
        )
    }
    preflight = _FixedPreflight(_weather_result())

    result = _run(engine, preflight, "과천시 오늘 날씨를 검색해 줘.\n/quit\n")

    assert result.exit_code == 0
    assert "현재 기온은 24도야." in result.output
    assert "https://weather.example/gwacheon" in result.output
    messages = engine.generate.call_args.args[0]
    combined = "\n".join(message.content for message in messages)
    assert '"blocks"' in combined
    assert "UNTRUSTED WEB SEARCH RESULTS" in combined
    assert "itinerary" not in combined.casefold()


def test_malformed_search_answer_gets_one_local_repair():
    repaired = (
        '{"lead":"확인했어.","blocks":[{"kind":"fact",'
        '"text":"현재 기온은 24도야.","supports":['
        '{"source_id":"S1","excerpt":"현재 기온 24도"}]}],'
        '"follow_up":""}'
    )
    engine = MagicMock()
    engine.generate.side_effect = [
        {"content": "형식이 잘못된 답"},
        {"content": repaired},
    ]

    result = _run(
        engine,
        _FixedPreflight(_weather_result()),
        "과천시 날씨를 검색해 줘.\n/quit\n",
    )

    assert result.exit_code == 0
    assert "현재 기온은 24도야." in result.output
    assert engine.generate.call_count == 2


def test_action_and_search_failure_skip_final_generation_and_are_archived_once():
    engine = MagicMock()
    action = AssistantPreflightResult(
        triggered=True,
        mode="action",
        error="웹사이트나 앱을 직접 조작하는 기능은 아직 연결되지 않았어.",
    )
    failure = AssistantPreflightResult(
        triggered=True,
        mode="search",
        success=False,
        error="인터넷에서 필요한 정보를 확인하지 못했어.",
    )
    preflight = _FixedPreflight(action, failure)

    with patch(
        "openjarvis.cli.chat_cmd.record_and_publish_completed_exchange"
    ) as record_exchange:
        result = _run(
            engine,
            preflight,
            "쿠팡에서 음료수를 주문해 줘.\n날씨를 검색해 줘.\n/quit\n",
        )

    assert result.exit_code == 0
    assert "직접 조작하는 기능은 아직 연결되지 않았어" in result.output
    assert "필요한 정보를 확인하지 못했어" in result.output
    engine.generate.assert_not_called()
    assert record_exchange.call_count == 2


def test_search_and_personal_memory_context_both_reach_generation():
    engine = MagicMock()
    engine.generate.return_value = {
        "content": (
            '{"lead":"확인했어.","blocks":[{"kind":"fact",'
            '"text":"현재 기온은 24도야.","supports":['
            '{"source_id":"S1","excerpt":"현재 기온 24도"}]}],'
            '"follow_up":""}'
        )
    }
    config = _config()
    config.agent.context_from_memory = True
    memory_message = Message(
        role=Role.SYSTEM,
        content="PERSONAL MEMORY CONTEXT\ndirect user rule",
        metadata={"memory_context": True},
    )

    with (
        patch("openjarvis.memory.build_memory_service", return_value=None),
        patch("openjarvis.cli.ask._get_memory_backend", return_value=None),
        patch(
            "openjarvis.tools.storage.context.inject_context",
            return_value=[memory_message],
        ),
    ):
        result = _run(
            engine,
            _FixedPreflight(_weather_result()),
            "과천시 날씨를 검색해 줘.\n/quit\n",
            config=config,
        )

    assert result.exit_code == 0
    messages = engine.generate.call_args.args[0]
    combined = "\n".join(message.content for message in messages)
    assert "PERSONAL MEMORY CONTEXT" in combined
    assert "UNTRUSTED WEB SEARCH RESULTS" in combined
