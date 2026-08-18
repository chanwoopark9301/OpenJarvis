"""Shared fixtures — clear all registries and the event bus between tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openjarvis.core.config import GpuInfo, HardwareInfo
from openjarvis.core.events import EventBus, reset_event_bus
from openjarvis.core.registry import (
    AgentRegistry,
    BenchmarkRegistry,
    ChannelRegistry,
    CompressionRegistry,
    ConnectorRegistry,
    EngineRegistry,
    FactStoreRegistry,
    MemoryRegistry,
    MinerRegistry,
    ModelRegistry,
    RouterPolicyRegistry,
    SkillRegistry,
    SpeechRegistry,
    ToolRegistry,
    TTSRegistry,
)

_EXTERNAL_TEST_MARKERS = {
    "cloud": "OPENJARVIS_RUN_CLOUD_TESTS",
    "hub": "OPENJARVIS_RUN_HUB_TESTS",
    "live": "OPENJARVIS_RUN_LIVE_TESTS",
}


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Keep the default suite local and deterministic.

    Tests marked ``cloud``, ``hub``, or ``live`` require credentials,
    a network download, or an installed model. They are only run when the
    developer explicitly opts in with the marker's environment variable.
    """
    for item in items:
        for marker, opt_in_var in _EXTERNAL_TEST_MARKERS.items():
            if marker in item.keywords and os.environ.get(opt_in_var) != "1":
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"requires explicit opt-in with {opt_in_var}=1",
                    )
                )
                break


@pytest.fixture(autouse=True)
def _no_update_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the CLI's PyPI update-check nag run during tests.

    ``check_for_updates`` writes its banner to stderr, which ``CliRunner``
    merges into ``result.output`` — polluting JSON/CSV output of any test
    that invokes a CLI command. It already self-disables when ``CI`` is
    set, but that only helps in CI; locally (e.g. a dev with a stale
    version-check cache and network access) it fires for real.
    """
    monkeypatch.setenv("OPENJARVIS_NO_UPDATE_CHECK", "1")


@pytest.fixture(autouse=True)
def _no_external_analytics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests from starting real PostHog workers.

    Server construction normally honours the user's analytics setting.  Tests
    often build an app without running its ASGI lifespan, so a real SDK client
    would never receive the matching shutdown event and could block interpreter
    exit while its network worker drains.  Patch only the test process's server
    enablement boundary; production configuration and client lifecycle stay
    unchanged.
    """
    import openjarvis.analytics as analytics

    monkeypatch.setattr(analytics, "is_analytics_enabled", lambda _config: False)


@pytest.fixture(autouse=True)
def _clean_registries() -> None:
    """Ensure each test starts with empty registries and a fresh event bus."""
    ModelRegistry.clear()
    EngineRegistry.clear()
    MemoryRegistry.clear()
    FactStoreRegistry.clear()
    MinerRegistry.clear()
    AgentRegistry.clear()
    ToolRegistry.clear()
    RouterPolicyRegistry.clear()
    BenchmarkRegistry.clear()
    ChannelRegistry.clear()
    SpeechRegistry.clear()
    CompressionRegistry.clear()
    ConnectorRegistry.clear()
    TTSRegistry.clear()
    SkillRegistry.clear()
    reset_event_bus()


# ---------------------------------------------------------------------------
# Hardware fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nvidia_gpu() -> GpuInfo:
    """NVIDIA A100 GPU fixture."""
    return GpuInfo(vendor="nvidia", name="NVIDIA A100-SXM4-80GB", vram_gb=80.0, count=1)


@pytest.fixture
def nvidia_consumer_gpu() -> GpuInfo:
    """NVIDIA consumer GPU fixture."""
    return GpuInfo(
        vendor="nvidia",
        name="NVIDIA GeForce RTX 4090",
        vram_gb=24.0,
        count=1,
    )


@pytest.fixture
def nvidia_multi_gpu() -> GpuInfo:
    """NVIDIA multi-GPU fixture."""
    return GpuInfo(vendor="nvidia", name="NVIDIA H100", vram_gb=80.0, count=4)


@pytest.fixture
def amd_gpu() -> GpuInfo:
    """AMD MI300X GPU fixture."""
    return GpuInfo(vendor="amd", name="AMD Instinct MI300X", vram_gb=192.0, count=1)


@pytest.fixture
def apple_gpu() -> GpuInfo:
    """Apple Silicon GPU fixture."""
    return GpuInfo(vendor="apple", name="Apple M4 Max", vram_gb=128.0, count=1)


@pytest.fixture
def hardware_nvidia(nvidia_gpu: GpuInfo) -> HardwareInfo:
    """Full NVIDIA hardware profile."""
    return HardwareInfo(
        platform="linux",
        cpu_brand="AMD EPYC 7763",
        cpu_count=64,
        ram_gb=512.0,
        gpu=nvidia_gpu,
    )


@pytest.fixture
def hardware_nvidia_consumer(nvidia_consumer_gpu: GpuInfo) -> HardwareInfo:
    """Consumer NVIDIA hardware profile."""
    return HardwareInfo(
        platform="linux",
        cpu_brand="Intel Core i9-14900K",
        cpu_count=24,
        ram_gb=64.0,
        gpu=nvidia_consumer_gpu,
    )


@pytest.fixture
def hardware_amd(amd_gpu: GpuInfo) -> HardwareInfo:
    """Full AMD hardware profile."""
    return HardwareInfo(
        platform="linux",
        cpu_brand="AMD EPYC 9654",
        cpu_count=96,
        ram_gb=768.0,
        gpu=amd_gpu,
    )


@pytest.fixture
def hardware_apple(apple_gpu: GpuInfo) -> HardwareInfo:
    """Apple Silicon hardware profile."""
    return HardwareInfo(
        platform="darwin",
        cpu_brand="Apple M4 Max",
        cpu_count=16,
        ram_gb=128.0,
        gpu=apple_gpu,
    )


@pytest.fixture
def hardware_cpu_only() -> HardwareInfo:
    """CPU-only hardware profile (no GPU)."""
    return HardwareInfo(
        platform="linux",
        cpu_brand="Intel Xeon E5-2686 v4",
        cpu_count=8,
        ram_gb=32.0,
        gpu=None,
    )


# ---------------------------------------------------------------------------
# Engine availability fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def has_ollama() -> bool:
    """Check if Ollama is running locally."""
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture
def has_vllm() -> bool:
    """Check if vLLM is running locally."""
    try:
        import httpx

        resp = httpx.get("http://localhost:8000/v1/models", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture
def has_llamacpp() -> bool:
    """Check if llama.cpp server is running locally."""
    try:
        import httpx

        resp = httpx.get("http://localhost:8080/v1/models", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Cloud API key fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def has_openai_key() -> bool:
    """Check if OPENAI_API_KEY is set."""
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.fixture
def has_anthropic_key() -> bool:
    """Check if ANTHROPIC_API_KEY is set."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.fixture
def has_gemini_key() -> bool:
    """Check if GEMINI_API_KEY or GOOGLE_API_KEY is set."""
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


# ---------------------------------------------------------------------------
# Mock engine factory
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    """Factory for mock InferenceEngine instances."""

    def _factory(
        engine_id: str = "mock",
        model_response: str = "Hello!",
        tool_calls: list | None = None,
        models: list[str] | None = None,
    ) -> MagicMock:
        engine = MagicMock()
        engine.engine_id = engine_id
        engine.health.return_value = True
        engine.list_models.return_value = models or ["test-model"]

        result = {
            "content": model_response,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        }
        if tool_calls:
            result["tool_calls"] = tool_calls
            result["finish_reason"] = "tool_calls"
        engine.generate.return_value = result
        return engine

    return _factory


@pytest.fixture
def event_bus() -> EventBus:
    """Fresh EventBus with history recording enabled."""
    return EventBus(record_history=True)


# ---------------------------------------------------------------------------
# Mining sidecar fixtures (shared across tests/mining/ and tests/engine/)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_sidecar_payload() -> dict:
    """A valid vllm-pearl sidecar payload with all expected fields."""
    return {
        "provider": "vllm-pearl",
        "vllm_endpoint": "http://127.0.0.1:8000/v1",
        "model": "pearl-ai/Llama-3.3-70B-Instruct-pearl",
        "gateway_url": "http://127.0.0.1:8337",
        "gateway_metrics_url": "http://127.0.0.1:8339",
        "container_id": "abc123def456",
        "wallet_address": "prl1qexampleaddress",
        "started_at": 1714867200,
    }


@pytest.fixture
def sidecar_path(tmp_path: Path) -> Path:
    """Path to a (not-yet-written) mining sidecar JSON file."""
    return tmp_path / "mining.json"


@pytest.fixture
def written_sidecar(sidecar_path: Path, sample_sidecar_payload: dict) -> Path:
    """A written mining sidecar JSON file; returns the path."""
    sidecar_path.write_text(json.dumps(sample_sidecar_payload))
    return sidecar_path
