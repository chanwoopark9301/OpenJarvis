"""Tests for the web search tool."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.core.registry import ToolRegistry
from openjarvis.tools.web_search import WebSearchTool


def mock_geocoder_and_forecast(monkeypatch, *, place: str, temperature: float):
    geocoding_response = MagicMock()
    geocoding_response.raise_for_status.return_value = None
    geocoding_response.json.return_value = [
        {
            "name": place,
            "display_name": f"{place}, 경기도, 대한민국",
            "lat": "37.42889",
            "lon": "126.98917",
        }
    ]
    weather_response = MagicMock()
    weather_response.raise_for_status.return_value = None
    weather_response.json.return_value = {
        "current_units": {
            "temperature_2m": "°C",
            "apparent_temperature": "°C",
            "precipitation": "mm",
            "wind_speed_10m": "km/h",
        },
        "current": {
            "time": "2026-08-18T20:15",
            "temperature_2m": temperature,
            "apparent_temperature": 29.4,
            "precipitation": 0.0,
            "weather_code": 2,
            "wind_speed_10m": 8.2,
        },
    }
    monkeypatch.setattr(
        "httpx.get", MagicMock(side_effect=[geocoding_response, weather_response])
    )
    monkeypatch.setattr(
        WebSearchTool,
        "_duckduckgo_search",
        MagicMock(
            return_value=(
                "### fallback\nSource: https://example.com\nSummary: link snippet"
            )
        ),
    )


class TestWebSearchTool:
    def test_spec_name_and_category(self):
        tool = WebSearchTool(api_key="test-key")
        assert tool.spec.name == "web_search"
        assert tool.spec.category == "search"

    def test_spec_requires_api_key_metadata(self):
        tool = WebSearchTool(api_key="test-key")
        assert tool.spec.metadata["requires_api_key"] == "TAVILY_API_KEY"

    def test_spec_parameters_require_query(self):
        tool = WebSearchTool(api_key="test-key")
        assert "query" in tool.spec.parameters["properties"]
        assert "query" in tool.spec.parameters["required"]

    def test_execute_no_query(self):
        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="")
        assert result.success is False
        assert "No query" in result.content
        assert result.metadata["mode"] == "search"
        assert result.metadata["engine"] == "tavily"
        assert result.metadata["content_available"] is False
        assert result.metadata["query"] == ""
        assert result.metadata["retrieved_at"]

    def test_execute_no_query_param(self):
        tool = WebSearchTool(api_key="test-key")
        result = tool.execute()
        assert result.success is False
        assert "No query" in result.content

    def test_korean_weather_without_administrative_suffix_uses_structured_data(
        self, monkeypatch
    ):
        mock_geocoder_and_forecast(monkeypatch, place="과천시", temperature=24.7)
        result = WebSearchTool().execute(query="과천 오늘 날씨")
        assert result.metadata["engine"] == "open-meteo"
        assert "기온 24.7°C" in result.content

    def test_model_generated_korean_weather_connective_uses_structured_data(
        self, monkeypatch
    ):
        mock_geocoder_and_forecast(monkeypatch, place="과천시", temperature=24.7)
        result = WebSearchTool().execute(query="과천시 오늘의 날씨")
        assert result.metadata["engine"] == "open-meteo"
        assert "기온 24.7°C" in result.content

    def test_spec_tells_model_it_can_fetch_a_result_url(self):
        description = WebSearchTool().spec.description
        assert "public URL" in description
        assert "fetch" in description

    def test_korean_current_weather_uses_structured_public_data(self, monkeypatch):
        geocoding_response = MagicMock()
        geocoding_response.raise_for_status.return_value = None
        geocoding_response.json.return_value = [
            {
                "name": "과천시",
                "display_name": "과천시, 경기도, 대한민국",
                "lat": "37.42889",
                "lon": "126.98917",
            }
        ]
        weather_response = MagicMock()
        weather_response.raise_for_status.return_value = None
        weather_response.json.return_value = {
            "current_units": {
                "temperature_2m": "°C",
                "apparent_temperature": "°C",
                "precipitation": "mm",
                "wind_speed_10m": "km/h",
            },
            "current": {
                "time": "2026-08-18T20:15",
                "temperature_2m": 24.7,
                "apparent_temperature": 29.4,
                "precipitation": 0.0,
                "weather_code": 2,
                "wind_speed_10m": 8.2,
            },
        }
        get = MagicMock(side_effect=[geocoding_response, weather_response])
        monkeypatch.setattr("httpx.get", get)

        result = WebSearchTool(api_key=None).execute(
            query="경기도 과천시 현재 날씨 기온 강수"
        )

        assert result.success is True
        assert result.metadata["engine"] == "open-meteo"
        assert result.metadata["mode"] == "search"
        assert result.metadata["content_available"] is True
        assert result.metadata["retrieved_at"]
        assert result.metadata["source"].startswith("https://api.open-meteo.com/")
        assert result.metadata["query"] == "경기도 과천시 현재 날씨 기온 강수"
        assert "경기도 과천시 현재 날씨" in result.content
        assert "기온 24.7°C" in result.content
        assert "체감 기온 29.4°C" in result.content
        assert "구름 조금" in result.content
        assert "Source: https://api.open-meteo.com/" in result.content
        assert get.call_count == 2

    def test_structured_weather_failure_falls_back_to_normal_search(self, monkeypatch):
        monkeypatch.setattr("httpx.get", MagicMock(side_effect=OSError("offline")))
        monkeypatch.setattr(
            WebSearchTool,
            "_duckduckgo_search",
            MagicMock(
                return_value=("### fallback\nSource: https://example.com\nSummary: ok")
            ),
        )

        result = WebSearchTool(api_key=None).execute(
            query="경기도 과천시 현재 날씨 기온 강수"
        )

        assert result.success is True
        assert result.metadata["engine"] == "duckduckgo"

    def test_execute_no_api_key(self, monkeypatch):
        """When no API key, falls back to DuckDuckGo."""
        tool = WebSearchTool(api_key=None)
        monkeypatch.setattr(
            tool,
            "_duckduckgo_search",
            MagicMock(
                return_value="### fallback\nSource: https://example.com\nSummary: ok"
            ),
        )
        with patch.dict("os.environ", {}, clear=True):
            tool._api_key = None
            monkeypatch.delitem(sys.modules, "tavily", raising=False)
            result = tool.execute(query="test query")
        assert result.success is True
        assert result.metadata["engine"] == "duckduckgo"

    def test_execute_mocked_tavily(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "content": "Content about test.",
                },
                {
                    "title": "Result 2",
                    "url": "https://example.com/2",
                    "content": "More content.",
                },
            ]
        }
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client

        import builtins

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "tavily":
                return mock_tavily_module
            if name == "tavily.errors":
                mock_errors = MagicMock()
                mock_errors.UsageLimitExceededError = Exception
                return mock_errors
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        monkeypatch.setattr(
            WebSearchTool,
            "_duckduckgo_search",
            MagicMock(
                return_value="### fallback\nSource: https://example.com\nSummary: ok"
            ),
        )

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="test query")
        assert result.success is True
        assert "Result 1" in result.content
        assert "Result 2" in result.content
        assert result.metadata["num_results"] == 2
        assert result.metadata["mode"] == "search"
        assert result.metadata["engine"] == "tavily"
        assert result.metadata["content_available"] is True
        assert result.metadata["query"] == "test query"
        assert result.metadata["retrieved_at"]
        assert result.content.startswith("UNTRUSTED PUBLIC WEB DATA.")

    def test_execute_tavily_error(self, monkeypatch):
        """When Tavily errors (any error), falls back to DuckDuckGo."""
        import builtins
        from typing import Any

        original_import = builtins.__import__

        class TavilyError(Exception):
            def __init__(self, message: str):
                super().__init__(message)

        mock_client = MagicMock()
        mock_client.search.side_effect = TavilyError("API error")
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client

        def _mock_import(name: str, *args: Any, **kwargs: Any):
            if name == "tavily":
                return mock_tavily_module
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        monkeypatch.setattr(
            WebSearchTool,
            "_duckduckgo_search",
            MagicMock(
                return_value="### fallback\nSource: https://example.com\nSummary: ok"
            ),
        )

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="test query")
        assert result.success is True
        assert result.metadata["engine"] == "duckduckgo"

    def test_execute_duckduckgo_fallback_format(self, monkeypatch):
        """DuckDuckGo fallback returns properly formatted results."""
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.side_effect = ImportError(
            "No module named 'tavily'"
        )
        monkeypatch.setitem(sys.modules, "tavily", mock_tavily_module)

        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {
                "title": "DDG Result 1",
                "href": "https://example.com/1",
                "body": "Content 1",
            },
            {
                "title": "DDG Result 2",
                "href": "https://example.com/2",
                "body": "Content 2",
            },
        ]
        mock_ddgs_module = MagicMock()
        mock_ddgs_module.DDGS.return_value = mock_ddgs
        monkeypatch.setitem(sys.modules, "ddgs", mock_ddgs_module)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="test query")
        assert result.success is True
        assert "DDG Result 1" in result.content
        assert "DDG Result 2" in result.content
        assert "https://example.com/1" in result.content
        assert result.metadata["engine"] == "duckduckgo"
        assert result.metadata["mode"] == "search"
        assert result.metadata["content_available"] is True
        assert result.metadata["query"] == "test query"
        assert result.metadata["retrieved_at"]
        assert result.content.startswith("UNTRUSTED PUBLIC WEB DATA.")

    def test_duckduckgo_uses_korean_region_for_korean_query(self, monkeypatch):
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = []
        mock_ddgs_module = MagicMock()
        mock_ddgs_module.DDGS.return_value = mock_ddgs
        monkeypatch.setitem(sys.modules, "ddgs", mock_ddgs_module)

        WebSearchTool(api_key=None)._duckduckgo_search("과천시 현재 날씨", 5)

        mock_ddgs.text.assert_called_once_with(
            "과천시 현재 날씨",
            max_results=5,
            region="kr-kr",
        )

    def test_duckduckgo_keeps_default_region_for_non_korean_query(self, monkeypatch):
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = []
        mock_ddgs_module = MagicMock()
        mock_ddgs_module.DDGS.return_value = mock_ddgs
        monkeypatch.setitem(sys.modules, "ddgs", mock_ddgs_module)

        WebSearchTool(api_key=None)._duckduckgo_search("current weather", 5)

        mock_ddgs.text.assert_called_once_with("current weather", max_results=5)

    def test_max_results_parameter(self, monkeypatch):
        import builtins

        original_import = builtins.__import__

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client
        mock_errors = MagicMock()

        def _mock_import(name, *args, **kwargs):
            if name == "tavily":
                return mock_tavily_module
            if name == "tavily.errors":
                return mock_errors
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        tool = WebSearchTool(api_key="test-key", max_results=3)
        tool.execute(query="test", max_results=7)
        mock_client.search.assert_called_once_with(
            "test", max_results=7, search_depth="advanced", include_usage=True
        )

    def test_to_openai_function(self):
        tool = WebSearchTool(api_key="test-key")
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "web_search"
        assert "query" in fn["function"]["parameters"]["properties"]

    def test_execute_import_error(self, monkeypatch):
        """When tavily-python not installed, falls back to DuckDuckGo."""
        monkeypatch.delitem(sys.modules, "tavily", raising=False)
        import builtins

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "tavily":
                raise ImportError("No module named 'tavily'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        monkeypatch.setattr(
            WebSearchTool,
            "_duckduckgo_search",
            MagicMock(
                return_value="### fallback\nSource: https://example.com\nSummary: ok"
            ),
        )

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="test query")
        assert result.success is True
        assert result.metadata["engine"] == "duckduckgo"

    def test_empty_results(self, monkeypatch):
        import builtins

        original_import = builtins.__import__

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client
        mock_errors = MagicMock()

        def _mock_import(name, *args, **kwargs):
            if name == "tavily":
                return mock_tavily_module
            if name == "tavily.errors":
                return mock_errors
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="obscure query")
        assert result.success is True
        assert result.content == "No results found."
        assert result.metadata["mode"] == "search"
        assert result.metadata["engine"] == "tavily"
        assert result.metadata["content_available"] is False
        assert result.metadata["query"] == "obscure query"
        assert result.metadata["retrieved_at"]

    def test_tool_id(self):
        tool = WebSearchTool(api_key="test-key")
        assert tool.tool_id == "web_search"

    def test_registry_registration(self):
        ToolRegistry.register_value("web_search", WebSearchTool)
        assert ToolRegistry.contains("web_search")

    def test_tavily_results_use_labeled_content_format(self, monkeypatch):
        """Regression for #390: results expose page CONTENT under labeled
        Source/Summary headings (so agents synthesize content, not echo
        URLs), and Tavily is queried with search_depth='advanced'."""
        import builtins

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/1",
                    "content": "Content about test.",
                },
            ]
        }
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client
        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "tavily":
                return mock_tavily_module
            if name == "tavily.errors":
                mock_errors = MagicMock()
                mock_errors.UsageLimitExceededError = Exception
                return mock_errors
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="test query")
        assert result.success is True
        # Labeled structure with the page content surfaced.
        assert "### Result 1" in result.content
        assert "Source: https://example.com/1" in result.content
        assert "Summary: Content about test." in result.content
        # search_depth='advanced' is what pulls richer content from Tavily.
        _, kwargs = mock_client.search.call_args
        assert kwargs.get("search_depth") == "advanced"

    def test_tavily_falls_back_to_snippet_when_no_content(self, monkeypatch):
        """When a Tavily result lacks 'content', the 'snippet' field is used
        for the Summary rather than rendering an empty summary."""
        import builtins

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Snippet Only",
                    "url": "https://example.com/s",
                    "snippet": "Fallback snippet text.",
                },
            ]
        }
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client
        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "tavily":
                return mock_tavily_module
            if name == "tavily.errors":
                mock_errors = MagicMock()
                mock_errors.UsageLimitExceededError = Exception
                return mock_errors
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="test query")
        assert "Summary: Fallback snippet text." in result.content

    def test_tavily_ignores_blank_provider_records(self, monkeypatch):
        import builtins

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": [{}, {"title": "  "}]}
        mock_tavily_module = MagicMock()
        mock_tavily_module.TavilyClient.return_value = mock_client
        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "tavily":
                return mock_tavily_module
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        result = WebSearchTool(api_key="test-key").execute(query="blank records")

        assert result.content == "No results found."
        assert result.metadata["num_results"] == 0
        assert result.metadata["content_available"] is False
        assert result.metadata["query"] == "blank records"

    def test_non_location_weather_queries_skip_structured_weather_lookup(self):
        from openjarvis.tools.web_search import _KOREAN_WEATHER_QUERY

        for query in ("내일 날씨", "오늘 날씨", "이번 주 날씨", "전국 날씨"):
            assert _KOREAN_WEATHER_QUERY.search(query) is None


# ---------------------------------------------------------------------------
# URL detection and fetching tests
# ---------------------------------------------------------------------------


class TestUrlDetection:
    def test_is_url_https(self):
        assert WebSearchTool._is_url("https://example.com") is True

    def test_is_url_http(self):
        assert WebSearchTool._is_url("http://example.com") is True

    def test_is_url_with_whitespace(self):
        assert WebSearchTool._is_url("  https://example.com  ") is True

    def test_is_url_plain_text(self):
        assert WebSearchTool._is_url("what are punic wars") is False

    def test_is_url_empty(self):
        assert WebSearchTool._is_url("") is False

    def test_extract_url_from_text(self):
        url = WebSearchTool._extract_url(
            "Summarize this: https://example.com/page please"
        )
        assert url == "https://example.com/page"

    def test_extract_url_none_when_absent(self):
        assert WebSearchTool._extract_url("no urls here") is None

    def test_extract_url_strips_trailing_punctuation(self):
        url = WebSearchTool._extract_url("See https://example.com/page.")
        assert url == "https://example.com/page"

    def test_extract_url_from_complex_text(self):
        url = WebSearchTool._extract_url(
            "Read https://arxiv.org/abs/2310.03714 and summarize"
        )
        assert url == "https://arxiv.org/abs/2310.03714"


class TestUrlNormalization:
    def test_arxiv_pdf_to_abs(self):
        url = WebSearchTool._normalize_url("https://arxiv.org/pdf/2310.03714")
        assert url == "https://arxiv.org/abs/2310.03714"

    def test_arxiv_pdf_with_extension(self):
        url = WebSearchTool._normalize_url("https://arxiv.org/pdf/2310.03714.pdf")
        assert url == "https://arxiv.org/abs/2310.03714"

    def test_non_arxiv_unchanged(self):
        url = WebSearchTool._normalize_url("https://example.com/page")
        assert url == "https://example.com/page"

    def test_arxiv_abs_unchanged(self):
        url = WebSearchTool._normalize_url("https://arxiv.org/abs/2310.03714")
        assert url == "https://arxiv.org/abs/2310.03714"


class TestUrlFetching:
    def _mock_ssrf(self, monkeypatch):
        """Stub out the SSRF check (requires Rust backend)."""
        import openjarvis.tools.web_search as _ws

        monkeypatch.setattr(_ws, "check_ssrf", lambda url: None)

    def test_fetch_url_success(self, monkeypatch):
        """Mocked HTTP GET returns HTML, stripped to text."""
        import httpx

        self._mock_ssrf(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><p>Hello world</p></body></html>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr(httpx, "get", MagicMock(return_value=mock_resp))

        content = WebSearchTool._fetch_url("https://example.com")
        assert "Hello world" in content

    def test_fetch_url_strips_scripts(self, monkeypatch):
        import httpx

        self._mock_ssrf(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.text = "<html><script>var x=1;</script><body>Content</body></html>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr(httpx, "get", MagicMock(return_value=mock_resp))

        content = WebSearchTool._fetch_url("https://example.com")
        assert "var x" not in content
        assert "Content" in content

    def test_fetch_url_truncates_long_content(self, monkeypatch):
        import httpx

        self._mock_ssrf(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.text = "<p>" + "x" * 10000 + "</p>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr(httpx, "get", MagicMock(return_value=mock_resp))

        content = WebSearchTool._fetch_url("https://example.com", max_chars=100)
        assert len(content) < 200
        assert "[Content truncated]" in content

    def test_fetch_url_pdf_content_type(self, monkeypatch):
        import httpx

        self._mock_ssrf(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.text = "%PDF-1.4 binary data"
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr(httpx, "get", MagicMock(return_value=mock_resp))

        content = WebSearchTool._fetch_url("https://example.com/file.pdf")
        assert "PDF" in content
        assert "cannot be read" in content

    def test_fetch_url_blocks_private_redirect_target(self, monkeypatch):
        import httpx

        import openjarvis.tools.web_search as web_search

        redirect = MagicMock()
        redirect.status_code = 302
        redirect.headers = {
            "location": "http://127.0.0.1/admin",
            "content-type": "text/html",
        }
        redirect.url = httpx.URL("https://public.example/start")
        redirect.text = "<html><body>Not a redirect target</body></html>"
        monkeypatch.setattr(httpx, "get", MagicMock(return_value=redirect))
        monkeypatch.setattr(
            web_search,
            "check_ssrf",
            MagicMock(side_effect=[None, "private IP blocked"]),
        )

        with pytest.raises(ValueError, match="private IP blocked"):
            WebSearchTool._fetch_url("https://public.example/start")

        assert httpx.get.call_count == 1

    def test_fetch_url_follows_validated_public_redirect(self, monkeypatch):
        import httpx

        import openjarvis.tools.web_search as web_search

        redirect = MagicMock()
        redirect.status_code = 302
        redirect.headers = {"location": "/article"}
        redirect.url = httpx.URL("https://public.example/start")
        final = MagicMock()
        final.status_code = 200
        final.headers = {"content-type": "text/html"}
        final.url = httpx.URL("https://public.example/article")
        final.text = "<html><body>Public article</body></html>"
        final.raise_for_status = MagicMock()
        get = MagicMock(side_effect=[redirect, final])
        ssrf = MagicMock(return_value=None)
        monkeypatch.setattr(httpx, "get", get)
        monkeypatch.setattr(web_search, "check_ssrf", ssrf)

        content = WebSearchTool._fetch_url("https://public.example/start")

        assert "Public article" in content
        assert get.call_count == 2
        assert get.call_args_list[0].kwargs["follow_redirects"] is False
        assert ssrf.call_args_list == [
            (("https://public.example/start",), {}),
            (("https://public.example/article",), {}),
        ]


class TestExecuteWithUrl:
    def _mock_ssrf(self, monkeypatch):
        """Stub out the SSRF check (requires Rust backend)."""
        import openjarvis.tools.web_search as _ws

        monkeypatch.setattr(_ws, "check_ssrf", lambda url: None)

    def test_execute_with_url_query(self, monkeypatch):
        """When query is a URL, fetch instead of search."""
        import httpx

        self._mock_ssrf(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>Page content here</body></html>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr(httpx, "get", MagicMock(return_value=mock_resp))

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="https://example.com/article")
        assert result.success is True
        assert "Page content here" in result.content
        assert result.metadata.get("mode") == "fetch"
        assert result.metadata["engine"] == "http"
        assert result.metadata["source"] == "https://example.com/article"
        assert result.metadata["query"] == "https://example.com/article"
        assert result.metadata["content_available"] is True
        assert result.metadata["retrieved_at"]
        assert result.content.startswith("UNTRUSTED PUBLIC WEB DATA.")

    def test_execute_with_normalized_url_uses_actual_source(self, monkeypatch):
        import httpx

        self._mock_ssrf(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = httpx.URL("https://arxiv.org/abs/2310.03714")
        mock_resp.text = "<html><body>Paper abstract</body></html>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        get = MagicMock(return_value=mock_resp)
        monkeypatch.setattr(httpx, "get", get)

        query = "https://arxiv.org/pdf/2310.03714"
        result = WebSearchTool(api_key="test-key").execute(query=query)

        assert result.success is True
        assert result.metadata["url"] == query
        assert result.metadata["query"] == query
        assert result.metadata["source"] == "https://arxiv.org/abs/2310.03714"
        assert get.call_args.args[0] == "https://arxiv.org/abs/2310.03714"

    def test_execute_with_embedded_url(self, monkeypatch):
        """When query contains a URL within text, detect and fetch it."""
        import httpx

        self._mock_ssrf(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>Article text</body></html>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr(httpx, "get", MagicMock(return_value=mock_resp))

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="Summarize https://example.com/article please")
        assert result.success is True
        assert result.metadata.get("mode") == "fetch"

    def test_execute_url_ssrf_blocked(self, monkeypatch):
        """SSRF check rejects unsafe URLs before any HTTP request."""
        import openjarvis.tools.web_search as _ws

        monkeypatch.setattr(
            _ws,
            "check_ssrf",
            lambda url: "private IP blocked",
        )

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="http://169.254.169.254/metadata")
        assert result.success is False
        assert "private IP blocked" in result.content
        assert result.metadata["mode"] == "fetch"
        assert result.metadata["engine"] == "http"
        assert result.metadata["content_available"] is False
        assert result.metadata["query"] == "http://169.254.169.254/metadata"
        assert result.metadata["retrieved_at"]

    def test_execute_url_fetch_failure(self, monkeypatch):
        """URL fetch failure returns error result."""
        import httpx

        self._mock_ssrf(monkeypatch)
        monkeypatch.setattr(
            httpx,
            "get",
            MagicMock(side_effect=httpx.HTTPError("Connection failed")),
        )

        tool = WebSearchTool(api_key="test-key")
        result = tool.execute(query="https://example.com/broken")
        assert result.success is False
        assert "Failed to fetch URL" in result.content
        assert result.metadata["mode"] == "fetch"
        assert result.metadata["engine"] == "http"
        assert result.metadata["content_available"] is False
        assert result.metadata["query"] == "https://example.com/broken"
        assert result.metadata["retrieved_at"]

    def test_execute_empty_fetched_page_is_not_available_content(self, monkeypatch):
        import httpx

        self._mock_ssrf(monkeypatch)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.url = httpx.URL("https://example.com/empty")
        mock_resp.text = "<html><body>   </body></html>"
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr(httpx, "get", MagicMock(return_value=mock_resp))

        result = WebSearchTool(api_key="test-key").execute(
            query="https://example.com/empty"
        )

        assert result.success is True
        assert result.content == "No content found at URL."
        assert result.metadata["content_available"] is False
        assert result.metadata["query"] == "https://example.com/empty"
