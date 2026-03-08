"""Tests for OpenAILLMProvider and AnthropicLLMProvider."""

import json
import os
from unittest.mock import patch, MagicMock

import httpx
import pytest

from src.config import Config, LLMConfig
from src.understanding.llm_provider import (
    AnthropicLLMProvider,
    OpenAILLMProvider,
    get_llm_provider,
)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class TestOpenAILLMProviderInit:
    def test_requires_api_key_at_call_time(self):
        config = LLMConfig(provider="openai", model="gpt-4o")
        provider = OpenAILLMProvider(config)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                provider.generate("Hello")

    def test_init_with_api_key(self):
        config = LLMConfig(provider="openai", model="gpt-4o")
        provider = OpenAILLMProvider(config)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            assert provider.api_key == "sk-test123"
            assert provider.base_url == "https://api.openai.com/v1"
            assert provider.timeout == 300

    def test_custom_base_url(self):
        config = LLMConfig(provider="openai", model="gpt-4o")
        provider = OpenAILLMProvider(config)
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "https://custom.api.com/v1/",
        }):
            assert provider.base_url == "https://custom.api.com/v1"

    def test_custom_timeout(self):
        config = LLMConfig(provider="openai", model="gpt-4o")
        provider = OpenAILLMProvider(config, timeout=60)
        assert provider.timeout == 60


class TestOpenAIGenerate:
    @pytest.fixture(autouse=True)
    def _set_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            yield

    @pytest.fixture
    def provider(self):
        config = LLMConfig(provider="openai", model="gpt-4o", max_tokens=1024, temperature=0.5)
        return OpenAILLMProvider(config)

    @patch("httpx.post")
    def test_generate_success(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "Hello back!"}}]
            },
        )

        result = provider.generate("Hello")
        assert result == "Hello back!"

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["model"] == "gpt-4o"
        assert body["max_tokens"] == 1024
        assert body["temperature"] == 0.5
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    @patch("httpx.post")
    def test_generate_with_system_prompt(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "response"}}]
            },
        )

        provider.generate("Hello", system_prompt="Be helpful")

        body = mock_post.call_args.kwargs["json"]
        assert body["messages"] == [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]

    @patch("httpx.post")
    def test_generate_api_error(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=429,
            text="Rate limited",
        )

        with pytest.raises(RuntimeError, match="OpenAI API error.*429"):
            provider.generate("Hello")

    @patch("httpx.post")
    def test_generate_sends_correct_url(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        )

        provider.generate("Hello")

        assert mock_post.call_args.args[0] == "https://api.openai.com/v1/chat/completions"

    @patch("httpx.post")
    def test_generate_sends_auth_header(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        )

        provider.generate("Hello")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-test"


class TestOpenAIGenerateJson:
    @pytest.fixture(autouse=True)
    def _set_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            yield

    @pytest.fixture
    def provider(self):
        config = LLMConfig(provider="openai", model="gpt-4o")
        return OpenAILLMProvider(config)

    @patch("httpx.post")
    def test_generate_json_success(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"result": "ok"}'}}]
            },
        )

        result = provider.generate_json("Return JSON")
        assert result == {"result": "ok"}

    @patch("httpx.post")
    def test_generate_json_requests_json_format(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"a": 1}'}}]
            },
        )

        provider.generate_json("Get data")

        body = mock_post.call_args.kwargs["json"]
        assert body["response_format"] == {"type": "json_object"}

    @patch("httpx.post")
    def test_generate_json_appends_json_instruction(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '{"a": 1}'}}]
            },
        )

        provider.generate_json("Analyze this")

        body = mock_post.call_args.kwargs["json"]
        prompt = body["messages"][-1]["content"]
        assert "JSON" in prompt

    @patch("httpx.post")
    def test_generate_json_parses_markdown_wrapped(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": '```json\n{"data": [1,2]}\n```'}}]
            },
        )

        result = provider.generate_json("Get data")
        assert result == {"data": [1, 2]}

    @patch("httpx.post")
    def test_generate_json_invalid_json_raises(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "not json at all"}}]
            },
        )

        with pytest.raises(RuntimeError, match="Failed to parse JSON"):
            provider.generate_json("Get data")


class TestOpenAIParseJson:
    def test_plain_json(self):
        assert OpenAILLMProvider._parse_json('{"a": 1}') == {"a": 1}

    def test_json_in_markdown_block(self):
        text = '```json\n{"key": "val"}\n```'
        assert OpenAILLMProvider._parse_json(text) == {"key": "val"}

    def test_json_array(self):
        assert OpenAILLMProvider._parse_json('[1, 2, 3]') == [1, 2, 3]

    def test_json_surrounded_by_text(self):
        text = 'Here is the result:\n{"status": "ok"}\nDone.'
        assert OpenAILLMProvider._parse_json(text) == {"status": "ok"}

    def test_invalid_json(self):
        with pytest.raises(RuntimeError, match="Failed to parse JSON"):
            OpenAILLMProvider._parse_json("no json here")


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class TestAnthropicLLMProviderInit:
    def test_requires_api_key_at_call_time(self):
        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        provider = AnthropicLLMProvider(config)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                provider.generate("Hello")

    def test_init_with_api_key(self):
        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        provider = AnthropicLLMProvider(config)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            assert provider.api_key == "sk-ant-test"
            assert provider.base_url == "https://api.anthropic.com"

    def test_custom_base_url(self):
        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        provider = AnthropicLLMProvider(config)
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "ANTHROPIC_BASE_URL": "https://custom.anthropic.com/",
        }):
            assert provider.base_url == "https://custom.anthropic.com"


class TestAnthropicGenerate:
    @pytest.fixture(autouse=True)
    def _set_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            yield

    @pytest.fixture
    def provider(self):
        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            temperature=0.7,
        )
        return AnthropicLLMProvider(config)

    @patch("httpx.post")
    def test_generate_success(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "content": [{"type": "text", "text": "Hello from Claude!"}]
            },
        )

        result = provider.generate("Hello")
        assert result == "Hello from Claude!"

    @patch("httpx.post")
    def test_generate_sends_correct_headers(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"content": [{"type": "text", "text": "ok"}]},
        )

        provider.generate("Hello")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "sk-ant-test"
        assert headers["anthropic-version"] == "2023-06-01"

    @patch("httpx.post")
    def test_generate_sends_correct_url(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"content": [{"type": "text", "text": "ok"}]},
        )

        provider.generate("Hello")

        assert mock_post.call_args.args[0] == "https://api.anthropic.com/v1/messages"

    @patch("httpx.post")
    def test_generate_with_system_prompt(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"content": [{"type": "text", "text": "ok"}]},
        )

        provider.generate("Hello", system_prompt="Be concise")

        body = mock_post.call_args.kwargs["json"]
        assert body["system"] == "Be concise"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

    @patch("httpx.post")
    def test_generate_without_system_prompt(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"content": [{"type": "text", "text": "ok"}]},
        )

        provider.generate("Hello")

        body = mock_post.call_args.kwargs["json"]
        assert "system" not in body

    @patch("httpx.post")
    def test_generate_sends_model_config(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"content": [{"type": "text", "text": "ok"}]},
        )

        provider.generate("Hello")

        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["max_tokens"] == 2048

    @patch("httpx.post")
    def test_generate_api_error(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=500,
            text="Internal server error",
        )

        with pytest.raises(RuntimeError, match="Anthropic API error.*500"):
            provider.generate("Hello")


class TestAnthropicGenerateJson:
    @pytest.fixture(autouse=True)
    def _set_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            yield

    @pytest.fixture
    def provider(self):
        config = LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        return AnthropicLLMProvider(config)

    @patch("httpx.post")
    def test_generate_json_success(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "content": [{"type": "text", "text": '{"result": "ok"}'}]
            },
        )

        result = provider.generate_json("Return JSON")
        assert result == {"result": "ok"}

    @patch("httpx.post")
    def test_generate_json_appends_instruction(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "content": [{"type": "text", "text": '{"a": 1}'}]
            },
        )

        provider.generate_json("Analyze this")

        body = mock_post.call_args.kwargs["json"]
        prompt = body["messages"][0]["content"]
        assert "JSON" in prompt

    @patch("httpx.post")
    def test_generate_json_parses_markdown_wrapped(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "content": [{"type": "text", "text": '```json\n{"items": [1]}\n```'}]
            },
        )

        result = provider.generate_json("List items")
        assert result == {"items": [1]}

    @patch("httpx.post")
    def test_generate_json_invalid_raises(self, mock_post, provider):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "content": [{"type": "text", "text": "just plain text"}]
            },
        )

        with pytest.raises(RuntimeError, match="Failed to parse JSON"):
            provider.generate_json("Get data")


class TestAnthropicParseJson:
    def test_plain_json(self):
        assert AnthropicLLMProvider._parse_json('{"a": 1}') == {"a": 1}

    def test_json_in_markdown_block(self):
        text = '```json\n{"key": "val"}\n```'
        assert AnthropicLLMProvider._parse_json(text) == {"key": "val"}

    def test_invalid_json(self):
        with pytest.raises(RuntimeError, match="Failed to parse JSON"):
            AnthropicLLMProvider._parse_json("nope")


# ---------------------------------------------------------------------------
# Factory function — new providers
# ---------------------------------------------------------------------------


class TestFactoryNewProviders:
    def test_factory_creates_openai_provider(self):
        config = Config(llm=LLMConfig(provider="openai", model="gpt-4o"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = get_llm_provider(config)
            assert isinstance(provider, OpenAILLMProvider)

    def test_factory_creates_anthropic_provider(self):
        config = Config(llm=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"))
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = get_llm_provider(config)
            assert isinstance(provider, AnthropicLLMProvider)

    def test_factory_openai_case_insensitive(self):
        config = Config(llm=LLMConfig(provider="OpenAI", model="gpt-4o"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = get_llm_provider(config)
            assert isinstance(provider, OpenAILLMProvider)

    def test_factory_anthropic_case_insensitive(self):
        config = Config(llm=LLMConfig(provider="Anthropic", model="claude-sonnet-4-20250514"))
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = get_llm_provider(config)
            assert isinstance(provider, AnthropicLLMProvider)

    def test_factory_unknown_provider_raises(self):
        config = Config(llm=LLMConfig(provider="gemini", model="gemini-pro"))
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider(config)
