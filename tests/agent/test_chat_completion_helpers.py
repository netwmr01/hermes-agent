"""Tests for RateControlledClient wrapping in chat_completion_helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.chat_completion_helpers import (
    interruptible_api_call,
    interruptible_streaming_api_call,
    handle_max_iterations,
)


class TestInterruptibleApiCallWrapsClient:
    """Test that interruptible_api_call wraps the OpenAI client for chat_completions."""

    def test_interruptible_api_call_wraps_client(self):
        """For api_mode='chat_completions', request_client should be wrapped
        with RateControlledClient before calling .chat.completions.create()."""
        agent = MagicMock()
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._compute_non_stream_stale_timeout.return_value = 30.0
        agent._is_openai_codex_backend.return_value = False

        real_client = MagicMock()
        real_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            model="gpt-4",
        )
        agent._create_request_openai_client.return_value = real_client

        mock_bucket = MagicMock()
        mock_bucket.wait_for_tokens.return_value = True

        with patch(
            "agent.chat_completion_helpers.get_global_bucket",
            return_value=mock_bucket,
        ):
            with patch(
                "agent.chat_completion_helpers.RateControlledClient",
                wraps=lambda client, bucket: client,
            ) as mock_rc:
                result = interruptible_api_call(agent, {"model": "gpt-4", "messages": []})

        mock_rc.assert_called_once_with(real_client, bucket=mock_bucket)
        real_client.chat.completions.create.assert_called_once()


class TestInterruptibleStreamingApiCallWrapsClient:
    """Test that interruptible_streaming_api_call wraps the OpenAI client."""

    def test_interruptible_streaming_api_call_wraps_client(self):
        """For streaming chat_completions, request_client should be wrapped
        with RateControlledClient before calling .chat.completions.create()."""
        agent = MagicMock()
        agent.api_mode = "chat_completions"
        agent._interrupt_requested = False
        agent._has_stream_consumers.return_value = False
        agent.reasoning_callback = None
        agent.stream_delta_callback = None
        agent.base_url = None

        real_client = MagicMock()
        stream_mock = MagicMock()
        stream_mock.__iter__ = lambda self: iter([
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))],
                model="gpt-4",
            ),
            SimpleNamespace(choices=[], model="gpt-4", usage=None),
        ])
        stream_mock.response = MagicMock()
        real_client.chat.completions.create.return_value = stream_mock
        agent._create_request_openai_client.return_value = real_client

        mock_bucket = MagicMock()
        mock_bucket.wait_for_tokens.return_value = True

        with patch(
            "agent.chat_completion_helpers.get_global_bucket",
            return_value=mock_bucket,
        ):
            with patch(
                "agent.chat_completion_helpers.RateControlledClient",
                wraps=lambda client, bucket: client,
            ) as mock_rc:
                result = interruptible_streaming_api_call(
                    agent, {"model": "gpt-4", "messages": []}
                )

        mock_rc.assert_called_once_with(real_client, bucket=mock_bucket)
        real_client.chat.completions.create.assert_called_once()


class TestSummaryPathWrapsClient:
    """Test that max-iterations summary paths wrap the OpenAI client."""

    def test_summary_path_wraps_client(self):
        """The summary generation path should wrap _ensure_primary_openai_client
        with RateControlledClient before calling .chat.completions.create()."""
        agent = MagicMock()
        agent.api_mode = "chat_completions"
        agent.model = "gpt-4"
        agent.max_iterations = 10
        agent.max_tokens = 1024
        agent._summary_temperature = 0.7
        agent._lm_reasoning_effort = None
        agent.openrouter_min_coding_score = None
        agent._is_anthropic_oauth = False
        agent._anthropic_preserve_dots.return_value = False
        agent._base_url_lower = ""
        agent.provider = None
        agent._is_openrouter_url.return_value = False
        agent._should_sanitize_tool_calls.return_value = False
        agent._supports_reasoning_extra_body.return_value = False
        agent._resolve_lmstudio_summary_reasoning_effort.return_value = None
        agent._cached_system_prompt = None
        agent.ephemeral_system_prompt = None
        agent.prefill_messages = None
        agent.providers_allowed = None
        agent.providers_ignored = None
        agent.providers_order = None
        agent.provider_sort = None
        agent._max_tokens_param.return_value = {"max_tokens": 1024}
        agent._sanitize_api_messages.side_effect = lambda x: x
        agent._drop_thinking_only_and_merge_users.side_effect = lambda x: x
        agent._copy_reasoning_content_for_api = MagicMock()

        real_client = MagicMock()
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Summary here")
                )
            ],
            model="gpt-4",
        )
        real_client.chat.completions.create.return_value = response
        agent._ensure_primary_openai_client.return_value = real_client

        transport = MagicMock()
        transport.normalize_response.return_value = SimpleNamespace(
            content="Summary here"
        )
        agent._get_transport.return_value = transport

        messages = [{"role": "user", "content": "test"}]
        api_call_count = 10

        mock_bucket = MagicMock()
        mock_bucket.wait_for_tokens.return_value = True

        with patch(
            "agent.chat_completion_helpers.get_global_bucket",
            return_value=mock_bucket,
        ):
            with patch(
                "agent.chat_completion_helpers.RateControlledClient",
                wraps=lambda client, bucket: client,
            ) as mock_rc:
                result = handle_max_iterations(
                    agent,
                    messages,
                    api_call_count=10,
                )

        mock_rc.assert_called_once_with(real_client, bucket=mock_bucket)
        assert real_client.chat.completions.create.call_count == 1

    def test_summary_retry_path_wraps_client(self):
        """The retry summary generation path should also wrap the client."""
        agent = MagicMock()
        agent.api_mode = "chat_completions"
        agent.model = "gpt-4"
        agent.max_iterations = 10
        agent.max_tokens = 1024
        agent._summary_temperature = 0.7
        agent._lm_reasoning_effort = None
        agent.openrouter_min_coding_score = None
        agent._is_anthropic_oauth = False
        agent._anthropic_preserve_dots.return_value = False
        agent._base_url_lower = ""
        agent.provider = None
        agent._is_openrouter_url.return_value = False
        agent._should_sanitize_tool_calls.return_value = False
        agent._supports_reasoning_extra_body.return_value = False
        agent._resolve_lmstudio_summary_reasoning_effort.return_value = None
        agent._cached_system_prompt = None
        agent.ephemeral_system_prompt = None
        agent.prefill_messages = None
        agent.providers_allowed = None
        agent.providers_ignored = None
        agent.providers_order = None
        agent.provider_sort = None
        agent._max_tokens_param.return_value = {"max_tokens": 1024}
        agent._sanitize_api_messages.side_effect = lambda x: x
        agent._drop_thinking_only_and_merge_users.side_effect = lambda x: x
        agent._copy_reasoning_content_for_api = MagicMock()

        real_client = MagicMock()
        # First call returns empty content to trigger retry path
        empty_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
            model="gpt-4",
        )
        retry_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Retry summary here")
                )
            ],
            model="gpt-4",
        )

        def _create_side_effect(*args, **kwargs):
            # Return empty on first call, retry response on second
            _create_side_effect.call_count += 1
            if _create_side_effect.call_count == 1:
                return empty_response
            return retry_response
        _create_side_effect.call_count = 0
        real_client.chat.completions.create.side_effect = _create_side_effect
        agent._ensure_primary_openai_client.return_value = real_client

        transport = MagicMock()
        # First normalize returns empty content to trigger retry path
        empty_norm = SimpleNamespace(content="")
        retry_norm = SimpleNamespace(content="Retry summary here")
        transport.normalize_response.side_effect = [
            empty_norm,
            retry_norm,
        ]
        agent._get_transport.return_value = transport

        messages = [{"role": "user", "content": "test"}]

        mock_bucket = MagicMock()
        mock_bucket.wait_for_tokens.return_value = True

        with patch(
            "agent.chat_completion_helpers.get_global_bucket",
            return_value=mock_bucket,
        ):
            with patch(
                "agent.chat_completion_helpers.RateControlledClient",
                wraps=lambda client, bucket: client,
            ) as mock_rc:
                result = handle_max_iterations(
                    agent,
                    messages,
                    api_call_count=10,
                )

        assert mock_rc.call_count == 2
        assert real_client.chat.completions.create.call_count == 2
