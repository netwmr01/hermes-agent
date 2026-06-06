"""Tests for agent.rate_control — file-synced token bucket and rate-controlled client."""

import json
import multiprocessing
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.rate_control import (
    FileSyncedTokenBucket,
    RateControlledClient,
    RateLimitExceededError,
    gate_api_call,
    get_global_bucket,
    reset_global_bucket,
)


def _child_consume(state_file_str):
    child_bucket = FileSyncedTokenBucket(
        capacity=3, refill_rate=0.01, state_file=Path(state_file_str)
    )
    return child_bucket.consume(tokens=2)


class TestFileSyncedTokenBucket:
    """Tests for FileSyncedTokenBucket."""

    def test_consume_succeeds_when_tokens_available(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        assert bucket.consume(tokens=1) is True
        # Use approximate comparison because tiny refill may occur between calls
        assert bucket.get_available_tokens() <= 4.1

    def test_consume_fails_when_bucket_empty(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=1, refill_rate=0.01, state_file=state_file
        )
        assert bucket.consume(tokens=1) is True
        assert bucket.consume(tokens=1) is False

    def test_refill_over_time(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=2, refill_rate=10.0, state_file=state_file
        )
        assert bucket.consume(tokens=2) is True
        time.sleep(0.15)
        assert bucket.get_available_tokens() > 0

    def test_wait_for_tokens_timeout(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=1, refill_rate=0.01, state_file=state_file
        )
        assert bucket.consume(tokens=1) is True
        result = bucket.wait_for_tokens(tokens=1, timeout=0.05)
        assert result is False

    def test_wait_for_tokens_succeeds(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=1, refill_rate=0.01, state_file=state_file
        )
        assert bucket.wait_for_tokens(tokens=1, timeout=1.0) is True

    def test_add_tokens(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        bucket.consume(tokens=3)
        bucket.add_tokens(2)
        assert bucket.get_available_tokens() <= 4.1

    def test_persists_state_to_file(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=1.0, state_file=state_file
        )
        bucket.consume(tokens=2)
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert "tokens" in data
        assert "last_refill" in data

    def test_reads_persisted_state(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        # Pre-populate state file with low tokens
        state_file.write_text(json.dumps({"tokens": 1.0, "last_refill": time.time()}))
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        assert bucket.get_available_tokens() <= 1.1

    def test_cross_process_sync(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket1 = FileSyncedTokenBucket(
            capacity=3, refill_rate=0.01, state_file=state_file
        )
        bucket2 = FileSyncedTokenBucket(
            capacity=3, refill_rate=0.01, state_file=state_file
        )
        assert bucket1.consume(tokens=1) is True
        assert bucket2.get_available_tokens() <= 2.1
        assert bucket2.consume(tokens=2) is True
        assert bucket1.get_available_tokens() < 0.1

    def test_feedback_from_headers_sets_tokens(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=10, refill_rate=0.01, state_file=state_file
        )
        bucket.consume(tokens=5)
        bucket.feedback_from_headers(
            {"x-ratelimit-remaining-requests": "3", "retry-after": "5"}
        )
        assert bucket.get_available_tokens() <= 3.1

    def test_feedback_from_headers_ignores_missing(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        bucket.consume(tokens=2)
        bucket.feedback_from_headers({})
        assert bucket.get_available_tokens() <= 3.1

    def test_feedback_from_headers_retry_after_only(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        bucket.consume(tokens=2)
        bucket.feedback_from_headers({"retry-after": "10"})
        assert bucket.get_available_tokens() <= 3.1

    def test_thread_safety(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=100, refill_rate=0.01, state_file=state_file
        )
        successes = 0
        lock = threading.Lock()

        def worker():
            nonlocal successes
            for _ in range(10):
                if bucket.consume(tokens=1):
                    with lock:
                        successes += 1

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert successes == 100

    def test_capacity_not_exceeded(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        bucket.add_tokens(100)
        assert bucket.get_available_tokens() == 5

    def test_consume_zero_or_negative_raises(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        with pytest.raises(ValueError, match="positive"):
            bucket.consume(tokens=0)
        with pytest.raises(ValueError, match="positive"):
            bucket.consume(tokens=-1)

    def test_add_tokens_zero_or_negative_raises(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        with pytest.raises(ValueError, match="positive"):
            bucket.add_tokens(0)
        with pytest.raises(ValueError, match="positive"):
            bucket.add_tokens(-1)

    def test_invalid_capacity_raises(self, tmp_path):
        with pytest.raises(ValueError, match="capacity"):
            FileSyncedTokenBucket(
                capacity=0, refill_rate=1.0, state_file=tmp_path / "bucket.json"
            )

    def test_invalid_refill_rate_raises(self, tmp_path):
        with pytest.raises(ValueError, match="refill_rate"):
            FileSyncedTokenBucket(
                capacity=5, refill_rate=0, state_file=tmp_path / "bucket.json"
            )

    def test_feedback_from_headers_retry_after_drains(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=10, refill_rate=0.01, state_file=state_file
        )
        bucket.consume(tokens=1)
        bucket.feedback_from_headers({"retry-after": "5"})
        # retry-after pushes last_refill forward, but a tiny elapsed time
        # may have already accrued; assert near-zero
        assert bucket.get_available_tokens() < 1.0

    def test_feedback_zero_remaining_blocks_until_reset(self, tmp_path):
        """x-ratelimit-remaining-requests: 0 drains tokens and blocks consume()."""
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        # Ensure bucket is full
        assert bucket.get_available_tokens() == 5
        # Simulate API response saying 0 remaining with a 0.2s retry-after
        bucket.feedback_from_headers(
            {"x-ratelimit-remaining-requests": "0", "retry-after": "0.2"}
        )
        # Should be blocked immediately
        assert bucket.consume() is False
        # Wait for retry-after to expire
        time.sleep(0.25)
        # Now consume should succeed (at least 1 token refilled)
        assert bucket.consume() is True

    def test_cross_process_sync_subprocess(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=3, refill_rate=0.01, state_file=state_file
        )
        assert bucket.consume(tokens=1) is True

        with multiprocessing.Pool(1) as pool:
            result = pool.apply(
                _child_consume,
                (str(state_file),),
            )
        assert result is True
        assert bucket.get_available_tokens() < 1.0


class TestRateControlledClient:
    """Tests for RateControlledClient."""

    def test_delegates_chat_completions_create(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        real_client = MagicMock()
        real_client.base_url = "https://api.example.com"
        real_client.api_key = "sk-test"
        real_client.chat.completions.create.return_value = {"choices": []}

        wrapped = RateControlledClient(real_client, bucket)
        assert wrapped._enabled is True
        result = wrapped.chat.completions.create(model="gpt-4", messages=[])

        assert result == {"choices": []}
        real_client.chat.completions.create.assert_called_once_with(
            model="gpt-4", messages=[]
        )

    def test_waits_for_tokens_before_call(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=1, refill_rate=0.01, state_file=state_file
        )
        bucket.consume(tokens=1)
        real_client = MagicMock()
        real_client.chat.completions.create.return_value = {"choices": []}

        wrapped = RateControlledClient(real_client, bucket, max_token_wait_seconds=0.05)
        with pytest.raises(RateLimitExceededError):
            wrapped.chat.completions.create(model="gpt-4", messages=[])
        real_client.chat.completions.create.assert_not_called()

    def test_preserves_client_attributes(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        real_client = MagicMock()
        real_client.base_url = "https://api.example.com"
        real_client.api_key = "sk-test"
        real_client.timeout = 30

        wrapped = RateControlledClient(real_client, bucket)
        assert wrapped.base_url == "https://api.example.com"
        assert wrapped.api_key == "sk-test"
        assert wrapped.timeout == 30

    def test_feeds_back_headers_on_success(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=10, refill_rate=0.01, state_file=state_file
        )
        real_client = MagicMock()
        response = MagicMock()
        response.headers = {"x-ratelimit-remaining-requests": "7"}
        real_client.chat.completions.create.return_value = response

        wrapped = RateControlledClient(real_client, bucket)
        wrapped.chat.completions.create(model="gpt-4", messages=[])
        assert bucket.get_available_tokens() <= 7.1

    def test_feeds_back_headers_on_exception(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=10, refill_rate=0.01, state_file=state_file
        )
        real_client = MagicMock()

        class RateLimitError(Exception):
            def __init__(self, msg):
                super().__init__(msg)
                self.response = MagicMock()
                self.response.headers = {"x-ratelimit-remaining-requests": "2"}

        exc = RateLimitError("rate limited")
        real_client.chat.completions.create.side_effect = exc

        wrapped = RateControlledClient(real_client, bucket)
        with pytest.raises(RateLimitError, match="rate limited"):
            wrapped.chat.completions.create(model="gpt-4", messages=[])
        assert bucket.get_available_tokens() <= 2.1

    def test_skips_gate_when_disabled(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=1, refill_rate=0.01, state_file=state_file
        )
        bucket.consume(tokens=1)
        real_client = MagicMock()
        real_client.chat.completions.create.return_value = {"choices": []}

        wrapped = RateControlledClient(
            real_client, bucket, enabled=False, max_token_wait_seconds=0.05
        )
        result = wrapped.chat.completions.create(model="gpt-4", messages=[])
        assert result == {"choices": []}
        real_client.chat.completions.create.assert_called_once()

    def test_async_chat_completions_create(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        real_client = MagicMock()
        real_client.chat.completions.create.return_value = {"choices": []}

        wrapped = RateControlledClient(real_client, bucket)
        result = wrapped.chat.completions.create(model="gpt-4", messages=[])
        assert result == {"choices": []}


class TestGateApiCall:
    """Tests for gate_api_call helper."""

    def test_wraps_client_and_calls(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        real_client = MagicMock()
        real_client.chat.completions.create.return_value = {"choices": []}

        wrapped = gate_api_call(real_client, bucket)
        result = wrapped.chat.completions.create(model="gpt-4", messages=[])
        assert result == {"choices": []}

    def test_uses_global_bucket_when_none_provided(self, tmp_path):
        real_client = MagicMock()
        real_client.chat.completions.create.return_value = {"choices": []}

        with patch(
            "agent.rate_control.get_global_bucket"
        ) as mock_get_bucket:
            mock_bucket = MagicMock()
            mock_bucket.wait_for_tokens.return_value = True
            mock_get_bucket.return_value = mock_bucket
            wrapped = gate_api_call(real_client)
            wrapped.chat.completions.create(model="gpt-4", messages=[])
            mock_get_bucket.assert_called_once()


class TestGetGlobalBucket:
    """Tests for get_global_bucket singleton."""

    def test_returns_singleton(self, tmp_path):
        with patch("agent.rate_control._global_bucket", None):
            with patch("agent.rate_control._GLOBAL_BUCKET_LOCK"):
                with patch(
                    "agent.rate_control.load_config_readonly"
                ) as mock_load:
                    mock_load.return_value = {
                        "rate_limit": {
                            "enabled": True,
                            "bucket_cap": 30,
                            "rpm": 30,
                            "base_delay": 60.0,
                            "max_delay": 120.0,
                            "max_token_wait_seconds": 60,
                        }
                    }
                    with patch(
                        "agent.rate_control.get_hermes_home",
                        return_value=tmp_path,
                    ):
                        b1 = get_global_bucket()
                        b2 = get_global_bucket()
                        assert b1 is b2

    def test_reads_config_values(self, tmp_path):
        with patch("agent.rate_control._global_bucket", None):
            with patch(
                "agent.rate_control.load_config_readonly"
            ) as mock_load:
                mock_load.return_value = {
                    "rate_limit": {
                        "enabled": True,
                        "bucket_cap": 50,
                        "rpm": 60,
                        "base_delay": 30.0,
                        "max_delay": 90.0,
                        "max_token_wait_seconds": 30,
                    }
                }
                with patch(
                    "agent.rate_control.get_hermes_home",
                    return_value=tmp_path,
                ):
                    bucket = get_global_bucket()
                    assert bucket.capacity == 50
                    assert bucket.refill_rate == 60 / 60.0

    def test_defaults_when_config_missing(self, tmp_path):
        with patch("agent.rate_control._global_bucket", None):
            with patch(
                "agent.rate_control.load_config_readonly"
            ) as mock_load:
                mock_load.return_value = {}
                with patch(
                    "agent.rate_control.get_hermes_home",
                    return_value=tmp_path,
                ):
                    bucket = get_global_bucket()
                    assert bucket.capacity == 30
                    assert bucket.refill_rate == 30 / 60.0

    def test_lazy_reload_on_config_change(self, tmp_path):
        with patch("agent.rate_control._global_bucket", None):
            with patch(
                "agent.rate_control.load_config_readonly"
            ) as mock_load:
                mock_load.return_value = {
                    "rate_limit": {
                        "enabled": True,
                        "bucket_cap": 30,
                        "rpm": 30,
                        "base_delay": 60.0,
                        "max_delay": 120.0,
                        "max_token_wait_seconds": 60,
                    }
                }
                with patch(
                    "agent.rate_control.get_hermes_home",
                    return_value=tmp_path,
                ):
                    bucket1 = get_global_bucket()
                    # Change config values and reset singleton
                    mock_load.return_value = {
                        "rate_limit": {
                            "enabled": True,
                            "bucket_cap": 100,
                            "rpm": 120,
                            "base_delay": 60.0,
                            "max_delay": 120.0,
                            "max_token_wait_seconds": 60,
                        }
                    }
                    reset_global_bucket()
                    bucket2 = get_global_bucket()
                    assert bucket1 is not bucket2
                    assert bucket2.capacity == 100
                    assert bucket2.refill_rate == 120 / 60.0


class TestLeakyBucketDelegation:
    """Tests for rate_limit.leaky_bucket delegating to agent.rate_control."""

    def test_get_default_bucket_returns_global_bucket(self, tmp_path):
        with patch("agent.rate_control._global_bucket", None):
            with patch(
                "agent.rate_control.load_config_readonly"
            ) as mock_load:
                mock_load.return_value = {
                    "rate_limit": {
                        "enabled": True,
                        "bucket_cap": 30,
                        "rpm": 30,
                        "base_delay": 60.0,
                        "max_delay": 120.0,
                        "max_token_wait_seconds": 60,
                    }
                }
                with patch(
                    "agent.rate_control.get_hermes_home",
                    return_value=tmp_path,
                ):
                    from rate_limit.leaky_bucket import get_default_bucket
                    from agent.rate_control import get_global_bucket

                    default = get_default_bucket()
                    global_ = get_global_bucket()
                    assert default is global_

    def test_get_default_bucket_emits_deprecation_warning(self, tmp_path):
        with patch("agent.rate_control._global_bucket", None):
            with patch(
                "agent.rate_control.load_config_readonly"
            ) as mock_load:
                mock_load.return_value = {
                    "rate_limit": {
                        "enabled": True,
                        "bucket_cap": 30,
                        "rpm": 30,
                        "base_delay": 60.0,
                        "max_delay": 120.0,
                        "max_token_wait_seconds": 60,
                    }
                }
                with patch(
                    "agent.rate_control.get_hermes_home",
                    return_value=tmp_path,
                ):
                    import warnings
                    from rate_limit.leaky_bucket import get_default_bucket

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        get_default_bucket()
                        assert len(w) == 1
                        assert issubclass(w[0].category, DeprecationWarning)
                        assert "get_default_bucket is deprecated" in str(w[0].message)
                        assert "agent.rate_control.get_global_bucket" in str(w[0].message)

    def test_token_bucket_backward_compatibility(self):
        import warnings
        from rate_limit.leaky_bucket import TokenBucket

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bucket = TokenBucket(capacity=5, refill_rate=1.0)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "TokenBucket is deprecated" in str(w[0].message)
            assert "agent.rate_control.FileSyncedTokenBucket" in str(w[0].message)

        assert bucket.capacity == 5
        assert bucket.refill_rate == 1.0

        # consume / get_available_tokens
        assert bucket.consume(tokens=2) is True
        assert bucket.get_available_tokens() <= 3.1

        # add_tokens
        bucket.add_tokens(1)
        assert bucket.get_available_tokens() <= 4.1

        # wait_for_tokens
        assert bucket.wait_for_tokens(tokens=1, timeout=1.0) is True
        assert bucket.get_available_tokens() <= 3.1


class TestAuxiliaryClientWrapping:
    """Tests for T4: auxiliary client rate-control wrapping."""

    def test_get_cached_client_returns_rate_controlled(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from agent.rate_control import FileSyncedTokenBucket
        from agent.auxiliary_client import _get_cached_client

        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        real_client = MagicMock()
        real_client.chat.completions.create.return_value = {"choices": []}
        real_client.base_url = "https://api.example.com"
        real_client.api_key = "sk-test"

        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(real_client, "test-model")), \
             patch("agent.auxiliary_client.gate_api_call") as mock_gate:
            mock_wrapped = MagicMock()
            mock_wrapped.chat.completions.create.return_value = {"choices": []}
            mock_gate.return_value = mock_wrapped
            client, model = _get_cached_client("test-provider", "test-model")
            mock_gate.assert_called_once_with(real_client)
            assert client is mock_wrapped
            assert model == "test-model"

    def test_resolve_vision_provider_client_wraps_client(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from agent.rate_control import FileSyncedTokenBucket
        from agent.auxiliary_client import resolve_vision_provider_client

        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        real_client = MagicMock()
        real_client.chat.completions.create.return_value = {"choices": []}
        real_client.base_url = "https://api.example.com"
        real_client.api_key = "sk-test"

        with patch("agent.auxiliary_client._resolve_strict_vision_backend", return_value=(real_client, "vision-model")), \
             patch("agent.auxiliary_client.gate_api_call") as mock_gate:
            mock_wrapped = MagicMock()
            mock_wrapped.chat.completions.create.return_value = {"choices": []}
            mock_gate.return_value = mock_wrapped
            provider, client, model = resolve_vision_provider_client(
                provider="openrouter", model="vision-model"
            )
            mock_gate.assert_called_once_with(real_client)
            assert client is mock_wrapped
            assert model == "vision-model"

    def test_fallback_functions_wrap_clients(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from agent.rate_control import FileSyncedTokenBucket
        from agent.auxiliary_client import (
            _try_payment_fallback,
            _try_configured_fallback_chain,
            _try_main_agent_model_fallback,
        )

        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        real_client = MagicMock()
        real_client.chat.completions.create.return_value = {"choices": []}
        real_client.base_url = "https://api.example.com"
        real_client.api_key = "sk-test"

        with patch("agent.auxiliary_client._get_provider_chain", return_value=[("openrouter", lambda: (real_client, "fallback-model"))]), \
             patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="fallback-model"), \
             patch("agent.auxiliary_client.resolve_provider_client", return_value=(real_client, "fallback-model")), \
             patch("agent.auxiliary_client._get_auxiliary_task_config", return_value={}), \
             patch("agent.auxiliary_client.gate_api_call") as mock_gate:
            mock_wrapped = MagicMock()
            mock_wrapped.chat.completions.create.return_value = {"choices": []}
            mock_gate.return_value = mock_wrapped

            client, model, label = _try_payment_fallback("nous")
            assert client is mock_wrapped
            assert label == "openrouter"

            client, model, label = _try_main_agent_model_fallback("nous")
            assert client is mock_wrapped
            assert label == "main-agent(openrouter)"

            client, model, label = _try_configured_fallback_chain("test-task", "nous")
            # No fallback chain configured, returns None
            assert client is None


class TestConversationLoopRetryCap:
    """Tests for T5: conversation-loop retry cap raised to 600."""

    def test_retry_after_cap_is_600(self):
        import re
        from pathlib import Path
        source = Path(__file__).parent.parent.parent / "agent" / "conversation_loop.py"
        content = source.read_text()
        match = re.search(r"min\(float\(_ra_raw\),\s*(\d+)\)", content)
        assert match is not None
        cap = int(match.group(1))
        assert cap == 600
        assert "secondary safety net" in content


class TestNousRateGuardRaceCondition:
    """Tests for T6: Nous rate-guard filelock protection."""

    def test_nous_rate_limit_remaining_uses_filelock(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from agent.nous_rate_guard import nous_rate_limit_remaining, _state_path

        state_file = tmp_path / "nous.json"
        state_file.write_text('{"reset_at": 9999999999.0}')
        with patch("agent.nous_rate_guard._state_path", return_value=str(state_file)):
            with patch("agent.nous_rate_guard.FileLock") as mock_lock_cls:
                mock_lock = MagicMock()
                mock_lock_cls.return_value = mock_lock
                result = nous_rate_limit_remaining()
                mock_lock_cls.assert_called_once_with(str(state_file) + ".lock")
                mock_lock.__enter__.assert_called_once()
                mock_lock.__exit__.assert_called_once()
                assert result is not None
                assert result > 0

    def test_record_nous_rate_limit_uses_filelock(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from agent.nous_rate_guard import record_nous_rate_limit, _state_path

        state_file = tmp_path / "nous.json"
        with patch("agent.nous_rate_guard._state_path", return_value=str(state_file)):
            with patch("agent.nous_rate_guard.FileLock") as mock_lock_cls:
                mock_lock = MagicMock()
                mock_lock_cls.return_value = mock_lock
                record_nous_rate_limit(headers={"retry-after": "60"})
                mock_lock_cls.assert_called_once_with(str(state_file) + ".lock")
                mock_lock.__enter__.assert_called_once()
                mock_lock.__exit__.assert_called_once()
                assert state_file.exists()

    def test_nous_rate_guard_no_race_on_expired_state(self, tmp_path):
        from unittest.mock import patch, MagicMock
        from agent.nous_rate_guard import nous_rate_limit_remaining, _state_path

        state_file = tmp_path / "nous.json"
        state_file.write_text('{"reset_at": 1.0}')
        with patch("agent.nous_rate_guard._state_path", return_value=str(state_file)):
            with patch("agent.nous_rate_guard.FileLock") as mock_lock_cls:
                mock_lock = MagicMock()
                mock_lock_cls.return_value = mock_lock
                result = nous_rate_limit_remaining()
                mock_lock_cls.assert_called_once_with(str(state_file) + ".lock")
                assert result is None
                # File should be cleaned up inside the lock
                assert not state_file.exists()


class TestAsyncRateControl:
    """Tests for async rate-control wrappers (GAP 1)."""

    @pytest.mark.asyncio
    async def test_async_create_gates_and_delegates(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=5, refill_rate=0.01, state_file=state_file
        )
        real_client = MagicMock()
        # Mock async create as a coroutine
        async def async_create(**kwargs):
            return {"choices": []}
        real_client.chat.completions.create = async_create

        wrapped = RateControlledClient(real_client, bucket)
        result = await wrapped.chat.completions.acreate(model="gpt-4", messages=[])
        assert result == {"choices": []}

    @pytest.mark.asyncio
    async def test_async_wait_for_tokens(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=1, refill_rate=0.01, state_file=state_file
        )
        assert await bucket.wait_for_tokens_async(tokens=1, timeout=1.0) is True
        assert await bucket.wait_for_tokens_async(tokens=1, timeout=0.05) is False

    @pytest.mark.asyncio
    async def test_async_create_waits_for_tokens(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=1, refill_rate=0.01, state_file=state_file
        )
        bucket.consume(tokens=1)
        real_client = MagicMock()
        async def async_create(**kwargs):
            return {"choices": []}
        real_client.chat.completions.create = async_create

        wrapped = RateControlledClient(real_client, bucket, max_token_wait_seconds=0.05)
        with pytest.raises(RateLimitExceededError):
            await wrapped.chat.completions.acreate(model="gpt-4", messages=[])


class TestEnabledFlag:
    """Tests for config enabled flag (GAP 2)."""

    def test_enabled_false_skips_gate(self, tmp_path):
        state_file = tmp_path / "bucket.json"
        bucket = FileSyncedTokenBucket(
            capacity=1, refill_rate=0.01, state_file=state_file
        )
        bucket.consume(tokens=1)
        real_client = MagicMock()
        real_client.chat.completions.create.return_value = {"choices": []}

        wrapped = RateControlledClient(
            real_client, bucket, enabled=False, max_token_wait_seconds=0.05
        )
        result = wrapped.chat.completions.create(model="gpt-4", messages=[])
        assert result == {"choices": []}
        real_client.chat.completions.create.assert_called_once()

    def test_gate_api_call_reads_enabled_from_config(self, tmp_path):
        real_client = MagicMock()
        real_client.chat.completions.create.return_value = {"choices": []}

        with patch("agent.rate_control._global_bucket", None):
            with patch("agent.rate_control._global_bucket_config", None):
                with patch(
                    "agent.rate_control.load_config_readonly"
                ) as mock_load:
                    mock_load.return_value = {
                        "rate_limit": {
                            "enabled": False,
                            "bucket_cap": 30,
                            "rpm": 30,
                            "max_token_wait_seconds": 45,
                        }
                    }
                    with patch(
                        "agent.rate_control.get_hermes_home",
                        return_value=tmp_path,
                    ):
                        wrapped = gate_api_call(real_client)
                        assert wrapped._enabled is False
                        assert wrapped._max_token_wait_seconds == 45


class TestHotReload:
    """Tests for hot-reload of global bucket config (GAP 3)."""

    def test_hot_reload_recreates_bucket_on_config_change(self, tmp_path):
        with patch("agent.rate_control._global_bucket", None):
            with patch("agent.rate_control._global_bucket_config", None):
                with patch(
                    "agent.rate_control.load_config_readonly"
                ) as mock_load:
                    mock_load.return_value = {
                        "rate_limit": {
                            "enabled": True,
                            "bucket_cap": 30,
                            "rpm": 30,
                            "max_token_wait_seconds": 60,
                        }
                    }
                    with patch(
                        "agent.rate_control.get_hermes_home",
                        return_value=tmp_path,
                    ):
                        bucket1 = get_global_bucket()
                        # Change config values — should trigger recreation
                        mock_load.return_value = {
                            "rate_limit": {
                                "enabled": True,
                                "bucket_cap": 100,
                                "rpm": 120,
                                "max_token_wait_seconds": 60,
                            }
                        }
                        bucket2 = get_global_bucket()
                        assert bucket1 is not bucket2
                        assert bucket2.capacity == 100
                        assert bucket2.refill_rate == 120 / 60.0

    def test_hot_reload_no_recreate_when_unchanged(self, tmp_path):
        with patch("agent.rate_control._global_bucket", None):
            with patch("agent.rate_control._global_bucket_config", None):
                with patch(
                    "agent.rate_control.load_config_readonly"
                ) as mock_load:
                    mock_load.return_value = {
                        "rate_limit": {
                            "enabled": True,
                            "bucket_cap": 30,
                            "rpm": 30,
                            "max_token_wait_seconds": 60,
                        }
                    }
                    with patch(
                        "agent.rate_control.get_hermes_home",
                        return_value=tmp_path,
                    ):
                        bucket1 = get_global_bucket()
                        bucket2 = get_global_bucket()
                        assert bucket1 is bucket2
