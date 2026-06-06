"""Rate-controlled API client with file-synced token bucket.

Provides cross-process rate limiting by persisting token bucket state
to a JSON file protected by filelock.FileLock.
"""

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from filelock import FileLock

from hermes_cli.config import load_config_readonly
from hermes_constants import get_hermes_home
from rate_limit.exceptions import RateLimitExceededError

logger = logging.getLogger(__name__)


class FileSyncedTokenBucket:
    """Token bucket with cross-process sync via file locking.

    Same interface as rate_limit.leaky_bucket.TokenBucket:
    ``consume()``, ``wait_for_tokens()``, ``add_tokens()``,
    ``get_available_tokens()``.
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        state_file: Path,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file_lock = FileLock(str(state_file) + ".lock")

    def _read_state(self) -> Dict[str, float]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Corrupted bucket state file %s: %s. Resetting to full capacity.",
                    self.state_file,
                    exc,
                )
        return {"tokens": self.capacity, "last_refill": time.time()}

    def _write_state(self, tokens: float, last_refill: float) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({"tokens": tokens, "last_refill": last_refill}, f)

    def _refill(self, tokens: float, last_refill: float) -> tuple[float, float]:
        now = time.time()
        elapsed = now - last_refill
        if elapsed > 0:
            increment = elapsed * self.refill_rate
            tokens = min(self.capacity, tokens + increment)
            last_refill = now
        return tokens, last_refill

    def consume(self, tokens: float = 1.0) -> bool:
        """Attempt to consume tokens from the bucket.

        Returns True if tokens were successfully consumed.
        """
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        with self._file_lock:
            state = self._read_state()
            available, last_refill = self._refill(
                state["tokens"], state["last_refill"]
            )
            if available >= tokens:
                available -= tokens
                self._write_state(available, last_refill)
                return True
            self._write_state(available, last_refill)
            return False

    def add_tokens(self, amount: float) -> None:
        """Manually add tokens to the bucket."""
        if amount <= 0:
            raise ValueError("amount must be positive")
        with self._file_lock:
            state = self._read_state()
            available, last_refill = self._refill(
                state["tokens"], state["last_refill"]
            )
            available = min(self.capacity, available + amount)
            self._write_state(available, last_refill)

    def get_available_tokens(self) -> float:
        """Get the current number of available tokens."""
        with self._file_lock:
            state = self._read_state()
            available, _ = self._refill(state["tokens"], state["last_refill"])
            return available

    def wait_for_tokens(
        self, tokens: float = 1.0, timeout: Optional[float] = None
    ) -> bool:
        """Wait until tokens are available and consume them.

        Polls with 0.1s sleep, re-acquiring the file lock each iteration.
        """
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        start_time = time.time()
        while True:
            if self.consume(tokens):
                return True
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                time.sleep(min(0.1, max(0, timeout - elapsed)))
            else:
                time.sleep(0.1)

    async def wait_for_tokens_async(
        self, tokens: float = 1.0, timeout: Optional[float] = None
    ) -> bool:
        """Wait until tokens are available and consume them (async version).

        Polls with asyncio.sleep, re-acquiring the file lock each iteration.
        """
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        start_time = time.time()
        while True:
            if self.consume(tokens):
                return True
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                await asyncio.sleep(min(0.1, max(0, timeout - elapsed)))
            else:
                await asyncio.sleep(0.1)

    def feedback_from_headers(self, headers: Dict[str, str]) -> None:
        """Adjust bucket state based on API response headers.

        Parses ``x-ratelimit-remaining-requests`` and ``retry-after`` headers.
        If ``retry-after`` is present, drains tokens to 0 and sets the refill
        timestamp into the future so the bucket blocks until the retry period
        expires.
        """
        lowered = {k.lower(): v for k, v in headers.items()}
        retry_after = lowered.get("retry-after")
        if retry_after is not None:
            try:
                retry_seconds = float(retry_after)
                with self._file_lock:
                    # Drain tokens and push last_refill forward so no refill
                    # happens until retry_seconds have elapsed.
                    # The - (1.0 / refill_rate) offset ensures that after the
                    # retry period, the bucket will immediately grant 1 token
                    # (since refill logic adds elapsed_time * refill_rate tokens).
                    self._write_state(
                        tokens=0.0,
                        last_refill=time.time() + retry_seconds - (1.0 / self.refill_rate),
                    )
                return
            except (ValueError, TypeError):
                pass
        remaining = lowered.get("x-ratelimit-remaining-requests")
        if remaining is not None:
            try:
                target = float(remaining)
                with self._file_lock:
                    state = self._read_state()
                    available, last_refill = self._refill(
                        state["tokens"], state["last_refill"]
                    )
                    available = min(self.capacity, target)
                    self._write_state(available, last_refill)
            except (ValueError, TypeError):
                pass


class RateControlledClient:
    """Wraps a real OpenAI client with token-bucket rate limiting.

    Exposes ``.chat.completions.create(**kwargs)`` and preserves all
    original client attributes for compatibility.
    """

    def __init__(
        self,
        client: Any,
        bucket: FileSyncedTokenBucket,
        enabled: bool = True,
        max_token_wait_seconds: float = 60.0,
    ) -> None:
        self._client = client
        self._bucket = bucket
        self._enabled = enabled
        self._max_token_wait_seconds = max_token_wait_seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _apply_feedback(self, result: Any) -> None:
        headers: Dict[str, str] = {}
        try:
            if hasattr(result, "headers"):
                headers = dict(result.headers)
            elif hasattr(result, "response") and hasattr(result.response, "headers"):
                headers = dict(result.response.headers)
            elif hasattr(result, "_response") and hasattr(result._response, "headers"):
                headers = dict(result._response.headers)
        except (AttributeError, TypeError):
            pass
        if headers:
            self._bucket.feedback_from_headers(headers)

    def _create_with_gate(self, **kwargs: Any) -> Any:
        if self._enabled:
            if not self._bucket.wait_for_tokens(
                tokens=1, timeout=self._max_token_wait_seconds
            ):
                raise RateLimitExceededError(
                    f"Rate limit: no token available within {self._max_token_wait_seconds}s"
                )
        try:
            result = self._client.chat.completions.create(**kwargs)
            self._apply_feedback(result)
            return result
        except Exception as exc:
            self._apply_feedback(exc)
            raise

    async def _create_with_gate_async(self, **kwargs: Any) -> Any:
        if self._enabled:
            if not await self._bucket.wait_for_tokens_async(
                tokens=1, timeout=self._max_token_wait_seconds
            ):
                raise RateLimitExceededError(
                    f"Rate limit: no token available within {self._max_token_wait_seconds}s"
                )
        try:
            result = await self._client.chat.completions.create(**kwargs)
            self._apply_feedback(result)
            return result
        except Exception as exc:
            self._apply_feedback(exc)
            raise

    @property
    def chat(self) -> Any:
        return _RateControlledChat(self)


class _RateControlledChat:
    """Proxy for ``client.chat`` that returns a gated completions proxy."""

    def __init__(self, wrapper: RateControlledClient) -> None:
        self._wrapper = wrapper
        self._real = wrapper._client.chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    @property
    def completions(self) -> Any:
        return _RateControlledChatCompletions(self._wrapper)


class _RateControlledChatCompletions:
    """Proxy for ``client.chat.completions`` that gates ``create()``."""

    def __init__(self, wrapper: RateControlledClient) -> None:
        self._wrapper = wrapper
        self._real = wrapper._client.chat.completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def create(self, **kwargs: Any) -> Any:
        return self._wrapper._create_with_gate(**kwargs)

    async def acreate(self, **kwargs: Any) -> Any:
        return await self._wrapper._create_with_gate_async(**kwargs)


# ── Module-level singleton ─────────────────────────────────────────────────

_global_bucket: Optional[FileSyncedTokenBucket] = None
_global_bucket_config: Optional[Dict[str, Any]] = None
_GLOBAL_BUCKET_LOCK = threading.Lock()


def get_global_bucket() -> FileSyncedTokenBucket:
    """Return the module-level singleton FileSyncedTokenBucket.

    Reads ``rate_limit.*`` values from config.yaml via
    ``load_config_readonly()`` on first call and re-reads lazily
    when the config file changes.
    """
    global _global_bucket, _global_bucket_config
    config = load_config_readonly()
    rl = config.get("rate_limit", {})
    new_config = {
        "bucket_cap": rl.get("bucket_cap", 30),
        "rpm": rl.get("rpm", 30),
        "enabled": rl.get("enabled", True),
        "max_token_wait_seconds": rl.get("max_token_wait_seconds", 60),
    }
    if _global_bucket is None or _global_bucket_config != new_config:
        with _GLOBAL_BUCKET_LOCK:
            if _global_bucket is None or _global_bucket_config != new_config:
                capacity = new_config["bucket_cap"]
                refill_rate = new_config["rpm"] / 60.0
                state_file = (
                    get_hermes_home() / "rate_limits" / "global_bucket.json"
                )
                _global_bucket = FileSyncedTokenBucket(
                    capacity=capacity,
                    refill_rate=refill_rate,
                    state_file=state_file,
                )
                _global_bucket_config = new_config
    return _global_bucket


def gate_api_call(
    client: Any, bucket: Optional[FileSyncedTokenBucket] = None
) -> RateControlledClient:
    """Wrap a client for one-off rate-controlled API calls.

    If ``bucket`` is None, uses ``get_global_bucket()``.
    """
    if bucket is None:
        bucket = get_global_bucket()
    config = load_config_readonly()
    rl = config.get("rate_limit", {})
    enabled = rl.get("enabled", True)
    max_token_wait_seconds = rl.get("max_token_wait_seconds", 60)
    return RateControlledClient(
        client, bucket, enabled=enabled, max_token_wait_seconds=max_token_wait_seconds
    )


def reset_global_bucket() -> None:
    """Reset the module-level singleton for testing."""
    global _global_bucket, _global_bucket_config
    with _GLOBAL_BUCKET_LOCK:
        _global_bucket = None
        _global_bucket_config = None
