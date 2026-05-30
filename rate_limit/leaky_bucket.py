"""
Token Bucket Rate Limiter

Implements a thread-safe token bucket algorithm for rate limiting.
Supports both time-driven refill and manual token addition.
"""
import time
import threading
from typing import Optional


class TokenBucket:
    """
    A thread-safe token bucket rate limiter.

    The bucket refills at a constant rate (refill_rate tokens per second).
    Tokens are consumed when requests arrive. If insufficient tokens are
    available, the consume() method returns False.

    Attributes:
        capacity: Maximum number of tokens the bucket can hold.
        refill_rate: Rate of token refill in tokens per second.
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        """
        Initialize the token bucket.

        Args:
            capacity: Maximum number of tokens (burst capacity).
            refill_rate: Token refill rate in tokens per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = capacity
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """
        Refill tokens based on elapsed time since last refill.

        This method is called internally within locked sections.
        """
        now = time.time()
        elapsed = now - self._last_refill
        if elapsed > 0:
            increment = elapsed * self.refill_rate
            self._tokens = min(self.capacity, self._tokens + increment)
            self._last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """
        Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume (default: 1).

        Returns:
            True if tokens were successfully consumed, False otherwise.
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def add_tokens(self, amount: float) -> None:
        """
        Manually add tokens to the bucket.

        Args:
            amount: Number of tokens to add.
        """
        with self._lock:
            self._tokens = min(self.capacity, self._tokens + amount)

    def get_available_tokens(self) -> float:
        """
        Get the current number of available tokens.

        Returns:
            Number of tokens currently available in the bucket.
        """
        with self._lock:
            self._refill()
            return self._tokens

    def wait_for_tokens(
        self, tokens: float = 1.0, timeout: Optional[float] = None
    ) -> bool:
        """
        Wait until tokens are available and consume them.

        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            True if tokens were consumed, False if timeout occurred.
        """
        start_time = time.time()
        while True:
            if self.consume(tokens):
                return True
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False
                # Wait a short time before retrying
                time.sleep(min(0.1, timeout - elapsed))
            else:
                time.sleep(0.1)


# ── Module-level singleton ─────────────────────────────────────────────────

# Hardcoded limits for the global backend-protection bucket.
#   capacity  = 30 tokens  (allow 30 burst requests)
#   refill_rate = 0.5 tokens/sec  (= 30 tokens/min steady-state)
# This enforces a ceiling of ~30 requests/minute to external backends.
_RATE_LIMIT_CAPACITY = 30.0
_RATE_LIMIT_REFILL_RATE = 0.5  # tokens per second  →  30/min

_default_bucket: Optional[TokenBucket] = None
_default_bucket_lock = threading.Lock()


def get_default_bucket() -> TokenBucket:
    """
    Return the module-level singleton TokenBucket.

    Thread-safe on first creation. All subsequent calls return the same
    instance, so token state (available count, refill headroom) is shared
    across every call site that uses ``get_default_bucket()``.
    """
    global _default_bucket
    if _default_bucket is None:
        with _default_bucket_lock:
            # Double-check inside the lock
            if _default_bucket is None:
                _default_bucket = TokenBucket(
                    capacity=_RATE_LIMIT_CAPACITY,
                    refill_rate=_RATE_LIMIT_REFILL_RATE,
                )
    return _default_bucket