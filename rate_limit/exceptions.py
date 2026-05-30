"""
Rate Limiting Exceptions

Defines exceptions used by the rate limiting system.
"""


class RateLimitExceededError(Exception):
    """
    Exception raised when a rate limit is exceeded.

    This typically occurs when too many requests are made within
    a short time window.
    """

    def __init__(self, message: str = "Rate limit exceeded. Please try again later.") -> None:
        """
        Initialize the exception.

        Args:
            message: Error message describing the rate limit violation.
        """
        self.message = message
        super().__init__(self.message)


class RateLimitConfigurationError(Exception):
    """
    Exception raised when rate limiting configuration is invalid.
    """

    def __init__(self, message: str = "Invalid rate limit configuration.") -> None:
        """
        Initialize the exception.

        Args:
            message: Error message describing the configuration error.
        """
        self.message = message
        super().__init__(self.message)