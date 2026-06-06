"""Tests for agent.conversation_loop — rate-limit retry behaviour."""

import re
from pathlib import Path


class TestConversationLoopRetryAfter:
    """Tests for Retry-After header handling in the conversation loop."""

    def test_retry_after_300_not_capped_at_120(self):
        """Retry-After 300s must be honoured, not capped at the old 120s limit."""
        source = Path(__file__).parent.parent.parent / "agent" / "conversation_loop.py"
        content = source.read_text()
        match = re.search(r"min\(float\(_ra_raw\),\s*(\d+)\)", content)
        assert match is not None
        cap = int(match.group(1))
        assert cap == 600, f"Expected cap=600, got {cap}"
        # Ensure the cap is high enough that 300s Retry-After is honoured
        assert cap >= 300, f"Cap {cap} would still truncate 300s Retry-After"
