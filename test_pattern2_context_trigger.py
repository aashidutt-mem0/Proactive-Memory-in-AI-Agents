"""
Pattern 2 integration tests: Context-Trigger Scan
==================================================
Calls the real Mem0 API and real OpenRouter-backed chat API.

Requirements:
    export MEM0_API_KEY=your-key-from-app.mem0.ai
    export OPENROUTER_API_KEY=your-openrouter-key
    pip install mem0ai openai pytest
"""

import os

import pytest

from demos import detect_intent, on_context_message
from live_clients import Mem0MemoryStore, OpenRouterChatModel

USER_ID = "test_p2_alice"

REQUIRED_ENV_VARS = ("MEM0_API_KEY", "OPENROUTER_API_KEY")
missing_env = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]

pytestmark = pytest.mark.skipif(
    bool(missing_env),
    reason=f"Integration test requires env vars: {', '.join(REQUIRED_ENV_VARS)}",
)


def test_context_trigger_searches_on_demand_turn() -> None:
    """
    Seeds a prior bug into real Mem0, then sends a message that explicitly
    references the same file. Verifies the real intent classifier returns
    DEMAND and that a reply is produced.

    The message names database.py directly and asks about connection pool
    settings - both strong DEMAND signals per the classifier prompt.
    """
    memory = Mem0MemoryStore()
    chat = OpenRouterChatModel()

    memory.add(
        [
            {"role": "user", "content": "We had a nasty bug in database.py. Connection pool exhausted under load."},
            {"role": "assistant", "content": "Fixed by setting pool_size=20 and pool_pre_ping=True in SQLAlchemy."},
        ],
        user_id=USER_ID,
    )

    # Use an unambiguous DEMAND message: explicit file reference + prior context
    result = on_context_message(
        memory,
        chat,
        USER_ID,
        "Open database.py - I need to check the connection pool config from our last session.",
        history=[],
    )

    assert result["decision"] == "DEMAND", (
        f"Expected DEMAND for an explicit file-reference turn, got {result['decision']}. "
        f"Query attempted: {result['query']}"
    )
    assert result["reply"], "Expected a non-empty agent reply"


def test_context_trigger_skips_search_on_no_demand_turn() -> None:
    """
    Sends a simple acknowledgment turn. Verifies the real intent classifier
    returns NO_DEMAND and that Mem0 search is never called.
    """
    memory = Mem0MemoryStore()
    chat = OpenRouterChatModel()

    result = on_context_message(
        memory,
        chat,
        USER_ID,
        "Sounds good, thanks.",
        history=[{"role": "assistant", "content": "We can start with the database layer."}],
    )

    assert result["decision"] == "NO_DEMAND", (
        f"Expected NO_DEMAND for an acknowledgment turn, got {result['decision']}"
    )
    assert result["memories"] == [], "Expected no memories retrieved on NO_DEMAND"
    assert result["reply"], "Expected a non-empty agent reply even on NO_DEMAND"


def test_detect_intent_returns_valid_intent_object() -> None:
    """
    Calls the real OpenAI classifier directly and verifies the Intent
    object is well-formed for a clear context-shift message.
    """
    chat = OpenRouterChatModel()

    intent = detect_intent(chat, "Let's look at the auth module now.", history=[])

    assert intent.decision in {"DEMAND", "NO_DEMAND"}
    if intent.decision == "DEMAND":
        assert intent.query, "DEMAND decision must include a search query"
        assert intent.should_search is True
