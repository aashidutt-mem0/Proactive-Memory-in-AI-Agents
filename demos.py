from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


class MemoryStore(Protocol):
    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...

    def search(self, query: str, *, user_id: str, limit: int = 3) -> list[dict[str, Any]]:
        ...

    def get_all(self, *, user_id: str) -> dict[str, Any]:
        ...


class ChatModel(Protocol):
    def complete_json(self, prompt: str, *, temperature: float = 0.0) -> Any:
        ...

    def complete_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> str:
        ...


@dataclass(frozen=True)
class Intent:
    decision: str
    query: str | None = None

    @property
    def should_search(self) -> bool:
        return self.decision == "DEMAND" and bool(self.query)


def _memory_text(memory: dict[str, Any]) -> str:
    return str(memory.get("memory") or memory.get("content") or "")


_DEMAND_PATTERNS = (
    r"\b[\w\-]+\.(py|ts|tsx|js|jsx|java|go|rs|rb|sql|yaml|yml|json|toml)\b",
    r"\b(module|component|file|config|schema|migration|deploy|deployment)\b",
    r"\b(error|bug|blocker|failing|failure|issue)\b",
    r"\bwhere we left off\b",
    r"\b(last|previous)\s+session\b",
    r"\bpick up\b",
    r"\bcheck\b",
)


def _looks_like_demand(message: str) -> bool:
    lowered = message.lower()
    return any(re.search(pattern, lowered) for pattern in _DEMAND_PATTERNS)


def _fallback_query(message: str) -> str:
    cleaned = " ".join(message.strip().split())
    return cleaned[:200] if cleaned else "recent task context"


def session_start_scan(
    memory: MemoryStore,
    user_id: str,
    context: dict[str, str],
    *,
    now: datetime | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Pattern 1: search before the first user message using ambient context."""

    current = now or datetime.now()
    if current.hour < 12:
        time_of_day = "morning"
    elif current.hour < 17:
        time_of_day = "afternoon"
    else:
        time_of_day = "evening"

    project = context.get("project", "")
    filename = context.get("open_file", "")
    query = "What was this user recently working on or blocked by"
    if project:
        query += f" in project {project}"
    if filename:
        query += f" related to {filename}"
    query += f"? It is {time_of_day}."

    return memory.search(query, user_id=user_id, limit=limit)


def build_system_prompt(memories: list[dict[str, Any]]) -> str:
    base = "You are a helpful coding assistant with persistent memory."
    useful_memories = [_memory_text(item) for item in memories if _memory_text(item)]
    if not useful_memories:
        return base

    lines = "\n".join(f"- {item}" for item in useful_memories)
    return f"{base}\n\n[Context from previous sessions, use naturally if relevant:]\n{lines}"


def detect_intent(chat: ChatModel, message: str, history: list[dict[str, str]]) -> Intent:
    """Pattern 2 gate: classify whether a turn deserves long-term memory search."""

    recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
    if not recent:
        recent = "(start of conversation)"

    prompt = f"""You are an intent detection model for an AI coding assistant with long-term memory.

Your job: decide whether the latest user message warrants a search of long-term memory.

DEMAND (search memory) when the message:
- Names a specific file or module ("database.py", "auth.py", "the config module")
- References past work, a prior bug, or a previous decision ("where we left off", "that pool issue")
- Starts a new task or switches topic
- Mentions an error, blocker, or deployment step

NO_DEMAND (skip search) when the message:
- Is a short acknowledgment with no new content ("ok", "thanks", "got it", "sounds good")
- Directly continues the immediately preceding assistant reply with no new topic

Recent history:
{recent}

Latest message: {message}

Examples of correct responses:
  message: "Open database.py and check the pool config"
  response: {{"decision": "DEMAND", "query": "database.py connection pool configuration"}}

  message: "Thanks, that makes sense"
  response: {{"decision": "NO_DEMAND", "query": null}}

  message: "Can you check where we left off on the auth module?"
  response: {{"decision": "DEMAND", "query": "auth module recent work and blockers"}}

Now respond for the latest message above. Return a JSON object only, no other text:
{{"decision": "DEMAND", "query": "..."}} or {{"decision": "NO_DEMAND", "query": null}}"""

    raw = chat.complete_json(prompt, temperature=0.0)
    if isinstance(raw, str):
        raw = json.loads(raw)

    # Use startswith to handle any model that adds trailing text to the decision value
    raw_decision = str(raw.get("decision", "NO_DEMAND")).strip().upper()
    decision = "DEMAND" if raw_decision.startswith("DEMAND") else "NO_DEMAND"
    query = raw.get("query")
    normalized_query = query if isinstance(query, str) and query else None

    # Guard against occasional LLM misclassification on explicit demand signals.
    if decision == "NO_DEMAND" and _looks_like_demand(message):
        return Intent(decision="DEMAND", query=normalized_query or _fallback_query(message))

    if decision == "DEMAND" and not normalized_query:
        normalized_query = _fallback_query(message)

    return Intent(decision=decision, query=normalized_query)


def on_context_message(
    memory: MemoryStore,
    chat: ChatModel,
    user_id: str,
    user_message: str,
    history: list[dict[str, str]],
    *,
    limit: int = 3,
) -> dict[str, Any]:
    """Pattern 2: gate Mem0 search with an intent classifier, then answer normally."""

    intent = detect_intent(chat, user_message, history)
    memories: list[dict[str, Any]] = []
    if intent.should_search:
        memories = memory.search(intent.query or "", user_id=user_id, limit=limit)

    context = "\n".join(f"- {_memory_text(item)}" for item in memories if _memory_text(item))
    system = "You are a helpful coding assistant."
    if context:
        system += f"\n\n[Relevant context:]\n{context}"

    reply = chat.complete_text(
        [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_message}]
    )

    return {
        "decision": intent.decision,
        "query": intent.query,
        "memories": memories,
        "reply": reply,
    }


def run_reflection(
    memory: MemoryStore,
    chat: ChatModel,
    user_id: str,
    *,
    max_items: int = 3,
) -> list[str]:
    """Pattern 3: pre-compute proactive hints after a session and store them."""

    envelope = memory.get_all(user_id=user_id)
    memories = envelope.get("results", [])
    if not memories:
        return []

    memory_text = "\n".join(f"- {_memory_text(item)}" for item in memories if _memory_text(item))
    prompt = f"""Review these stored memories and identify 2-3 things to proactively
surface at the start of the next session. Focus on unresolved blockers,
decisions that might be forgotten, and follow-up actions not yet completed.

Memories:
{memory_text}

Return a JSON array of short actionable strings. Maximum {max_items} items.
Return ONLY the JSON array, no other text."""

    raw = chat.complete_json(prompt, temperature=0.3)
    if isinstance(raw, str):
        raw = json.loads(raw)

    items = [str(item) for item in raw[:max_items] if str(item).strip()]
    if items:
        memory.add(
            [{"role": "system", "content": f"[PROACTIVE] {item}"} for item in items],
            user_id=user_id,
            metadata={"type": "proactive_hint"},
        )
    return items


def on_reflection_session_open(
    memory: MemoryStore,
    user_id: str,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Pattern 3 next-session retrieval: cheap search for pre-computed hints."""

    return memory.search("[PROACTIVE]", user_id=user_id, limit=limit)
