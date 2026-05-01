from __future__ import annotations

import json
import os
from typing import Any


class Mem0MemoryStore:
    """Adapter around mem0.MemoryClient that satisfies the MemoryStore protocol."""

    def __init__(self, api_key: str | None = None) -> None:
        from mem0 import MemoryClient

        self.client = MemoryClient(api_key=api_key or os.environ["MEM0_API_KEY"])

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"user_id": user_id}
        if metadata:
            kwargs["metadata"] = metadata
        return self.client.add(messages, **kwargs)

    def search(self, query: str, *, user_id: str, limit: int = 3) -> list[dict[str, Any]]:
        try:
            # Older mem0 versions accept top-level user_id.
            result = self.client.search(query, user_id=user_id, limit=limit)
        except ValueError as exc:
            # Newer mem0 versions require entity filters instead of top-level params.
            if "Top-level entity parameters" not in str(exc):
                raise
            result = self.client.search(query, filters={"user_id": user_id}, limit=limit)

        # Normalize both result shapes:
        # - legacy: list[dict]
        # - v1.1: {"results": list[dict], ...}
        if isinstance(result, dict):
            return result.get("results", [])
        return result

    def get_all(self, *, user_id: str) -> dict[str, Any]:
        return self.client.get_all(user_id=user_id)


class OpenRouterChatModel:
    """
    Adapter around OpenRouter using the OpenAI-compatible SDK surface.

    Setup:
        export OPENROUTER_API_KEY=your-key-from-openrouter.ai
        export OPENROUTER_MODEL=openai/gpt-4o-mini
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.environ["OPENROUTER_API_KEY"], base_url=base_url)
        self.model = model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    def complete_json(self, prompt: str, *, temperature: float = 0.0) -> Any:
        text = self.complete_text([{"role": "user", "content": prompt}], temperature=temperature)
        return json.loads(text.strip())

    def complete_text(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


# Backward compatibility for older imports.
OpenAIChatModel = OpenRouterChatModel
