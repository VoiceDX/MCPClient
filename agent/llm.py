"""LLM helper utilities for interacting with the OpenAI Completion API."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, List

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class CompletionMessage:
    """Represents a message sent to the LLM."""

    role: str
    content: str


class LLMClient:
    """Thin wrapper around the OpenAI client for structured prompting."""

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        print(
            "[agent/llm.py][LLMClient.__init__][Start] "
            f"model={model}"
        )
        self._client = OpenAI()
        self.model = model
        print(
            "[agent/llm.py][LLMClient.__init__][End] client_initialized=True"
        )

    def complete(self, messages: List[CompletionMessage], **kwargs: Any) -> str:
        """Send a completion request and return the model's response text."""
        print(
            "[agent/llm.py][LLMClient.complete][Start] "
            f"messages_count={len(messages)} kwargs={kwargs}"
        )
        payload = [message.__dict__ for message in messages]
        logger.debug("Sending completion payload: %s", json.dumps(payload, ensure_ascii=False, indent=2))
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": message.role, "content": message.content} for message in messages],
            **kwargs,
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        logger.debug("Received completion text: %s", text)
        print(
            "[agent/llm.py][LLMClient.complete][End] "
            f"response_length={len(text)}"
        )
        return text
