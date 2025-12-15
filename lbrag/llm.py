import os
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

from openai import OpenAI


@dataclass
class LLMUsage:
    """Accumulates token usage across requests."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0

    def add(self, usage: Any) -> None:
        """Add usage stats from an OpenAI response."""
        if usage is None:
            return
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        self.request_count += 1

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0


class LLMClient:
    """
    Thin wrapper over OpenAI client that records token usage.
    Use a single shared instance per experiment run to aggregate usage.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.usage = LLMUsage()

    def chat(
        self,
        messages: Sequence[dict],
        model: str,
        **kwargs: Any,
    ) -> Tuple[str, Any]:
        """
        Execute a chat completion.

        Returns (content, raw_response).
        The wrapper records usage into self.usage.
        """
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        self._record_usage(response)
        content = (response.choices[0].message.content or "").strip()
        return content, response

    def embed(
        self,
        input: str | Sequence[str],
        model: str = "text-embedding-3-small",
        **kwargs: Any,
    ) -> Any:
        """
        Execute an embeddings call.
        Returns raw response; usage (if provided by API) is accumulated.
        """
        response = self._client.embeddings.create(model=model, input=input, **kwargs)
        self._record_usage(response)
        return response

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.add(usage)


def format_usage_summary(usage: LLMUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "request_count": usage.request_count,
    }
