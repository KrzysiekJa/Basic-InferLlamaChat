"""Protocol definitions for LLM client abstraction.

This protocol describes the minimal surface our code expects from an LLM
client (both OpenAI-like and google-genai). It's used only for typing and
documentation; runtime behavior is duck-typed.
"""

from typing import Protocol, Any


class LLMClient(Protocol):
    """Protocol describing the minimal LLM client surface used by providers.

    Implementations may provide more attributes; this protocol focuses on the
    members the codebase references (responses, chat, models).
    """

    responses: Any
    chat: Any
    models: Any

    async def aclose(self) -> None:  # optional
        ...
