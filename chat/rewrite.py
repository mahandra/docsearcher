"""Rewrite a follow-up question into a standalone search query.

Retrieval sees only the text it searches for. A follow-up like "And for managers?" has
no subject of its own, so a small, fast model rewrites it with the conversation in view:
"What is the probation period for managers?". The answer is still generated for the
user's original question; only the search uses the rewritten one.

The reply is constrained to a JSON schema (structured outputs), so it always parses into
`SearchQuery`. Rewriting is an optimisation: on any failure the original question is
searched as before.

Takes the client from chat/llm.py (the only module that imports the Anthropic SDK).
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from .pricing import estimate_cost

logger = logging.getLogger(__name__)

# The latest exchanges carry the context of a follow-up; long answers are cut, the
# subject of the conversation is in their first lines.
MAX_HISTORY_MESSAGES = 6
MAX_ANSWER_CHARS = 1500


class SearchQuery(BaseModel):
    query: str = Field(description="The latest message as a standalone search question.")


@dataclass(frozen=True)
class Rewrite:
    query: str
    rewritten: bool = False  # the query differs from the question
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Decimal | None = None
    error: str = ""


def build_rewrite_input(history: list[dict[str, Any]], question: str) -> str:
    """`history`: earlier turns as API messages with plain-text content."""
    turns = []
    for message in history[-MAX_HISTORY_MESSAGES:]:
        text = message["content"]
        if message["role"] == "assistant" and len(text) > MAX_ANSWER_CHARS:
            text = text[:MAX_ANSWER_CHARS] + " […]"
        turns.append(f"<{message['role']}>\n{text}\n</{message['role']}>")
    conversation = "\n".join(turns)
    return (
        f"<conversation>\n{conversation}\n</conversation>\n\n"
        f"<latest_message>\n{question}\n</latest_message>"
    )


class QueryRewriter:
    def __init__(self, client: Any, *, model: str, effort: str = "low", prompt: str) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.prompt = prompt

    async def rewrite(self, history: list[dict[str, Any]], question: str) -> Rewrite:
        if not history:
            return Rewrite(query=question)
        try:
            response = await self.client.messages.parse(
                model=self.model,
                max_tokens=2000,
                system=self.prompt,
                messages=[{"role": "user", "content": build_rewrite_input(history, question)}],
                output_config={"effort": self.effort},
                output_format=SearchQuery,
            )
        except Exception as exc:  # API, network or validation error: search as is
            logger.warning("Query rewriting failed: %s", exc)
            return Rewrite(query=question, error=str(exc))

        usage = response.usage
        parsed = response.parsed_output
        query = parsed.query.strip() if parsed is not None else ""
        error = "" if query else f"no query (stop_reason={response.stop_reason})"
        return Rewrite(
            query=query or question,
            rewritten=bool(query) and query != question,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=estimate_cost(response.model, usage.input_tokens, usage.output_tokens),
            error=error,
        )
