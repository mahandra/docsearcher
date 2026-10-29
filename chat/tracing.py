"""LLM observability via Langfuse — the only module that imports the Langfuse SDK.

One trace per assistant reply:

    chat-reply (span)          question -> answer; user and conversation as session
    ├── retrieval (retriever)  query -> retrieved chunks, embedding usage and cost
    └── claude (generation)    one per API attempt: model, prompt, answer, usage, cost,
                               time to first token; failed attempts are marked ERROR

Tracing is off (every call is a no-op) unless LANGFUSE_PUBLIC_KEY/SECRET_KEY are set,
so the app, tests and evals never depend on it. Spans are exported in batches by a
background thread (flushed at process exit); nothing here blocks or raises into the chat.

`stream_reply` is an async generator, and an OpenTelemetry "current span" must not be
kept open across a `yield` (the consumer may resume the generator in another context).
So observations are created explicitly (`start_observation`, not the `with` helpers)
and trace attributes are re-applied around each creation only.
"""

import logging
from datetime import UTC, datetime
from functools import cache
from typing import Any

from django.conf import settings
from langfuse import Langfuse, propagate_attributes

from .models import Message

logger = logging.getLogger(__name__)

TRACE_NAME = "chat-reply"


@cache
def get_langfuse() -> Langfuse | None:
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        return None
    return Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        base_url=settings.LANGFUSE_BASE_URL,
        environment=settings.LANGFUSE_ENVIRONMENT,
    )


def _content(value: Any) -> Any:
    """Questions, documents and answers are user data: send them only if allowed."""
    return value if settings.LANGFUSE_CAPTURE_CONTENT else None


class ReplyTrace:
    """Records one reply. Without a Langfuse client every method does nothing."""

    def __init__(
        self,
        client: Langfuse | None,
        *,
        user_id: int,
        conversation_id: int,
        question: str,
        tags: list[str] | None = None,
    ) -> None:
        self._client = client
        self._attributes = {
            "user_id": str(user_id),
            "session_id": f"conversation-{conversation_id}",
            "trace_name": TRACE_NAME,
            "tags": tags,
        }
        self._root = self._retrieval = self._generation = None
        self._first_token_seen = False
        if client is not None:
            self._root = self._safe(
                lambda: self._start(client, name=TRACE_NAME, input=_content(question))
            )

    @property
    def trace_id(self) -> str | None:
        return self._root.trace_id if self._root is not None else None

    def start_retrieval(self, query: str) -> None:
        self._retrieval = self._child(name="retrieval", as_type="retriever", input=_content(query))

    def end_retrieval(self, retrieval: Any) -> None:
        """`retrieval` is a chat.llm.Retrieval (not imported: llm imports this module)."""
        if self._retrieval is None:
            return
        sources = [
            {
                "title": s.title,
                "score": round(s.score, 4),
                "vector_rank": s.vector_rank,
                "keyword_rank": s.keyword_rank,
            }
            for s in retrieval.sources
        ]
        self._end(
            self._retrieval,
            output=_content(sources),
            metadata={
                "sources": len(retrieval.sources),
                "embedding_model": retrieval.embedding_model,
                "embedding_tokens": retrieval.embedding_tokens,
            },
        )
        self._retrieval = None

    def start_generation(self, params: dict[str, Any]) -> None:
        self._first_token_seen = False
        self._generation = self._child(
            name="claude",
            as_type="generation",
            model=params["model"],
            model_parameters={
                "max_tokens": params["max_tokens"],
                "effort": params.get("output_config", {}).get("effort", ""),
            },
            input=_content({"system": params["system"], "messages": params["messages"]}),
        )

    def first_token(self) -> None:
        """Time to first token is what the user feels as latency."""
        if self._generation is None or self._first_token_seen:
            return
        self._first_token_seen = True
        self._safe(lambda: self._generation.update(completion_start_time=datetime.now(UTC)))

    def fail_generation(self, error: str) -> None:
        """A failed attempt that will be retried."""
        if self._generation is not None:
            self._end(self._generation, level="ERROR", status_message=error)
            self._generation = None

    def finish(self, message: Message, error: str = "") -> None:
        """The reply is saved: close the last generation and the trace."""
        failed = message.status != Message.Status.COMPLETE
        level = "ERROR" if failed else None
        status_message = error or (message.status if failed else None)
        if self._generation is not None:
            usage = {}
            if message.input_tokens is not None:
                usage = {"input": message.input_tokens, "output": message.output_tokens or 0}
            self._end(
                self._generation,
                model=message.model,
                output=_content(message.content),
                # Our own prices (chat/pricing.py): Langfuse may not know new models.
                usage_details=usage or None,
                cost_details={"total": float(message.cost_usd)} if message.cost_usd else None,
                metadata={"stop_reason": message.stop_reason, "citations": len(message.citations)},
                level=level,
                status_message=status_message,
            )
            self._generation = None
        if self._root is not None:
            self._end(
                self._root,
                output=_content(message.content),
                metadata={"message_id": message.pk, "status": message.status},
                level=level,
                status_message=status_message,
            )
            self._root = None

    # --- helpers --------------------------------------------------------------------

    def _start(self, parent: Any, **kwargs: Any) -> Any:
        # Attributes propagate to observations created inside the block; the block ends
        # before any `await`/`yield`, so no OpenTelemetry context outlives it.
        with propagate_attributes(**{k: v for k, v in self._attributes.items() if v}):
            return parent.start_observation(**kwargs)

    def _child(self, **kwargs: Any) -> Any:
        if self._root is None:
            return None
        return self._safe(lambda: self._start(self._root, **kwargs))

    def _end(self, observation: Any, **update: Any) -> None:
        def end():
            observation.update(**{k: v for k, v in update.items() if v is not None})
            observation.end()

        self._safe(end)

    @staticmethod
    def _safe(action: Any) -> Any:
        # Observability must never break the chat.
        try:
            return action()
        except Exception:
            logger.exception("Langfuse tracing failed")
            return None


def start_reply_trace(
    *, user_id: int, conversation_id: int, question: str, tags: list[str] | None = None
) -> ReplyTrace:
    return ReplyTrace(
        get_langfuse(),
        user_id=user_id,
        conversation_id=conversation_id,
        question=question,
        tags=tags,
    )
