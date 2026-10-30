"""LLM service layer — the only module that talks to the Anthropic SDK.

Views call `ChatService.stream_reply()` and receive plain domain events
(`SourcesFound`, `ThinkingStarted`, `TextDelta`, `Cited`, `Completed`, `Failed`), so the
SDK can be swapped or mocked without touching the web layer.

RAG: before each reply the latest question is searched in the user's documents and the
best chunks are sent as `search_result` blocks with citations enabled. Claude then
attaches citations (which result, which text) to the parts of the answer they support;
no "[1]" markers have to be prompted for or parsed.
"""

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types import MessageParam
from asgiref.sync import sync_to_async
from django.conf import settings
from django.utils import timezone

from documents.models import Chunk, Document
from documents.retrieval import SearchHit, search

from .models import Conversation, Message, MessageSource
from .pricing import estimate_cost, estimate_embedding_cost
from .tracing import ReplyTrace, start_reply_trace

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Error types the API may send inside an already-open stream (HTTP 200 + SSE "error" event).
# The SDK raises these as a generic APIStatusError, so we classify them by body.
RETRYABLE_STREAM_ERROR_TYPES = {"overloaded_error", "api_error", "rate_limit_error"}

# Sent instead of search results when nothing was found, so the model knows it searched.
NO_SOURCES_NOTE = "No passages relevant to this question were found in the user's documents."


# --- Retrieval ---------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """A retrieved chunk as sent to the model; `position` is its search_result_index."""

    position: int
    chunk_id: int
    document_id: int
    document_name: str
    chunk_index: int
    location: str
    text: str
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None

    @property
    def title(self) -> str:
        """Document name plus location, as the model sees it in the search_result title."""
        return f"{self.document_name} ({self.location})" if self.location else self.document_name

    @classmethod
    def from_hit(cls, position: int, hit: SearchHit) -> "Source":
        chunk = hit.chunk
        return cls(
            position=position,
            chunk_id=chunk.pk,
            document_id=chunk.document_id,
            document_name=chunk.document.original_name,
            chunk_index=chunk.index,
            location=chunk.location,
            text=chunk.text,
            score=hit.score,
            vector_rank=hit.vector_rank,
            keyword_rank=hit.keyword_rank,
        )


@dataclass(frozen=True)
class Retrieval:
    sources: list[Source]
    embedding_model: str = ""
    embedding_tokens: int = 0


Retriever = Callable[[int, str], Awaitable[Retrieval]]


async def retrieve_sources(user_id: int, query: str, *, embedder=None) -> Retrieval:
    result = await sync_to_async(search)(user_id, query, embedder=embedder)
    return Retrieval(
        sources=[Source.from_hit(i, hit) for i, hit in enumerate(result.hits)],
        embedding_model=result.embedding_model,
        embedding_tokens=result.embedding_tokens,
    )


def search_result_block(source: Source) -> dict[str, Any]:
    """One retrieved chunk as a `search_result` content block.

    Claude cites whole text blocks, never substrings, so the chunk is split into
    paragraphs: citations then point at (and quote) a paragraph, not the whole chunk.
    """
    paragraphs = [p.strip() for p in source.text.split("\n\n") if p.strip()] or [source.text]
    return {
        "type": "search_result",
        "source": f"document:{source.document_id}#chunk-{source.chunk_index}",
        "title": source.title,
        "content": [{"type": "text", "text": p} for p in paragraphs],
        # All-or-nothing per request: every search_result must use the same setting.
        "citations": {"enabled": True},
    }


def build_question_turn(question: str, sources: list[Source]) -> list[dict[str, Any]]:
    """Content of the latest user turn: search results first, then the question.

    Only the latest turn carries sources. Earlier turns stay plain text, otherwise every
    follow-up would re-send (and re-bill) all previously retrieved chunks.
    """
    if sources:
        blocks = [search_result_block(s) for s in sources]
    else:
        blocks = [{"type": "text", "text": NO_SOURCES_NOTE}]
    return [*blocks, {"type": "text", "text": question}]


# --- Domain events -----------------------------------------------------------------


@dataclass(frozen=True)
class SourcesFound:
    """Retrieval finished; these chunks were sent to the model (possibly none)."""

    sources: list[Source]


@dataclass(frozen=True)
class Cited:
    """The text streamed so far is supported by this source (Source.position)."""

    source: int
    cited_text: str


@dataclass(frozen=True)
class ThinkingStarted:
    """The model started reasoning before answering (no visible text yet)."""


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class Completed:
    message: Message


@dataclass(frozen=True)
class Failed:
    error: str  # safe to show to the user
    message: Message  # the saved assistant message (status interrupted or error)


StreamEvent = SourcesFound | ThinkingStarted | TextDelta | Cited | Completed | Failed


@dataclass
class _Attempt:
    """State of one streaming request, shared between the retry loop and the stream."""

    parts: list[str] = field(default_factory=list)
    length: int = 0  # characters streamed so far = where a finished block ends
    citations: list[dict[str, Any]] = field(default_factory=list)
    stream: Any = None
    final: Any = None  # anthropic.types.Message once the stream finished

    def snapshot_usage(self) -> Any:
        """Best-effort usage of an unfinished stream (input tokens are known early)."""
        if self.stream is None:
            return None
        try:
            return self.stream.current_message_snapshot.usage
        except Exception:
            return None


# --- Prompt / history --------------------------------------------------------------


@cache
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


async def build_messages(
    conversation: Conversation, max_messages: int | None = None
) -> list[MessageParam]:
    """Turn the stored conversation into the `messages` list for the API.

    The model has no memory between requests: the whole (windowed) history is sent
    every time, and every token of it is billed as input.
    """
    if max_messages is None:
        max_messages = settings.LLM_MAX_HISTORY_MESSAGES

    recent = [
        m
        async for m in conversation.messages.order_by("-created_at", "-id").exclude(content="")[
            :max_messages
        ]
    ]
    recent.reverse()

    messages: list[MessageParam] = []
    for m in recent:
        # Failed replies leave consecutive user turns; merge same-role neighbours so
        # the history always alternates user/assistant.
        if messages and messages[-1]["role"] == m.role:
            messages[-1]["content"] = f"{messages[-1]['content']}\n\n{m.content}"
        else:
            messages.append({"role": m.role, "content": m.content})

    # The window may cut the history so that it starts with an assistant turn.
    while messages and messages[0]["role"] != Message.Role.USER:
        messages.pop(0)
    return messages


CACHE_BREAKPOINT = {"type": "ephemeral"}  # 5-minute TTL, refreshed by every read


def cached(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text, "cache_control": CACHE_BREAKPOINT}]


def add_cache_breakpoints(
    system: str, messages: list[MessageParam]
) -> tuple[list[dict[str, Any]], list[MessageParam]]:
    """Mark the prompt prefixes that the next requests will repeat byte for byte.

    The cache matches a prefix (system, then messages) up to a marked block:
    - the system prompt is the same for every request of every user;
    - the history (all but the latest turn) starts the next request of this conversation.
    The latest turn is not marked: its search results are not kept in the history, so
    its bytes never recur and caching it would only pay the write premium. Prefixes
    shorter than the model's minimum (512 tokens) are silently not cached.
    """
    if len(messages) > 1 and isinstance(messages[-2]["content"], str):
        last = messages[-2]
        messages = [
            *messages[:-2],
            {"role": last["role"], "content": cached(last["content"])},
            messages[-1],
        ]
    return cached(system), messages


# --- Errors ------------------------------------------------------------------------


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, anthropic.APIConnectionError):  # includes timeouts
        return True
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code == 429 or exc.status_code >= 500:
            return True
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error") if isinstance(body.get("error"), dict) else {}
        return error.get("type") in RETRYABLE_STREAM_ERROR_TYPES
    return False


def user_facing_error(exc: Exception) -> str:
    # Most specific first: RateLimitError etc. are subclasses of APIStatusError.
    if isinstance(exc, anthropic.AuthenticationError):
        return "The AI service is not configured correctly (invalid API key)."
    if isinstance(exc, anthropic.RateLimitError):
        return "Too many requests right now. Please wait a moment and try again."
    if isinstance(exc, anthropic.APITimeoutError):
        return "The AI service took too long to respond. Please try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the AI service. Please try again."
    if is_retryable(exc):
        return "The AI service is temporarily overloaded. Please try again."
    return "The AI service rejected the request."


# --- Service -----------------------------------------------------------------------


class ChatService:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        model: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        max_stream_retries: int | None = None,
        retry_base_delay: float = 1.0,
        retriever: Retriever = retrieve_sources,
        tracer: Callable[..., ReplyTrace] = start_reply_trace,
        prompt_caching: bool | None = None,
        fallback_model: str | None = None,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.tracer = tracer
        self.model = model or settings.LLM_MODEL
        self.effort = settings.LLM_EFFORT if effort is None else effort
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.max_stream_retries = (
            settings.LLM_MAX_RETRIES if max_stream_retries is None else max_stream_retries
        )
        self.retry_base_delay = retry_base_delay
        self.prompt_caching = (
            settings.LLM_PROMPT_CACHING if prompt_caching is None else prompt_caching
        )
        fallback_model = settings.LLM_FALLBACK_MODEL if fallback_model is None else fallback_model
        self.fallback_model = fallback_model if fallback_model != self.model else ""

    def _request_params(self, messages: list[MessageParam]) -> dict[str, Any]:
        system: Any = load_prompt("system")
        if self.prompt_caching:
            system, messages = add_cache_breakpoints(system, messages)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        return params

    async def stream_reply(self, conversation: Conversation) -> AsyncIterator[StreamEvent]:
        """Generate the assistant reply to the latest user message, streaming it.

        Always ends with exactly one Completed or Failed event, and always persists an
        assistant Message — even if the client disconnects mid-stream.
        """
        messages = await build_messages(conversation)
        if not messages or messages[-1]["role"] != Message.Role.USER:
            raise ValueError("The conversation must end with a user message")

        question = messages[-1]["content"]
        trace = self.tracer(
            user_id=conversation.user_id, conversation_id=conversation.pk, question=question
        )
        trace.start_retrieval(question)
        retrieval = await self.retriever(conversation.user_id, question)
        trace.end_retrieval(retrieval)
        yield SourcesFound(retrieval.sources)
        messages[-1] = {"role": "user", "content": build_question_turn(question, retrieval.sources)}
        params = self._request_params(messages)

        attempt_no = 0
        attempt = _Attempt()
        saved: Message | None = None
        try:
            while True:
                attempt = _Attempt()
                trace.start_generation(params)
                try:
                    async with aclosing(self._stream_once(params, attempt)) as events:
                        async for event in events:
                            if isinstance(event, ThinkingStarted | TextDelta):
                                trace.first_token()
                            yield event
                    final = attempt.final
                    saved = await self._save(
                        conversation,
                        attempt,
                        retrieval,
                        trace,
                        Message.Status.COMPLETE,
                        model=final.model,
                        stop_reason=final.stop_reason or "",
                        usage=final.usage,
                    )
                    if final.stop_reason == "refusal":
                        logger.info("Model refused: %s", final.stop_details)
                    yield Completed(saved)
                    return
                except anthropic.APIError as exc:
                    # The SDK already retried failures to *open* the stream (max_retries).
                    # Here we only retry failures inside the stream, and only while no
                    # text was sent to the browser — otherwise the user would see the
                    # answer start over (or duplicated).
                    if (
                        attempt.stream is not None  # failed mid-stream, not on open
                        and not attempt.parts
                        and is_retryable(exc)
                        and attempt_no < self.max_stream_retries
                    ):
                        attempt_no += 1
                        trace.fail_generation(str(exc))
                        delay = self.retry_base_delay * 2 ** (attempt_no - 1)
                        delay *= random.uniform(1, 1.5)  # jitter
                        logger.warning(
                            "Retryable stream error (attempt %s): %s; sleeping %.1fs",
                            attempt_no,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if self._should_fall_back(params, attempt, exc):
                        # Retries did not help: the model is overloaded or unavailable
                        # for longer. Another model has its own capacity (and rate
                        # limits), but not this one's prompt cache.
                        logger.warning(
                            "Model %s unavailable (%s); falling back to %s",
                            params["model"],
                            exc,
                            self.fallback_model,
                        )
                        trace.fail_generation(str(exc))
                        params = {**params, "model": self.fallback_model}
                        attempt_no = 0
                        continue

                    logger.error(
                        "LLM request failed (request_id=%s): %s",
                        getattr(exc, "request_id", None),
                        exc,
                    )
                    status = Message.Status.INTERRUPTED if attempt.parts else Message.Status.ERROR
                    saved = await self._save(
                        conversation,
                        attempt,
                        retrieval,
                        trace,
                        status,
                        model=params["model"],
                        usage=attempt.snapshot_usage(),
                        error=str(exc),
                    )
                    yield Failed(user_facing_error(exc), saved)
                    return
        finally:
            # Reached without a saved message when the client disconnects (the view's
            # generator is closed/cancelled). Keep whatever was already generated.
            if saved is None:
                await self._save(
                    conversation,
                    attempt,
                    retrieval,
                    trace,
                    Message.Status.INTERRUPTED,
                    model=params["model"],
                    usage=attempt.snapshot_usage(),
                    error="client disconnected",
                )

    def _should_fall_back(self, params: dict[str, Any], attempt: _Attempt, exc: Exception) -> bool:
        """Once, before any text was streamed, and only for capacity/availability errors:
        a 4xx (bad request, auth) would fail the same way on any model."""
        return (
            bool(self.fallback_model)
            and params["model"] != self.fallback_model
            and not attempt.parts
            and is_retryable(exc)
        )

    async def _stream_once(
        self, params: dict[str, Any], attempt: _Attempt
    ) -> AsyncIterator[StreamEvent]:
        # Leaving the `async with` closes the HTTP connection, which stops generation —
        # so an abandoned request stops costing output tokens.
        async with self.client.messages.stream(**params) as stream:
            attempt.stream = stream
            async for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "thinking":
                    yield ThinkingStarted()
                elif event.type == "text":
                    attempt.parts.append(event.text)
                    attempt.length += len(event.text)
                    yield TextDelta(event.text)
                elif event.type == "content_block_stop" and event.content_block.type == "text":
                    # The answer arrives as several text blocks; citations belong to a whole
                    # block. Their `citation` deltas may come before the block's text, so we
                    # wait for the block to finish and mark its end.
                    for citation in self._block_citations(event.content_block, attempt.length):
                        attempt.citations.append(citation)
                        yield Cited(citation["source"], citation["cited_text"])
            attempt.final = await stream.get_final_message()

    @staticmethod
    def _block_citations(block: Any, offset: int) -> list[dict[str, Any]]:
        """One citation per source in the block; several cited paragraphs are joined."""
        by_source: dict[int, dict[str, Any]] = {}
        for c in block.citations or []:
            if c.type != "search_result_location":
                continue
            if c.search_result_index in by_source:
                by_source[c.search_result_index]["cited_text"] += "\n\n" + c.cited_text
            else:
                by_source[c.search_result_index] = {
                    "offset": offset,
                    "source": c.search_result_index,
                    "cited_text": c.cited_text,
                }
        return list(by_source.values())

    async def _save(
        self,
        conversation: Conversation,
        attempt: _Attempt,
        retrieval: Retrieval,
        trace: ReplyTrace,
        status: str,
        *,
        model: str | None = None,
        stop_reason: str = "",
        usage: Any = None,
        error: str = "",
    ) -> Message:
        model = model or self.model
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        cache_write_tokens = getattr(usage, "cache_creation_input_tokens", None)
        cache_read_tokens = getattr(usage, "cache_read_input_tokens", None)
        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = estimate_cost(
                model,
                input_tokens,
                output_tokens,
                cache_write_tokens=cache_write_tokens or 0,
                cache_read_tokens=cache_read_tokens or 0,
            )
        retrieval_cost = None
        if retrieval.embedding_model:
            retrieval_cost = estimate_embedding_cost(
                retrieval.embedding_model, retrieval.embedding_tokens
            )
        message = await Message.objects.acreate(
            conversation=conversation,
            role=Message.Role.ASSISTANT,
            content="".join(attempt.parts),
            status=status,
            stop_reason=stop_reason,
            model=model,
            input_tokens=input_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            retrieval_tokens=retrieval.embedding_tokens if retrieval.embedding_model else None,
            retrieval_cost_usd=retrieval_cost,
            citations=attempt.citations,
        )
        cited = {c["source"] for c in attempt.citations}
        # Generation takes a while: a document may have been deleted or re-processed
        # (chunks replaced) since retrieval. Keep the copied name/location, drop the links.
        chunk_ids = {s.chunk_id for s in retrieval.sources}
        live_chunks = {
            pk async for pk in Chunk.objects.filter(pk__in=chunk_ids).values_list("pk", flat=True)
        }
        document_ids = {s.document_id for s in retrieval.sources}
        live_documents = {
            pk
            async for pk in Document.objects.filter(pk__in=document_ids).values_list(
                "pk", flat=True
            )
        }
        await MessageSource.objects.abulk_create(
            MessageSource(
                message=message,
                position=s.position,
                chunk_id=s.chunk_id if s.chunk_id in live_chunks else None,
                document_id=s.document_id if s.document_id in live_documents else None,
                document_name=s.document_name[:255],
                chunk_index=s.chunk_index,
                location=s.location[:500],
                score=s.score,
                vector_rank=s.vector_rank,
                keyword_rank=s.keyword_rank,
                cited=s.position in cited,
            )
            for s in retrieval.sources
        )
        await Conversation.objects.filter(pk=conversation.pk).aupdate(updated_at=timezone.now())
        trace.finish(message, error)
        return message


@cache
def get_chat_service() -> ChatService:
    # One client per process: it keeps a pool of HTTP connections to the API.
    client = anthropic.AsyncAnthropic(
        api_key=settings.ANTHROPIC_API_KEY or None,
        max_retries=settings.LLM_MAX_RETRIES,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        # The SDK's own OpenTelemetry spans would reach Langfuse as separate traces with
        # no user/session; chat/tracing.py records these calls with more detail.
        open_telemetry=False,
    )
    return ChatService(client)
