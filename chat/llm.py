"""LLM service layer — the only module that talks to the Anthropic SDK.

Views call `ChatService.stream_reply()` and receive plain domain events
(`ThinkingStarted`, `TextDelta`, `Completed`, `Failed`), so the SDK can be swapped
or mocked without touching the web layer.
"""

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types import MessageParam
from django.conf import settings
from django.utils import timezone

from .models import Conversation, Message
from .pricing import estimate_cost

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Error types the API may send inside an already-open stream (HTTP 200 + SSE "error" event).
# The SDK raises these as a generic APIStatusError, so we classify them by body.
RETRYABLE_STREAM_ERROR_TYPES = {"overloaded_error", "api_error", "rate_limit_error"}


# --- Domain events -----------------------------------------------------------------


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


StreamEvent = ThinkingStarted | TextDelta | Completed | Failed


@dataclass
class _Attempt:
    """State of one streaming request, shared between the retry loop and the stream."""

    parts: list[str] = field(default_factory=list)
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
    ) -> None:
        self.client = client
        self.model = model or settings.LLM_MODEL
        self.effort = settings.LLM_EFFORT if effort is None else effort
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.max_stream_retries = (
            settings.LLM_MAX_RETRIES if max_stream_retries is None else max_stream_retries
        )
        self.retry_base_delay = retry_base_delay

    def _request_params(self, messages: list[MessageParam]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": load_prompt("system"),
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
        params = self._request_params(messages)

        attempt_no = 0
        attempt = _Attempt()
        saved: Message | None = None
        try:
            while True:
                attempt = _Attempt()
                try:
                    async with aclosing(self._stream_once(params, attempt)) as events:
                        async for event in events:
                            yield event
                    final = attempt.final
                    saved = await self._save(
                        conversation,
                        attempt.parts,
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

                    logger.error(
                        "LLM request failed (request_id=%s): %s",
                        getattr(exc, "request_id", None),
                        exc,
                    )
                    status = Message.Status.INTERRUPTED if attempt.parts else Message.Status.ERROR
                    saved = await self._save(
                        conversation, attempt.parts, status, usage=attempt.snapshot_usage()
                    )
                    yield Failed(user_facing_error(exc), saved)
                    return
        finally:
            # Reached without a saved message when the client disconnects (the view's
            # generator is closed/cancelled). Keep whatever was already generated.
            if saved is None:
                await self._save(
                    conversation,
                    attempt.parts,
                    Message.Status.INTERRUPTED,
                    usage=attempt.snapshot_usage(),
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
                    yield TextDelta(event.text)
            attempt.final = await stream.get_final_message()

    async def _save(
        self,
        conversation: Conversation,
        parts: list[str],
        status: str,
        *,
        model: str | None = None,
        stop_reason: str = "",
        usage: Any = None,
    ) -> Message:
        model = model or self.model
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = estimate_cost(model, input_tokens, output_tokens)
        message = await Message.objects.acreate(
            conversation=conversation,
            role=Message.Role.ASSISTANT,
            content="".join(parts),
            status=status,
            stop_reason=stop_reason,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        await Conversation.objects.filter(pk=conversation.pk).aupdate(updated_at=timezone.now())
        return message


@cache
def get_chat_service() -> ChatService:
    # One client per process: it keeps a pool of HTTP connections to the API.
    client = anthropic.AsyncAnthropic(
        api_key=settings.ANTHROPIC_API_KEY or None,
        max_retries=settings.LLM_MAX_RETRIES,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )
    return ChatService(client)
