"""A fake AsyncAnthropic client: replays scripted streams, never touches the network."""

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
from anthropic.types import CitationsSearchResultLocation, TextBlock, Usage
from anthropic.types import Message as ApiMessage

THINKING = object()  # marker: emit a thinking content_block_start event
BLOCK_END = object()  # marker: close the current text block (content_block_stop)


def cite(source: int, cited_text: str = "cited paragraph") -> CitationsSearchResultLocation:
    """Script item: a citations_delta for the search result at index `source`."""
    return CitationsSearchResultLocation(
        type="search_result_location",
        cited_text=cited_text,
        search_result_index=source,
        start_block_index=0,
        end_block_index=1,
        source=f"source-{source}",
        title=f"Title {source}",
    )


_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def api_message(
    text: str = "",
    *,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_write_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
) -> ApiMessage:
    return ApiMessage(
        id="msg_test",
        type="message",
        role="assistant",
        model=model,
        content=[TextBlock(type="text", text=text)] if text else [],
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_write_tokens,
            cache_read_input_tokens=cache_read_tokens,
        ),
    )


def rate_limit_error() -> anthropic.RateLimitError:
    return anthropic.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=_REQUEST), body=None
    )


def auth_error() -> anthropic.AuthenticationError:
    return anthropic.AuthenticationError(
        "invalid x-api-key", response=httpx2.Response(401, request=_REQUEST), body=None
    )


def overloaded_error() -> anthropic.APIStatusError:
    """HTTP 529 when opening the stream (raised after the SDK's own retries)."""
    return anthropic.OverloadedError(
        "overloaded", response=httpx2.Response(529, request=_REQUEST), body=None
    )


def midstream_overloaded_error() -> anthropic.APIStatusError:
    """How the SDK surfaces an SSE `error` event: HTTP 200, error type in the body."""
    return anthropic.APIStatusError(
        "overloaded",
        response=httpx2.Response(200, request=_REQUEST),
        body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )


class Script:
    """One scripted call to messages.stream().

    `items` may contain text chunks (str), THINKING, citations (`cite()`), BLOCK_END, or an
    exception to raise mid-stream. Citations belong to the current text block, which ends
    at BLOCK_END or at the end of the script — as in the real API, a citation may appear
    before the text of its block.
    `open_error` is raised when the stream is opened (after SDK retries, in real life).
    """

    def __init__(
        self,
        items: list[Any] | None = None,
        *,
        final: ApiMessage | None = None,
        open_error: Exception | None = None,
        input_tokens: int = 100,
    ) -> None:
        self.items = items or []
        self.final = final
        self.open_error = open_error
        self.input_tokens = input_tokens


class FakeStream:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.closed = False
        self.current_message_snapshot = SimpleNamespace(
            usage=Usage(input_tokens=script.input_tokens, output_tokens=0)
        )

    async def __aenter__(self) -> "FakeStream":
        if self.script.open_error:
            raise self.script.open_error
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True

    async def __aiter__(self):
        block: list[CitationsSearchResultLocation] | None = None  # None = no open text block

        def block_stop():
            content_block = SimpleNamespace(type="text", citations=block or None)
            return SimpleNamespace(type="content_block_stop", content_block=content_block)

        for item in self.script.items:
            if isinstance(item, Exception):
                raise item
            if item is THINKING:
                yield SimpleNamespace(
                    type="content_block_start", content_block=SimpleNamespace(type="thinking")
                )
            elif item is BLOCK_END:
                yield block_stop()
                block = None
            elif isinstance(item, CitationsSearchResultLocation):
                block = [*(block or []), item]
                yield SimpleNamespace(type="citation", citation=item)
            else:
                block = block or []
                self.current_message_snapshot.usage.output_tokens += 1
                yield SimpleNamespace(type="text", text=item)
        if block is not None:
            yield block_stop()

    async def get_final_message(self) -> ApiMessage:
        if self.script.final is not None:
            return self.script.final
        text = "".join(i for i in self.script.items if isinstance(i, str))
        return api_message(text)


class FakeMessages:
    def __init__(self, scripts: list[Script]) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []
        self.streams: list[FakeStream] = []

    def stream(self, **params: Any) -> FakeStream:
        self.calls.append(params)
        stream = FakeStream(self.scripts.pop(0))
        self.streams.append(stream)
        return stream


class FakeClient:
    def __init__(self, *scripts: Script) -> None:
        self.messages = FakeMessages(list(scripts))
