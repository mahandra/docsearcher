import json
import logging
from collections.abc import AsyncIterator
from datetime import timedelta

from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.db.models import Q
from django.http import HttpRequest, HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from .llm import (
    Cited,
    Completed,
    Failed,
    Source,
    SourcesFound,
    TextDelta,
    ThinkingStarted,
    get_chat_service,
)
from .models import Conversation, Message, conversation_cost

logger = logging.getLogger(__name__)

# A generation that has not finished after this long is considered dead (crashed worker).
GENERATION_LOCK_TTL = timedelta(minutes=5)
TITLE_MAX_LENGTH = 60


# --- Auth --------------------------------------------------------------------------


def signup(request: HttpRequest) -> HttpResponse:
    form = UserCreationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect("chat:conversation_list")
    return render(request, "registration/signup.html", {"form": form})


# --- Pages -------------------------------------------------------------------------


@login_required
def conversation_list(request: HttpRequest) -> HttpResponse:
    conversations = Conversation.objects.filter(user=request.user)
    logger.info("Conversation list page accessed")
    return render(request, "chat/conversation_list.html", {"conversations": conversations})


@login_required
@require_POST
def conversation_create(request: HttpRequest) -> HttpResponse:
    conversation = Conversation.objects.create(user=request.user)
    return redirect("chat:conversation_detail", pk=conversation.pk)


@login_required
@require_POST
def conversation_delete(request: HttpRequest, pk: int) -> HttpResponse:
    get_object_or_404(Conversation, pk=pk, user=request.user).delete()
    return redirect("chat:conversation_list")


@login_required
def conversation_detail(request: HttpRequest, pk: int) -> HttpResponse:
    conversation = get_object_or_404(Conversation, pk=pk, user=request.user)
    messages = list(conversation.messages.prefetch_related("sources"))
    # A trailing user message has no reply yet (e.g. the page was reloaded mid-stream):
    # render a placeholder that (re)connects to the stream.
    awaiting_reply = bool(messages) and messages[-1].role == Message.Role.USER
    return render(
        request,
        "chat/conversation_detail.html",
        {
            "conversation": conversation,
            "chat_messages": messages,
            "awaiting_reply": awaiting_reply,
            "total_cost": conversation_cost(conversation),
        },
    )


@login_required
@require_POST
def message_create(request: HttpRequest, pk: int) -> HttpResponse:
    """HTMX endpoint: store the user message, return it plus a streaming placeholder."""
    conversation = get_object_or_404(Conversation, pk=pk, user=request.user)
    content = request.POST.get("content", "").strip()
    if not content:
        return HttpResponse(status=204)

    message = Message.objects.create(
        conversation=conversation, role=Message.Role.USER, content=content
    )
    if not conversation.title:
        conversation.title = content[:TITLE_MAX_LENGTH]
    conversation.save(update_fields=["title", "updated_at"])
    return render(
        request,
        "chat/partials/exchange.html",
        {"conversation": conversation, "message": message},
    )


# --- Streaming (SSE) ---------------------------------------------------------------


def sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def claim_generation(conversation: Conversation) -> bool:
    """Atomically mark the conversation as generating.

    A conditional UPDATE is race-free: of two concurrent requests only one
    matches the WHERE clause and gets rowcount == 1.
    """
    now = timezone.now()
    claimed = await Conversation.objects.filter(
        Q(generation_started_at__isnull=True)
        | Q(generation_started_at__lt=now - GENERATION_LOCK_TTL),
        pk=conversation.pk,
    ).aupdate(generation_started_at=now)
    return claimed == 1


async def release_generation(conversation: Conversation) -> None:
    await Conversation.objects.filter(pk=conversation.pk).aupdate(generation_started_at=None)


@require_GET
async def message_stream(request: HttpRequest, pk: int) -> HttpResponse:
    """Stream the assistant reply to the latest user message as Server-Sent Events.

    Events: `sources` {sources}, `thinking`, `token` {text}, `citation` {source, cited_text},
    `done` {usage, cost}, `failed` {error}.
    ("failed" rather than "error": EventSource reserves `error` for connection errors.)
    """
    user = await request.auser()
    if not user.is_authenticated:
        return HttpResponse(status=401)
    conversation = await Conversation.objects.filter(pk=pk, user=user).afirst()
    if conversation is None:
        return HttpResponse(status=404)

    last = await conversation.messages.order_by("-created_at", "-id").afirst()
    # 204 tells EventSource to stop reconnecting.
    if last is None or last.role != Message.Role.USER:
        return HttpResponse(status=204)
    if not await claim_generation(conversation):
        return HttpResponse(status=204)

    response = StreamingHttpResponse(_event_stream(conversation), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # disable proxy buffering (nginx)
    return response


async def _event_stream(conversation: Conversation) -> AsyncIterator[str]:
    try:
        async for event in get_chat_service().stream_reply(conversation):
            logger.debug("SSE %s conversation=%s", type(event).__name__, conversation.pk)
            match event:
                case SourcesFound(sources=sources):
                    yield sse("sources", {"sources": [_source_data(s) for s in sources]})
                case Cited(source=source, cited_text=cited_text):
                    yield sse("citation", {"source": source, "cited_text": cited_text})
                case ThinkingStarted():
                    yield sse("thinking", {})
                case TextDelta(text=text):
                    yield sse("token", {"text": text})
                case Completed(message=m):
                    yield sse("done", _message_meta(m))
                case Failed(error=error, message=m):
                    yield sse("failed", {"error": error, **_message_meta(m)})
    except Exception:
        logger.exception("Unexpected error while streaming conversation %s", conversation.pk)
        yield sse("failed", {"error": "Something went wrong. Please try again."})
    finally:
        await release_generation(conversation)


def _source_data(source: Source) -> dict:
    url = reverse("documents:document_detail", args=[source.document_id])
    return {
        "position": source.position,
        "document": source.document_name,
        "location": source.location,
        "url": f"{url}#chunk-{source.chunk_index}",
    }


def _message_meta(message: Message) -> dict:
    return {
        "message_id": message.pk,
        "status": message.status,
        "stop_reason": message.stop_reason,
        "model": message.model,
        "input_tokens": message.input_tokens,
        "output_tokens": message.output_tokens,
        "cost_usd": str(message.cost_usd) if message.cost_usd is not None else None,
        "retrieval_cost_usd": (
            str(message.retrieval_cost_usd) if message.retrieval_cost_usd is not None else None
        ),
    }
