from decimal import Decimal

from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils.functional import cached_property


class Conversation(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="conversations"
    )
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    # Set while a reply is being generated; guards against duplicate generations
    # (EventSource reconnects, two open tabs). See chat.views.claim_generation().
    generation_started_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.title or f"Conversation #{self.pk}"


class Message(models.Model):
    class Role(models.TextChoices):
        # Mirrors the roles of the Messages API. The system prompt is not stored per
        # conversation: it lives in chat/prompts/ and is sent separately on every call.
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"

    class Status(models.TextChoices):
        COMPLETE = "complete", "Complete"
        # The stream broke after some text was already sent to the browser.
        INTERRUPTED = "interrupted", "Interrupted"
        # The request failed before any text was produced.
        ERROR = "error", "Error"

    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=16, choices=Role.choices)
    content = models.TextField(blank=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.COMPLETE)
    created_at = models.DateTimeField(auto_now_add=True)

    # Why the model stopped: end_turn, max_tokens (truncated), refusal, ...
    stop_reason = models.CharField(max_length=32, blank=True)

    # Cost tracking — filled for assistant messages only.
    model = models.CharField(max_length=100, blank=True)
    # Input is reported in three parts; `input_tokens` is the uncached remainder only.
    input_tokens = models.PositiveIntegerField(null=True, blank=True)
    cache_write_tokens = models.PositiveIntegerField(null=True, blank=True)
    cache_read_tokens = models.PositiveIntegerField(null=True, blank=True)
    output_tokens = models.PositiveIntegerField(null=True, blank=True)
    cost_usd = models.DecimalField(
        max_digits=12, decimal_places=6, null=True, blank=True, default=None
    )
    # Retrieval for this answer: embedding the question (Voyage), billed separately.
    retrieval_tokens = models.PositiveIntegerField(null=True, blank=True)
    retrieval_cost_usd = models.DecimalField(
        max_digits=12, decimal_places=6, null=True, blank=True, default=None
    )

    # A follow-up rewritten into a standalone query (chat/rewrite.py) for the search;
    # empty when the question was searched as typed. The rewrite is billed separately.
    search_query = models.TextField(blank=True)
    rewrite_cost_usd = models.DecimalField(
        max_digits=12, decimal_places=6, null=True, blank=True, default=None
    )

    # Where the answer cites its sources: [{"offset": int, "source": int, "cited_text": str}].
    # `offset` is the position in `content` right after the cited text; `source` is the
    # MessageSource.position (the search_result_index the API returned).
    citations = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:50]}"

    @property
    def prompt_tokens(self) -> int:
        """The whole prompt: uncached + written to the cache + read from it."""
        return (
            (self.input_tokens or 0)
            + (self.cache_write_tokens or 0)
            + (self.cache_read_tokens or 0)
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + (self.output_tokens or 0)

    # --- Citations, for rendering. Prefetch `sources` when rendering many messages. ---

    @cached_property
    def citation_numbers(self) -> dict[int, int]:
        """Source position -> footnote number, numbered in order of first citation.

        The prompt holds ~8 sources but an answer usually cites a few; numbering them
        1, 2, 3 by appearance reads better than the raw positions (2, 5, 7).
        stream.js numbers live citations the same way.
        """
        numbers: dict[int, int] = {}
        for citation in self.citations:
            numbers.setdefault(citation["source"], len(numbers) + 1)
        return numbers

    def content_segments(self) -> list[dict]:
        """The answer split into text runs and citation markers, in order."""
        sources = {s.position: s for s in self.sources.all()}
        segments: list[dict] = []
        start = 0
        for citation in self.citations:
            offset = min(citation["offset"], len(self.content))
            if offset > start:
                segments.append({"text": self.content[start:offset]})
                start = offset
            segments.append(
                {
                    "number": self.citation_numbers[citation["source"]],
                    "cited_text": citation["cited_text"],
                    "source": sources.get(citation["source"]),
                }
            )
        if start < len(self.content):
            segments.append({"text": self.content[start:]})
        return segments

    def footnotes(self) -> list["MessageSource"]:
        """Cited sources in footnote order."""
        sources = {s.position: s for s in self.sources.all()}
        return [sources[p] for p in self.citation_numbers if p in sources]


class MessageSource(models.Model):
    """A chunk that was put into the prompt for an assistant message.

    Name and location are copied, so the citation survives the document being deleted
    or re-processed (chunks are replaced then).
    """

    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name="sources")
    position = models.PositiveSmallIntegerField()  # order in the prompt, 0-based
    chunk = models.ForeignKey(
        "documents.Chunk", on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    document = models.ForeignKey(
        "documents.Document", on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    document_name = models.CharField(max_length=255)
    chunk_index = models.PositiveIntegerField()
    location = models.CharField(max_length=500, blank=True)

    # Retrieval diagnostics — useful for evaluation (phase 4).
    score = models.FloatField()
    vector_rank = models.PositiveIntegerField(null=True, blank=True)
    keyword_rank = models.PositiveIntegerField(null=True, blank=True)
    # Whether the model actually cited it (vs. merely receiving it).
    cited = models.BooleanField(default=False)

    class Meta:
        ordering = ["message", "position"]
        constraints = [
            models.UniqueConstraint(fields=["message", "position"], name="unique_source_position"),
        ]

    def __str__(self) -> str:
        return f"[{self.number}] {self.document_name} {self.location}".strip()

    @property
    def url(self) -> str:
        """Link to the chunk on the document page; empty if the document is gone."""
        if self.document_id is None:
            return ""
        return reverse("documents:document_detail", args=[self.document_id]) + (
            f"#chunk-{self.chunk_index}"
        )


def conversation_cost(conversation: Conversation) -> Decimal:
    totals = conversation.messages.aggregate(
        llm=models.Sum("cost_usd"),
        retrieval=models.Sum("retrieval_cost_usd"),
        rewrite=models.Sum("rewrite_cost_usd"),
    )
    return sum((total or Decimal("0") for total in totals.values()), Decimal("0"))
