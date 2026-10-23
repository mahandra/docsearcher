from decimal import Decimal

from django.conf import settings
from django.db import models


class Conversation(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="conversations"
    )
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

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
    input_tokens = models.PositiveIntegerField(null=True, blank=True)
    output_tokens = models.PositiveIntegerField(null=True, blank=True)
    cost_usd = models.DecimalField(
        max_digits=12, decimal_places=6, null=True, blank=True, default=None
    )

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self) -> str:
        return f"{self.role}: {self.content[:50]}"

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens or 0) + (self.output_tokens or 0)


def conversation_cost(conversation: Conversation) -> Decimal:
    total = conversation.messages.aggregate(total=models.Sum("cost_usd"))["total"]
    return total or Decimal("0")
