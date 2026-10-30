from django.contrib import admin

from .models import Conversation, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    fields = (
        "role",
        "status",
        "content",
        "model",
        "input_tokens",
        "cache_write_tokens",
        "cache_read_tokens",
        "output_tokens",
        "cost_usd",
    )
    readonly_fields = fields


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("__str__", "user", "created_at", "updated_at")
    list_filter = ("user",)
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = (
        "conversation",
        "role",
        "status",
        "model",
        "input_tokens",
        "cache_read_tokens",
        "output_tokens",
        "cost_usd",
        "created_at",
    )
    list_filter = ("role", "status", "model", "uncited")
