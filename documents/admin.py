from django.contrib import admin

from .models import Chunk, Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = (
        "original_name",
        "user",
        "status",
        "page_count",
        "chunk_count",
        "embedding_tokens",
        "embedding_cost_usd",
        "created_at",
    )
    list_filter = ("status", "user")
    readonly_fields = ("sha256", "size_bytes", "processed_at")


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "index", "page_start", "page_end", "section", "token_count")
    list_filter = ("document",)
    exclude = ("embedding",)  # 1024 floats are not useful in a form
