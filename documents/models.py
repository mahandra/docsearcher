import uuid
from pathlib import Path

from django.conf import settings
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVector, SearchVectorField
from django.db import models
from pgvector.django import HnswIndex, VectorField

# Fixed by the database schema: vectors of different sizes (or from different models)
# cannot be compared. Changing the embedding model means re-embedding every chunk.
EMBEDDING_DIMENSIONS = 1024

# Text search configurations (stemmers) for full-text search. Documents are mixed
# English/Russian, so every chunk is indexed with both: "contracts" matches "contract",
# "договоры" matches "договор". A word unknown to a stemmer is still indexed as-is.
SEARCH_CONFIGS = ("english", "russian")


def document_upload_to(instance: "Document", filename: str) -> str:
    # Never trust the client's filename on disk; keep only a normalised extension.
    ext = Path(filename).suffix.lower()
    return f"documents/{instance.user_id}/{uuid.uuid4().hex}{ext}"


class Document(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="documents"
    )
    file = models.FileField(upload_to=document_upload_to)
    original_name = models.CharField(max_length=255)
    sha256 = models.CharField(max_length=64)
    size_bytes = models.PositiveBigIntegerField()

    status = models.CharField(max_length=16, choices=Status.choices, default=Status.UPLOADED)
    error = models.TextField(blank=True)

    page_count = models.PositiveIntegerField(null=True, blank=True)
    chunk_count = models.PositiveIntegerField(default=0)

    # Embedding cost tracking, like Message in the chat app.
    embedding_model = models.CharField(max_length=100, blank=True)
    embedding_tokens = models.PositiveIntegerField(null=True, blank=True)
    embedding_cost_usd = models.DecimalField(max_digits=12, decimal_places=6, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            # The same file is processed (and paid for) once per user.
            models.UniqueConstraint(fields=["user", "sha256"], name="unique_document_per_user"),
        ]

    def __str__(self) -> str:
        return self.original_name

    @property
    def is_final(self) -> bool:
        return self.status in (self.Status.READY, self.Status.FAILED)


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    index = models.PositiveIntegerField()  # position within the document
    text = models.TextField()

    # Where the chunk comes from — the basis for citations in phase 3.
    page_start = models.PositiveIntegerField(null=True, blank=True)  # PDF only, 1-based
    page_end = models.PositiveIntegerField(null=True, blank=True)
    section = models.CharField(max_length=500, blank=True)  # nearest heading (DOCX)

    # Estimated from the character count; exact usage is billed per batch.
    token_count = models.PositiveIntegerField()
    embedding = VectorField(dimensions=EMBEDDING_DIMENSIONS)

    # Computed by Postgres on every insert/update (GENERATED ALWAYS AS ... STORED), so it
    # can never get out of sync with `text`. Used for keyword search alongside vectors.
    search_vector = models.GeneratedField(
        expression=SearchVector("text", config=SEARCH_CONFIGS[0])
        + SearchVector("text", config=SEARCH_CONFIGS[1]),
        output_field=SearchVectorField(),
        db_persist=True,
    )

    class Meta:
        ordering = ["document", "index"]
        constraints = [
            models.UniqueConstraint(fields=["document", "index"], name="unique_chunk_index"),
        ]
        indexes = [
            # Approximate nearest-neighbour index for cosine distance.
            HnswIndex(
                name="chunk_embedding_hnsw",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
            GinIndex(name="chunk_search_vector_gin", fields=["search_vector"]),
        ]

    def __str__(self) -> str:
        return f"{self.document} #{self.index}"

    @property
    def location(self) -> str:
        if self.page_start is not None:
            if self.page_end and self.page_end != self.page_start:
                return f"pp. {self.page_start}–{self.page_end}"
            return f"p. {self.page_start}"
        return self.section
