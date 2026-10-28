"""Load the eval corpus into the database for a dedicated eval user.

The documents go through the real ingestion code (parse -> chunk -> embed), synchronously,
so the eval measures exactly what users get. Unchanged files are reused (same SHA-256),
so repeated runs do not pay for embeddings again.
"""

import hashlib
import logging

from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile

from documents.models import Document
from documents.tasks import ingest

from .corpus import render
from .dataset import Dataset

logger = logging.getLogger(__name__)

EVAL_USERNAME = "docqa-eval"


def get_eval_user():
    user, created = get_user_model().objects.get_or_create(username=EVAL_USERNAME)
    if created:
        user.set_unusable_password()  # nobody logs in as the eval user
        user.save(update_fields=["password"])
    return user


def ensure_corpus(dataset: Dataset, *, reingest: bool = False) -> dict[str, Document]:
    """Return {document key: ready Document}, ingesting whatever is missing or stale."""
    user = get_eval_user()
    rendered = {d.key: render(d.markdown, d.format) for d in dataset.documents}
    hashes = {key: hashlib.sha256(data).hexdigest() for key, data in rendered.items()}

    # Older versions of edited corpus files would otherwise compete in the search.
    stale = Document.objects.filter(user=user).exclude(sha256__in=hashes.values())
    for document in stale:
        logger.info("Removing outdated eval document %s", document.original_name)
        document.file.delete(save=False)
        document.delete()

    documents = {}
    for corpus_document in dataset.documents:
        key = corpus_document.key
        document = Document.objects.filter(user=user, sha256=hashes[key]).first()
        if document is None:
            data = rendered[key]
            document = Document(
                user=user,
                original_name=corpus_document.name,
                sha256=hashes[key],
                size_bytes=len(data),
            )
            document.file.save(corpus_document.name, ContentFile(data), save=False)
            document.save()
        if reingest or document.status != Document.Status.READY:
            logger.info("Ingesting %s", corpus_document.name)
            ingest(document)  # errors propagate: a broken corpus must stop the eval
            document.refresh_from_db()
        documents[key] = document
    return documents
