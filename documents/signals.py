import logging

from django.db import transaction
from django.db.models.signals import post_delete
from django.dispatch import receiver

from .models import Document

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=Document)
def delete_document_file(sender, instance: Document, **kwargs) -> None:
    """Django never deletes files of FileFields; do it once the deletion is committed."""
    if not instance.file:
        return
    storage, name = instance.file.storage, instance.file.name

    def delete_file() -> None:
        try:
            storage.delete(name)
        except OSError:
            # The row is already gone; an orphaned file must not break the request.
            logger.exception("Could not delete file %s", name)

    transaction.on_commit(delete_file)
