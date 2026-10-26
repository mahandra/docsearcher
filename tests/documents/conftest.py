import hashlib

import pytest
from django.core.files.base import ContentFile

from documents.models import Document


@pytest.fixture(autouse=True)
def media_root(tmp_path, settings):
    """Keep uploaded files out of the project during tests."""
    settings.MEDIA_ROOT = tmp_path / "media"
    return settings.MEDIA_ROOT


@pytest.fixture
def make_document(user):
    def make(content: bytes = b"%PDF-1.4 test", name: str = "file.pdf", owner=None, **kwargs):
        document = Document(
            user=owner or user,
            original_name=name,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            **kwargs,
        )
        document.file.save(name, ContentFile(content), save=False)
        document.save()
        return document

    return make
