import pytest
from asgiref.sync import sync_to_async
from django.db import connections

from chat.models import Conversation


@pytest.fixture(autouse=True)
def fake_query_embedder(monkeypatch):
    """Retrieval runs on every chat reply; make sure it never calls the real Voyage API."""
    from tests.documents.fakes import FakeEmbedder

    embedder = FakeEmbedder()
    monkeypatch.setattr("documents.retrieval.get_embedder", lambda: embedder)
    return embedder


@pytest.fixture(autouse=True)
def no_langfuse(settings):
    """Real keys may be in .env: tests must never send traces to Langfuse."""
    from chat.tracing import get_langfuse

    settings.LANGFUSE_PUBLIC_KEY = settings.LANGFUSE_SECRET_KEY = ""
    get_langfuse.cache_clear()
    yield
    get_langfuse.cache_clear()


@pytest.fixture(autouse=True)
def no_fallback_model(settings):
    """Error tests script one call per attempt; fallback tests enable it explicitly."""
    settings.LLM_FALLBACK_MODEL = ""


@pytest.fixture
def user(django_user_model):
    return django_user_model.objects.create_user(username="alice", password="pw-alice-123")


@pytest.fixture
def conversation(user):
    return Conversation.objects.create(user=user, title="Test")


@pytest.fixture
async def close_async_db_connections():
    """Close the DB connection opened by async ORM calls in asgiref's worker thread.

    Otherwise it outlives the test and blocks dropping the test database.
    """
    yield
    await sync_to_async(connections.close_all)()
