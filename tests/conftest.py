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
