import pytest

from chat.models import Conversation


@pytest.fixture
def user(django_user_model):
    return django_user_model.objects.create_user(username="alice", password="pw-alice-123")


@pytest.fixture
def conversation(user):
    return Conversation.objects.create(user=user, title="Test")
