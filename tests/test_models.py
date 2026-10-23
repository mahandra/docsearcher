from decimal import Decimal

import pytest

from chat.models import Message, conversation_cost

pytestmark = pytest.mark.django_db


def test_conversation_cost_sums_message_costs(conversation):
    Message.objects.create(conversation=conversation, role=Message.Role.USER, content="hi")
    for cost in ("0.001200", "0.000300"):
        Message.objects.create(
            conversation=conversation,
            role=Message.Role.ASSISTANT,
            content="hello",
            cost_usd=Decimal(cost),
        )

    assert conversation_cost(conversation) == Decimal("0.0015")


def test_conversation_cost_is_zero_without_messages(conversation):
    assert conversation_cost(conversation) == Decimal("0")
