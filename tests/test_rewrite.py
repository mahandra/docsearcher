"""Follow-up questions are rewritten into standalone search queries (structured outputs)."""

from decimal import Decimal
from types import SimpleNamespace

from chat.rewrite import (
    MAX_ANSWER_CHARS,
    MAX_HISTORY_MESSAGES,
    QueryRewriter,
    SearchQuery,
    build_rewrite_input,
)

HISTORY = [
    {"role": "user", "content": "What is the probation period?"},
    {"role": "assistant", "content": "Three months."},
]


class FakeParse:
    """messages.parse() returning a scripted structured output (or raising)."""

    def __init__(self, result=None, *, error=None, stop_reason="end_turn"):
        self.result, self.error, self.stop_reason = result, error, stop_reason
        self.calls = []

    async def parse(self, **params):
        self.calls.append(params)
        if self.error:
            raise self.error
        return SimpleNamespace(
            parsed_output=self.result,
            stop_reason=self.stop_reason,
            model="claude-haiku-5-5",
            usage=SimpleNamespace(input_tokens=1000, output_tokens=40),
        )


def rewriter(messages):
    return QueryRewriter(SimpleNamespace(messages=messages), model="claude-haiku-5-5", prompt="P")


def test_input_has_the_conversation_and_the_latest_message():
    text = build_rewrite_input(HISTORY, "And for managers?")
    assert "<user>\nWhat is the probation period?\n</user>" in text
    assert "<assistant>\nThree months.\n</assistant>" in text
    assert text.endswith("<latest_message>\nAnd for managers?\n</latest_message>")


def test_input_keeps_recent_turns_and_cuts_long_answers():
    old = [{"role": "user", "content": f"old question {i}"} for i in range(10)]
    long_answer = {"role": "assistant", "content": "x" * (MAX_ANSWER_CHARS + 100)}
    text = build_rewrite_input([*old, long_answer], "Why?")
    assert "old question 0" not in text
    assert f"old question {10 - MAX_HISTORY_MESSAGES + 1}" in text
    assert "x" * MAX_ANSWER_CHARS + " […]" in text
    assert "x" * (MAX_ANSWER_CHARS + 1) not in text


async def test_rewrites_with_a_schema_constrained_reply():
    messages = FakeParse(SearchQuery(query="What is the probation period for managers?"))

    rewrite = await rewriter(messages).rewrite(HISTORY, "And for managers?")

    assert rewrite.query == "What is the probation period for managers?"
    assert rewrite.rewritten
    assert (rewrite.input_tokens, rewrite.output_tokens) == (1000, 40)
    assert rewrite.cost_usd == Decimal("0.000120")  # Haiku: 1000*$0.10 + 40*$0.50, per 1M
    params = messages.calls[0]
    assert params["output_format"] is SearchQuery
    assert params["model"] == "claude-haiku-5-5"
    assert params["system"] == "P"
    assert "temperature" not in params


async def test_standalone_question_comes_back_unchanged():
    messages = FakeParse(SearchQuery(query="How long is the notice period?"))
    rewrite = await rewriter(messages).rewrite(HISTORY, "How long is the notice period?")
    assert rewrite.query == "How long is the notice period?"
    assert not rewrite.rewritten


async def test_no_call_without_history():
    messages = FakeParse(SearchQuery(query="unused"))
    rewrite = await rewriter(messages).rewrite([], "What is the notice period?")
    assert rewrite.query == "What is the notice period?"
    assert messages.calls == []


async def test_api_error_falls_back_to_the_question():
    messages = FakeParse(error=RuntimeError("overloaded"))
    rewrite = await rewriter(messages).rewrite(HISTORY, "And for managers?")
    assert rewrite.query == "And for managers?"
    assert not rewrite.rewritten
    assert rewrite.error == "overloaded"


async def test_no_parsed_output_falls_back_to_the_question():
    messages = FakeParse(None, stop_reason="refusal")
    rewrite = await rewriter(messages).rewrite(HISTORY, "And for managers?")
    assert rewrite.query == "And for managers?"
    assert "refusal" in rewrite.error
    assert rewrite.cost_usd is not None  # the call is billed all the same
