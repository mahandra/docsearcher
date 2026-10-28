from types import SimpleNamespace

import pytest

from evals.dataset import Case, Evidence
from evals.judge import Judge, Verdict, build_judge_input

ANSWERABLE = Case(
    id="notice",
    category="follow_up",
    question="And for managers?",
    evidence=(Evidence("doc", "six months"),),
    facts=("Six months for managers.",),
    history=("How long is the probation period?",),
)
UNANSWERABLE = Case(id="parking", category="unanswerable", question="Parking?", answerable=False)


def verdict(**overrides) -> Verdict:
    fields = {
        "reasoning": "States the fact.",
        "declined": False,
        "correctness": "correct",
        "missing_facts": [],
        "faithful": True,
        "unsupported_claims": [],
    }
    return Verdict(**{**fields, **overrides})


class FakeParseMessages:
    def __init__(self, parsed: Verdict | None) -> None:
        self.parsed = parsed
        self.calls: list[dict] = []

    async def parse(self, **params):
        self.calls.append(params)
        return SimpleNamespace(
            parsed_output=self.parsed,
            stop_reason="end_turn" if self.parsed else "max_tokens",
            model=params["model"],
            usage=SimpleNamespace(input_tokens=1_000, output_tokens=200),
        )


def fake_judge(parsed):
    messages = FakeParseMessages(parsed)
    return Judge(SimpleNamespace(messages=messages)), messages


def test_judge_input_has_history_reference_and_numbered_passages():
    text = build_judge_input(ANSWERABLE, ["First passage.", "Second passage."], "Six months.")

    assert "<earlier_questions>\n- How long is the probation period?" in text
    assert "- Six months for managers." in text
    assert '<passage index="1">\nFirst passage.' in text
    assert '<passage index="2">\nSecond passage.' in text
    assert text.endswith("<answer>\nSix months.\n</answer>")


def test_judge_input_for_unanswerable_question_without_passages():
    text = build_judge_input(UNANSWERABLE, [], "The documents do not say.")

    assert "The documents do not answer this question" in text
    assert "(none)" in text


async def test_grade_uses_structured_output_and_records_cost():
    judge, messages = fake_judge(verdict(correctness="partial"))

    judgement = await judge.grade(ANSWERABLE, ["p"], "a")

    assert judgement.verdict.correctness == "partial"
    params = messages.calls[0]
    assert params["model"] == "claude-sonnet-5-5"
    assert params["output_format"] is Verdict
    assert params["output_config"] == {"effort": "medium"}
    assert "grade answers" in params["system"]
    assert judgement.cost_usd > 0


async def test_grade_without_a_verdict_fails_loudly():
    judge, _ = fake_judge(None)
    with pytest.raises(RuntimeError, match="max_tokens"):
        await judge.grade(ANSWERABLE, [], "a")
