import json
import textwrap
from types import SimpleNamespace

import pytest
from django.core.management import call_command

from chat.models import Conversation
from evals.answer_eval import summarize
from evals.judge import Judgement
from tests.documents.fakes import FakeEmbedder
from tests.evals.test_judge import verdict
from tests.fakes import FakeClient, Script, cite

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.usefixtures("close_async_db_connections"),
]

DATASET = """
[[documents]]
key = "policy"
source = "policy.md"
name = "Policy.docx"
format = "docx"

[[cases]]
id = "notice"
category = "paraphrase"
question = "What is the notice period?"
evidence = [{ doc = "policy", text = "The notice period is two months" }]
facts = ["Two months."]

[[cases]]
id = "parking"
category = "unanswerable"
answerable = false
question = "Is there a parking lot?"

[[cases]]
id = "writing"
category = "follow_up"
history = ["What is the notice period?"]
question = "And in what form?"
evidence = [{ doc = "policy", text = "in writing" }]
facts = ["In writing."]
"""


class FakeJudge:
    model = "claude-sonnet-5-5"
    effort = "medium"

    def __init__(self):
        self.calls = []

    async def grade(self, case, passages, answer):
        self.calls.append((case.id, passages, answer))
        declined = not case.answerable
        return Judgement(verdict(declined=declined), self.model, 1000, 100, None)


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "corpus").mkdir()
    # One small chunk: it is always search result 0.
    (tmp_path / "corpus" / "policy.md").write_text(
        "# Policy\n\nThe notice period is two months, given in writing."
    )
    path = tmp_path / "dataset.toml"
    path.write_text(textwrap.dedent(DATASET))
    return path


@pytest.fixture
def fakes(monkeypatch, fake_query_embedder):
    command = "evals.management.commands.eval_answers"
    monkeypatch.setattr("documents.tasks.get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(f"{command}.get_embedder", lambda: fake_query_embedder)
    client = FakeClient(
        Script(["Two months.", cite(0)]),  # notice
        Script(["The documents do not mention parking."]),  # parking
        Script(["Two months.", cite(0)]),  # writing: history turn
        Script(["In writing.", cite(0)]),  # writing: the follow-up
    )
    monkeypatch.setattr(f"{command}.get_chat_service", lambda: SimpleNamespace(client=client))
    judge = FakeJudge()
    monkeypatch.setattr(f"{command}.get_judge", lambda model: judge)
    return SimpleNamespace(client=client, judge=judge, embedder=fake_query_embedder)


def run(dataset, tmp_path, *args):
    output = tmp_path / "answers.json"
    call_command("eval_answers", "--dataset", dataset, "--output", output, *args)
    return json.loads(output.read_text()) if output.exists() else None


def test_without_yes_nothing_is_called(dataset, tmp_path, fakes):
    assert run(dataset, tmp_path) is None
    assert fakes.client.messages.calls == []
    assert fakes.judge.calls == []


def test_answers_are_generated_checked_and_judged(dataset, tmp_path, fakes):
    report = run(dataset, tmp_path, "--yes", "--concurrency", "1")

    cases = {c["id"]: c for c in report["cases"]}
    assert cases["notice"]["answer"] == "Two months."
    assert cases["notice"]["cited"] == [0]
    assert cases["notice"]["cited_evidence"] is True
    assert cases["notice"]["sources"] == ["Policy.docx (Policy)"]
    assert cases["parking"]["cited_evidence"] is None
    assert cases["parking"]["verdict"]["declined"] is True
    assert report["summary"]["overall"]["correct_decline"] == 1.0
    assert report["summary"]["overall"]["false_decline"] == 0.0
    # All questions, history included, were embedded in one batch (repeats once).
    assert fakes.embedder.queries == [
        "What is the notice period?",
        "Is there a parking lot?",
        "And in what form?",
    ]

    # The follow-up was asked after the model's own first answer, like in the app.
    follow_up = fakes.client.messages.calls[3]["messages"]
    assert follow_up[1] == {"role": "assistant", "content": "Two months."}
    assert follow_up[2]["content"][-1]["text"] == "And in what form?"
    judged = {case_id: answer for case_id, _, answer in fakes.judge.calls}
    assert judged["writing"] == "In writing."
    assert Conversation.objects.filter(user__username="docqa-eval").count() == 3


def test_case_selection(dataset, tmp_path, fakes):
    report = run(dataset, tmp_path, "--yes", "--case", "parking")
    assert [c["id"] for c in report["cases"]] == ["parking"]


def test_summary_scores_partial_answers_as_half():
    def result(correctness):
        return SimpleNamespace(
            category="paraphrase",
            answerable=True,
            cited_evidence=True,
            verdict=verdict(correctness=correctness).model_dump(),
            answer_cost_usd=0.01,
            judge_cost_usd=0.02,
            id=correctness,
        )

    summary = summarize([result("correct"), result("partial")])

    assert summary["overall"]["correctness"] == 0.75
    assert summary["failures"] == ["partial"]
    assert summary["cost_usd"]["judge"] == pytest.approx(0.04)
