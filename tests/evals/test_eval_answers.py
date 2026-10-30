import asyncio
import json
import textwrap
from types import SimpleNamespace

import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from chat.models import Conversation
from evals.answer_eval import AnswerResult, summarize
from evals.experiment import DATASET_NAME, scores
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
    monkeypatch.setattr(
        f"{command}.get_chat_service", lambda: SimpleNamespace(client=client, rewriter=None)
    )
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
    assert follow_up[1]["role"] == "assistant"
    assert follow_up[1]["content"][0]["text"] == "Two months."
    assert follow_up[2]["content"][-1]["text"] == "And in what form?"
    judged = {case_id: answer for case_id, _, answer in fakes.judge.calls}
    assert judged["writing"] == "In writing."
    # The judge sees passage titles too: the model may name the document.
    notice_passages = next(p for case_id, p, _ in fakes.judge.calls if case_id == "notice")
    assert [p.title for p in notice_passages] == ["Policy.docx (Policy)"]
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


# --- Langfuse experiment ----------------------------------------------------------


class FakeLangfuse:
    """Records dataset calls; runs an experiment the way the SDK calls our functions."""

    def __init__(self):
        self.items = {}
        self.datasets = []
        self.experiments = []
        self.item_scores = {}

    def create_dataset(self, *, name, description=None):
        self.datasets.append(name)

    def create_dataset_item(self, *, dataset_name, id, input, expected_output, metadata):
        self.items[id] = SimpleNamespace(
            id=id, input=input, expected_output=expected_output, metadata=metadata
        )

    def get_dataset(self, name):
        return SimpleNamespace(items=list(self.items.values()))

    def run_experiment(self, *, data, task, evaluators, run_evaluators, **kwargs):
        async def run_all():
            item_results = []
            for item in data:  # sequential, like --concurrency 1
                output = await task(item=item)
                evaluation = [
                    e
                    for evaluator in evaluators
                    for e in evaluator(
                        input=item.input,
                        output=output,
                        expected_output=item.expected_output,
                        metadata=item.metadata,
                    )
                ]
                self.item_scores[item.metadata["case_id"]] = {
                    e["name"]: e["value"] for e in evaluation
                }
                item_results.append(SimpleNamespace(item=item, output=output))
            return [e for r in run_evaluators for e in r(item_results=item_results)]

        self.run_scores = {e["name"]: e["value"] for e in asyncio.run(run_all())}
        self.experiments.append(kwargs)
        return SimpleNamespace(experiment_url="https://langfuse.test/experiment")


@pytest.fixture
def langfuse(monkeypatch):
    client = FakeLangfuse()
    monkeypatch.setattr("evals.management.commands.eval_answers.get_langfuse", lambda: client)
    return client


def test_run_is_recorded_as_a_langfuse_experiment(dataset, tmp_path, fakes, langfuse, capsys):
    report = run(
        dataset, tmp_path, "--yes", "--concurrency", "1", "--langfuse", "--run-name", "baseline"
    )

    assert langfuse.datasets == [DATASET_NAME]
    item = langfuse.items[f"{DATASET_NAME}-writing"]
    assert item.input == {
        "question": "And in what form?",
        "history": ["What is the notice period?"],
    }
    assert item.expected_output == {"answerable": True, "facts": ["In writing."]}
    assert item.metadata == {"case_id": "writing", "category": "follow_up"}

    experiment = langfuse.experiments[0]
    assert experiment["run_name"] == "baseline"
    assert experiment["max_concurrency"] == 1
    assert experiment["metadata"]["model"] == report["settings"]["model"]

    assert langfuse.item_scores["notice"]["cited_evidence"] == 1.0
    assert "cited_evidence" not in langfuse.item_scores["parking"]  # unanswerable
    assert langfuse.item_scores["parking"]["declined"] == 1.0
    assert langfuse.run_scores["avg_correctness"] == 1.0

    # The local report is written as usual.
    assert [c["id"] for c in report["cases"]] == ["notice", "parking", "writing"]
    assert "https://langfuse.test/experiment" in capsys.readouterr().out


def test_langfuse_flag_needs_keys(dataset, tmp_path, fakes):
    with pytest.raises(CommandError, match="LANGFUSE"):
        run(dataset, tmp_path, "--langfuse")


def test_scores_carry_the_judges_reasons():
    result = AnswerResult(
        id="x",
        category="paraphrase",
        answerable=True,
        question="q",
        answer="a",
        status="complete",
        sources=[],
        cited=[],
        cited_evidence=False,
        verdict={
            "reasoning": "Half right.",
            "declined": False,
            "correctness": "partial",
            "missing_facts": ["In writing."],
            "faithful": False,
            "unsupported_claims": ["Call HR."],
        },
        answer_cost_usd=0.0,
        judge_cost_usd=0.0,
        conversation_id=1,
    )

    by_name = {s["name"]: s for s in scores(result)}

    assert by_name["correctness"]["value"] == 0.5
    assert by_name["correctness"]["comment"] == "Half right.\nMissing: In writing."
    assert by_name["faithful"] == {
        "name": "faithful",
        "value": 0.0,
        "comment": "Unsupported: Call HR.",
    }
    assert by_name["cited_evidence"]["value"] == 0.0
