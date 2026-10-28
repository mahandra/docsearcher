import json
import textwrap

import pytest
from django.core.management import call_command

from documents.models import Document
from evals.ingest import EVAL_USERNAME
from tests.documents.fakes import FakeEmbedder

pytestmark = pytest.mark.django_db

SOURCE = """\
# Policy

## Payment

Invoices are payable within thirty days. Late payments carry interest of 1.5% per month.

## Termination

The notice period is two months. Notice must be given in writing to the other party.

## Liability

Liability is capped at the fees of the last twelve months, except in cases of fraud.
"""

CASES = """
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
id = "nothing"
category = "unanswerable"
answerable = false
question = "Is there a parking lot?"
"""


@pytest.fixture
def embedder(monkeypatch):
    embedder = FakeEmbedder()
    monkeypatch.setattr("documents.tasks.get_embedder", lambda: embedder)
    return embedder


@pytest.fixture
def dataset(tmp_path, settings):
    settings.CHUNK_SIZE_TOKENS = 30  # several chunks from a small document
    settings.CHUNK_OVERLAP_TOKENS = 0
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "policy.md").write_text(SOURCE)
    path = tmp_path / "dataset.toml"
    path.write_text(textwrap.dedent(CASES))
    return path


def run(dataset, tmp_path, *args):
    output = tmp_path / "report.json"
    call_command("eval_retrieval", "--dataset", dataset, "--output", output, *args)
    return json.loads(output.read_text())


def test_report_scores_each_mode(dataset, tmp_path, embedder):
    report = run(dataset, tmp_path)

    assert report["settings"]["chunks"] > 1
    notice = next(c for c in report["cases"] if c["id"] == "notice")
    # The fake embedder is random, but full-text search finds the exact words.
    assert notice["modes"]["keyword"]["relevant_ranks"][0] == 1
    assert notice["modes"]["keyword"]["hits"]["1"] is True
    assert notice["modes"]["hybrid"]["chunks"][0].startswith("policy#")
    summary = report["summary"]
    assert summary["modes"]["keyword"]["n"] == 1  # unanswerable cases are not scored
    assert summary["distances"]["unanswerable_top"]["n"] == 1
    assert report["embedding_tokens"] > 0


def test_all_questions_are_embedded_in_one_batch(dataset, tmp_path, embedder, fake_query_embedder):
    run(dataset, tmp_path)
    assert fake_query_embedder.queries == ["What is the notice period?", "Is there a parking lot?"]


def test_unchanged_corpus_is_not_embedded_again(dataset, tmp_path, embedder):
    run(dataset, tmp_path)
    calls = len(embedder.calls)

    run(dataset, tmp_path)
    assert len(embedder.calls) == calls

    run(dataset, tmp_path, "--reingest")
    assert len(embedder.calls) == calls + 1


def test_edited_corpus_replaces_the_old_version(dataset, tmp_path, embedder):
    run(dataset, tmp_path)
    source = tmp_path / "corpus" / "policy.md"
    source.write_text(source.read_text() + "\nA new paragraph.\n")

    run(dataset, tmp_path)

    documents = Document.objects.filter(user__username=EVAL_USERNAME)
    assert documents.count() == 1
    assert documents.get().status == Document.Status.READY
