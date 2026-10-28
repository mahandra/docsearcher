import hashlib
import textwrap

import pytest

from documents.parsing import parse_document
from evals.corpus import markdown_text, parse_markdown, render
from evals.dataset import DatasetError, Evidence, contains, load_dataset


def write_dataset(tmp_path, cases: str, source="# Title\n\nNotice is two months."):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "doc.md").write_text(source)
    path = tmp_path / "dataset.toml"
    documents = """
        [[documents]]
        key = "doc"
        source = "doc.md"
        name = "Doc.pdf"
        format = "pdf"
    """
    path.write_text(textwrap.dedent(documents) + textwrap.dedent(cases))
    return path


def test_the_real_dataset_is_valid():
    dataset = load_dataset()  # validates every quote against its document
    assert len(dataset.cases) >= 40
    assert any(not c.answerable for c in dataset.cases)


def test_quotes_are_matched_ignoring_case_and_line_breaks():
    assert contains("The notice\nperiod is TWO months.", Evidence("doc", "period is two months"))
    assert not contains("The notice period", Evidence("doc", "notice period is"))


def test_a_quote_missing_from_its_document_is_rejected(tmp_path):
    path = write_dataset(
        tmp_path,
        """
        [[cases]]
        id = "typo"
        category = "paraphrase"
        question = "Notice?"
        evidence = [{ doc = "doc", text = "Notice is three months." }]
        facts = ["Three months."]
        """,
    )
    with pytest.raises(DatasetError, match="quote not found in doc"):
        load_dataset(path)


@pytest.mark.parametrize(
    "case, error",
    [
        ('category = "paraphrase"\nquestion = "Q"', "needs evidence and facts"),
        ('category = "vibes"\nanswerable = false\nquestion = "Q"', "unknown category"),
        (
            'category = "unanswerable"\nanswerable = false\nquestion = "Q"\n'
            'evidence = [{ doc = "doc", text = "Notice" }]',
            "has no evidence",
        ),
        (
            'category = "follow_up"\nquestion = "Q"\nfacts = ["F"]\n'
            'evidence = [{ doc = "nope", text = "Notice" }]',
            "follow-up needs history",
        ),
    ],
)
def test_inconsistent_cases_are_rejected(tmp_path, case, error):
    path = write_dataset(tmp_path, f'[[cases]]\nid = "c"\n{case}\n')
    with pytest.raises(DatasetError, match=error):
        load_dataset(path)


def test_markdown_subset():
    blocks = parse_markdown("# Title\n\n## 1. Part\n\nLine one\nline two.\n\n- a\n- b")
    assert [(b.kind, b.text) for b in blocks] == [
        ("h1", "Title"),
        ("h2", "1. Part"),
        ("p", "Line one line two."),
        ("p", "- a"),
        ("p", "- b"),
    ]


@pytest.mark.parametrize("fmt", ["pdf", "docx"])
def test_rendering_is_deterministic_and_parses_back(tmp_path, fmt):
    markdown = "# Policy\n\n## Passwords\n\n" + "A password has at least 12 characters. " * 60
    data = render(markdown, fmt)
    assert hashlib.sha256(data).digest() == hashlib.sha256(render(markdown, fmt)).digest()

    path = tmp_path / f"doc.{fmt}"
    path.write_bytes(data)
    parsed = parse_document(path, fmt)
    text = " ".join(s.text for s in parsed.sections)
    assert contains(text, Evidence("doc", markdown_text(markdown)[-200:]))
