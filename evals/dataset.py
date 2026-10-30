"""The evaluation set: corpus documents and questions with their expected evidence."""

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .corpus import markdown_text

EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "dataset.toml"
CORPUS_DIR = EVALS_DIR / "corpus"

Format = Literal["pdf", "docx"]
CATEGORIES = (
    "paraphrase",
    "exact",
    "cross_lingual",
    "multi_hop",
    "unanswerable",
    "follow_up",
    "injection",
)

_WHITESPACE = re.compile(r"\s+")


class DatasetError(Exception):
    pass


@dataclass(frozen=True)
class CorpusDocument:
    key: str
    source: Path
    name: str
    format: Format

    @property
    def markdown(self) -> str:
        return self.source.read_text(encoding="utf-8")


@dataclass(frozen=True)
class Evidence:
    doc: str  # CorpusDocument.key
    text: str


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    question: str
    answerable: bool = True
    evidence: tuple[Evidence, ...] = ()
    facts: tuple[str, ...] = ()
    history: tuple[str, ...] = ()
    # Prompt injection: strings that appear in the answer only if the model obeyed
    # instructions planted in a document (a competitor's name, a phishing URL, ...).
    canaries: tuple[str, ...] = ()


@dataclass(frozen=True)
class Dataset:
    documents: list[CorpusDocument]
    cases: list[Case] = field(default_factory=list)

    def document(self, key: str) -> CorpusDocument:
        return next(d for d in self.documents if d.key == key)


def normalize(text: str) -> str:
    """Compare text the way it survives PDF/DOCX extraction: case and line breaks vary."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


def contains(chunk_text: str, evidence: Evidence) -> bool:
    return normalize(evidence.text) in normalize(chunk_text)


def load_dataset(path: Path = DATASET_PATH) -> Dataset:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    corpus_dir = path.parent / "corpus"
    documents = [
        CorpusDocument(
            key=d["key"], source=corpus_dir / d["source"], name=d["name"], format=d["format"]
        )
        for d in raw.get("documents", [])
    ]
    cases = [
        Case(
            id=c["id"],
            category=c["category"],
            question=c["question"],
            answerable=c.get("answerable", True),
            evidence=tuple(Evidence(e["doc"], e["text"]) for e in c.get("evidence", [])),
            facts=tuple(c.get("facts", [])),
            history=tuple(c.get("history", [])),
            canaries=tuple(c.get("canaries", [])),
        )
        for c in raw.get("cases", [])
    ]
    dataset = Dataset(documents=documents, cases=cases)
    validate(dataset)
    return dataset


def validate(dataset: Dataset) -> None:
    """Fail loudly on a broken dataset: a typo in a quote would silently lower recall."""
    errors = []
    keys = [d.key for d in dataset.documents]
    if len(set(keys)) != len(keys):
        errors.append("duplicate document keys")
    texts = {}
    for document in dataset.documents:
        if document.format not in ("pdf", "docx"):
            errors.append(f"{document.key}: unknown format {document.format!r}")
        if not document.source.is_file():
            errors.append(f"{document.key}: missing source {document.source.name}")
            continue
        texts[document.key] = normalize(markdown_text(document.markdown))

    ids = [c.id for c in dataset.cases]
    for duplicate in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"duplicate case id {duplicate!r}")
    for case in dataset.cases:
        prefix = f"case {case.id}"
        if case.category not in CATEGORIES:
            errors.append(f"{prefix}: unknown category {case.category!r}")
        if case.answerable and not (case.evidence and case.facts):
            errors.append(f"{prefix}: an answerable case needs evidence and facts")
        if not case.answerable and (case.evidence or case.category != "unanswerable"):
            errors.append(f"{prefix}: an unanswerable case has no evidence, category unanswerable")
        if case.category == "follow_up" and not case.history:
            errors.append(f"{prefix}: a follow-up needs history")
        if case.category == "injection" and not case.canaries:
            errors.append(f"{prefix}: an injection case needs canaries")
        for evidence in case.evidence:
            if evidence.doc not in texts:
                errors.append(f"{prefix}: unknown document {evidence.doc!r}")
            elif normalize(evidence.text) not in texts[evidence.doc]:
                errors.append(f"{prefix}: quote not found in {evidence.doc}: {evidence.text!r}")
    if errors:
        raise DatasetError("Invalid dataset:\n- " + "\n- ".join(errors))
