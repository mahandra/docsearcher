import pytest

from documents.chunking import CHARS_PER_TOKEN, chunk_sections, estimate_tokens
from documents.parsing import Section


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 35) == 10


def test_short_text_is_one_chunk():
    chunks = chunk_sections([Section("Hello world.", page=1)], chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)


def test_chunks_respect_size_limit():
    paragraphs = "\n\n".join(f"Paragraph {i}. " + words(30) for i in range(40))

    chunks = chunk_sections([Section(paragraphs)], chunk_size=100, overlap=20)

    assert len(chunks) > 5
    assert all(c.token_count <= 100 for c in chunks)


def test_consecutive_chunks_overlap_even_with_long_paragraphs():
    # Each paragraph (~40 tokens) is longer than the overlap budget (25 tokens), so the
    # overlap must come from the trailing sentences of the previous paragraph.
    paragraphs = [" ".join(f"P{i} sentence {j} has words." for j in range(6)) for i in range(20)]
    chunks = chunk_sections([Section("\n\n".join(paragraphs))], chunk_size=100, overlap=25)

    assert len(chunks) > 3
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        repeated = nxt.text.split("\n\n")[0]
        assert prev.text.endswith(repeated)
        assert 0 < estimate_tokens(repeated) <= 25
        assert repeated.endswith(".")  # whole sentences, not cut words


def test_splits_on_paragraphs_before_sentences():
    a, b = "A" * 200, "B" * 200  # ~57 tokens each
    chunks = chunk_sections([Section(f"{a}\n\n{b}")], chunk_size=80, overlap=0)

    assert [c.text for c in chunks] == [a, b]


def test_long_paragraph_is_split_by_sentences():
    sentences = [f"Sentence number {i} " + words(10) + "." for i in range(30)]
    chunks = chunk_sections([Section(" ".join(sentences))], chunk_size=60, overlap=0)

    assert len(chunks) > 1
    assert all(c.text.endswith(".") for c in chunks)  # no sentence cut in half


def test_unbreakable_text_is_hard_cut():
    blob = "x" * int(CHARS_PER_TOKEN * 250)
    chunks = chunk_sections([Section(blob)], chunk_size=100, overlap=0)

    assert all(c.token_count <= 100 for c in chunks)
    assert "".join(c.text.replace("\n\n", "") for c in chunks) == blob


def test_chunks_may_span_pdf_pages_and_record_range():
    sections = [Section("End of page one.", page=1), Section("Start of page two.", page=2)]

    chunks = chunk_sections(sections, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_docx_heading_always_starts_new_chunk():
    sections = [
        Section("Intro\n\nShort intro.", heading="Intro"),
        Section("Методы\n\nКороткий раздел.", heading="Методы"),
    ]

    chunks = chunk_sections(sections, chunk_size=500, overlap=50)

    assert [c.section for c in chunks] == ["Intro", "Методы"]
    assert "Intro" not in chunks[1].text  # no overlap across topics


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_sections([Section("x")], chunk_size=100, overlap=100)
