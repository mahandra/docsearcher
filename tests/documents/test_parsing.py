import pytest

from documents.parsing import ParsingError, Section, detect_kind, normalize, parse_document
from tests.documents.factories import encrypt_pdf, make_docx, make_pdf


@pytest.fixture
def write(tmp_path):
    def write(data: bytes, name: str):
        path = tmp_path / name
        path.write_bytes(data)
        return path

    return write


# --- detection & normalisation -----------------------------------------------------


def test_detect_kind_uses_magic_bytes():
    assert detect_kind(make_pdf([["x"]])[:8]) == "pdf"
    assert detect_kind(make_docx([("p", "x")])[:8]) == "docx"
    assert detect_kind(b"MZ\x90\x00 an exe") is None
    assert detect_kind(b"") is None


def test_normalize_cleans_pdf_artifacts():
    raw = "infor-\nmation\x00  with   spaces\r\n\n\n\n\nnext  "
    assert normalize(raw) == "information with spaces\n\nnext"


# --- PDF ---------------------------------------------------------------------------


def test_pdf_keeps_page_numbers(write):
    path = write(make_pdf([["First page text"], ["Second page text"]]), "a.pdf")

    parsed = parse_document(path, "pdf")

    assert parsed.page_count == 2
    assert parsed.sections == [
        Section(text="First page text", page=1),
        Section(text="Second page text", page=2),
    ]


def test_pdf_skips_blank_pages_but_counts_them(write):
    path = write(make_pdf([["Text"], [], ["More"]]), "a.pdf")

    parsed = parse_document(path, "pdf")

    assert parsed.page_count == 3
    assert [s.page for s in parsed.sections] == [1, 3]


def test_pdf_without_text_is_reported_as_scanned(write):
    path = write(make_pdf([[], []]), "scan.pdf")

    with pytest.raises(ParsingError, match="scanned"):
        parse_document(path, "pdf")


def test_password_protected_pdf(write):
    path = write(encrypt_pdf(make_pdf([["secret"]]), "pw"), "locked.pdf")

    with pytest.raises(ParsingError, match="password"):
        parse_document(path, "pdf")


def test_pdf_with_empty_password_is_readable(write):
    path = write(encrypt_pdf(make_pdf([["open text"]]), ""), "copy-protected.pdf")

    assert parse_document(path, "pdf").sections[0].text == "open text"


def test_broken_pdf(write):
    path = write(b"%PDF-1.4 garbage", "broken.pdf")

    with pytest.raises(ParsingError, match="damaged"):
        parse_document(path, "pdf")


# --- DOCX --------------------------------------------------------------------------


def test_docx_groups_paragraphs_under_headings(write):
    data = make_docx(
        [
            ("p", "Preface without heading."),
            ("h1", "Введение"),
            ("p", "Текст на русском."),
            ("p", "Mixed English paragraph."),
            ("h2", "Results"),
            ("table", [["Metric", "Value"], ["Recall", "0.9"]]),
        ]
    )

    parsed = parse_document(write(data, "a.docx"), "docx")

    assert parsed.page_count is None
    assert parsed.sections == [
        Section(text="Preface without heading.", heading=""),
        Section(
            text="Введение\n\nТекст на русском.\n\nMixed English paragraph.", heading="Введение"
        ),
        Section(text="Results\n\nMetric | Value\nRecall | 0.9", heading="Results"),
    ]


def test_zip_that_is_not_docx(write, tmp_path):
    import zipfile

    path = tmp_path / "fake.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("hello.txt", "not a word document")

    with pytest.raises(ParsingError, match="not a valid DOCX"):
        parse_document(path, "docx")


def test_empty_docx(write):
    with pytest.raises(ParsingError, match="No extractable text"):
        parse_document(write(make_docx([]), "empty.docx"), "docx")
