import numpy as np
import pytest
from django.contrib.postgres.search import SearchQuery
from django.db import IntegrityError

from documents.models import EMBEDDING_DIMENSIONS, Chunk

pytestmark = pytest.mark.django_db


def test_uploaded_file_gets_random_name_in_user_folder(make_document, user):
    document = make_document(name="../../Annual Report.PDF")

    assert document.file.name.startswith(f"documents/{user.pk}/")
    assert document.file.name.endswith(".pdf")
    assert "Annual" not in document.file.name


def test_same_file_twice_for_same_user_is_rejected(make_document):
    make_document(b"same bytes")
    with pytest.raises(IntegrityError):
        make_document(b"same bytes")


def test_chunk_embedding_round_trip(make_document):
    vector = np.linspace(-1, 1, EMBEDDING_DIMENSIONS)
    chunk = Chunk.objects.create(
        document=make_document(), index=0, text="hello", token_count=2, embedding=vector
    )

    chunk.refresh_from_db()
    assert np.allclose(chunk.embedding, vector, atol=1e-6)


def test_chunk_location():
    assert Chunk(page_start=3, page_end=3).location == "p. 3"
    assert Chunk(page_start=3, page_end=5).location == "pp. 3–5"
    assert Chunk(section="Introduction").location == "Introduction"


@pytest.mark.parametrize(
    ("query", "config"),
    [
        ("contract", "english"),  # stem of "contracts"
        ("terminated", "english"),  # "termination" and "terminated" share the stem
        ("договор", "russian"),  # stem of "договоры"
        ("расторжении", "russian"),  # another case form of "расторжение"
    ],
)
def test_search_vector_is_generated_with_english_and_russian_stemming(make_document, query, config):
    chunk = Chunk.objects.create(
        document=make_document(),
        index=0,
        text="All contracts allow termination. Договоры допускают расторжение.",
        token_count=10,
        embedding=np.zeros(EMBEDDING_DIMENSIONS) + 0.1,
    )

    matches = Chunk.objects.filter(search_vector=SearchQuery(query, config=config))
    assert list(matches) == [chunk]


def test_search_vector_follows_text_updates(make_document):
    chunk = Chunk.objects.create(
        document=make_document(),
        index=0,
        text="invoice",
        token_count=2,
        embedding=np.zeros(EMBEDDING_DIMENSIONS) + 0.1,
    )
    Chunk.objects.filter(pk=chunk.pk).update(text="receipt")

    query = SearchQuery("invoice", config="english")
    assert not Chunk.objects.filter(search_vector=query).exists()
