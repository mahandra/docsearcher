"""Retrieval-augmented replies: sources in the prompt, citations in the answer."""

from decimal import Decimal

import pytest
from asgiref.sync import sync_to_async

from chat.llm import (
    ChatService,
    Cited,
    Completed,
    Retrieval,
    Source,
    SourcesFound,
    build_question_turn,
)
from chat.models import Message, MessageSource, conversation_cost
from documents.models import Chunk, Document
from tests.documents.fakes import fake_vector
from tests.fakes import FakeClient, Script, cite

pytestmark = [
    pytest.mark.django_db(transaction=True),
    pytest.mark.usefixtures("close_async_db_connections"),
]


def make_source(position=0, text="First paragraph.\n\nSecond paragraph.", **kwargs):
    defaults = {
        "chunk_id": 10 + position,
        "document_id": 1,
        "document_name": "contract.pdf",
        "chunk_index": position,
        "location": "p. 3",
        "score": 0.03,
    }
    return Source(position=position, text=text, **{**defaults, **kwargs})


class FakeRetriever:
    def __init__(self, retrieval: Retrieval) -> None:
        self.retrieval = retrieval
        self.calls: list[tuple[int, str]] = []

    async def __call__(self, user_id: int, query: str) -> Retrieval:
        self.calls.append((user_id, query))
        return self.retrieval


def make_service(*scripts, retriever=None):
    client = FakeClient(*scripts)
    service = ChatService(
        client,
        model="claude-opus-5-5",
        effort="low",
        retry_base_delay=0,
        **({"retriever": retriever} if retriever else {}),
    )
    return service, client


async def collect(service, conversation):
    return [event async for event in service.stream_reply(conversation)]


# --- Prompt shape ---------------------------------------------------------------------


def test_question_turn_puts_search_results_before_the_question():
    content = build_question_turn("What is the notice period?", [make_source(0), make_source(1)])

    assert [b["type"] for b in content] == ["search_result", "search_result", "text"]
    assert content[-1]["text"] == "What is the notice period?"
    first = content[0]
    assert first["title"] == "contract.pdf (p. 3)"
    assert first["source"] == "document:1#chunk-0"
    assert first["citations"] == {"enabled": True}
    # One text block per paragraph: Claude cites whole blocks, so this is the granularity.
    assert first["content"] == [
        {"type": "text", "text": "First paragraph."},
        {"type": "text", "text": "Second paragraph."},
    ]


def test_title_without_location():
    block = build_question_turn("q", [make_source(location="")])[0]
    assert block["title"] == "contract.pdf"


async def test_only_the_latest_question_carries_search_results(conversation):
    for role, content in [("user", "Old question"), ("assistant", "Old answer"), ("user", "New")]:
        await Message.objects.acreate(conversation=conversation, role=role, content=content)
    retriever = FakeRetriever(Retrieval([make_source()]))
    service, client = make_service(Script(["ok"]), retriever=retriever)

    await collect(service, conversation)

    assert retriever.calls == [(conversation.user_id, "New")]
    messages = client.messages.calls[0]["messages"]
    assert messages[0] == {"role": "user", "content": "Old question"}
    assert messages[1] == {"role": "assistant", "content": "Old answer"}
    assert messages[2]["content"][0]["type"] == "search_result"


# --- Citations ------------------------------------------------------------------------


async def test_citations_are_streamed_and_saved_with_offsets(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Notice?")
    sources = [make_source(0), make_source(1), make_source(2)]
    script = Script(
        [
            "Two months",
            cite(1, "Notice is two months."),
            " in writing",
            cite(0, "Notice must be written."),
            cite(0, "Signed by both parties."),  # same source, same spot: merged
            ".",
        ]
    )
    service, _ = make_service(script, retriever=FakeRetriever(Retrieval(sources)))

    events = await collect(service, conversation)

    assert events[0] == SourcesFound(sources)
    assert [e.source for e in events if isinstance(e, Cited)] == [1, 0]
    saved = events[-1].message
    await saved.arefresh_from_db()
    assert saved.content == "Two months in writing."
    assert saved.citations == [
        {"offset": 10, "source": 1, "cited_text": "Notice is two months."},
        {
            "offset": 21,
            "source": 0,
            "cited_text": "Notice must be written.\n\nSigned by both parties.",
        },
    ]
    assert saved.content[: saved.citations[0]["offset"]] == "Two months"


async def test_sources_are_saved_with_cited_flags(conversation, user):
    await Message.objects.acreate(conversation=conversation, role="user", content="Q")
    document = await Document.objects.acreate(
        user=user, original_name="contract.pdf", sha256="x", size_bytes=1, status="ready"
    )
    chunk = await Chunk.objects.acreate(
        document=document, index=0, text="t", token_count=1, embedding=fake_vector("t")
    )
    sources = [
        make_source(0, chunk_id=chunk.pk, document_id=document.pk, vector_rank=1),
        make_source(1, chunk_id=chunk.pk, document_id=document.pk, keyword_rank=2),
    ]
    service, _ = make_service(Script(["A", cite(1)]), retriever=FakeRetriever(Retrieval(sources)))

    events = await collect(service, conversation)

    rows = [s async for s in MessageSource.objects.filter(message=events[-1].message)]
    assert [(s.position, s.cited) for s in rows] == [(0, False), (1, True)]
    assert rows[0].vector_rank == 1 and rows[1].keyword_rank == 2
    assert rows[1].number == 2
    assert (rows[1].document_name, rows[1].location) == ("contract.pdf", "p. 3")

    # Deleting the document keeps the citation readable.
    await document.adelete()
    row = await MessageSource.objects.aget(pk=rows[1].pk)
    assert row.document_id is None and row.chunk_id is None
    assert row.document_name == "contract.pdf"


async def test_retrieval_cost_is_recorded_and_counted_in_conversation_cost(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Q")
    retrieval = Retrieval([], embedding_model="voyage-4", embedding_tokens=1_000)
    service, _ = make_service(Script(["A"]), retriever=FakeRetriever(retrieval))

    events = await collect(service, conversation)

    saved = events[-1].message
    assert saved.retrieval_tokens == 1_000
    assert saved.retrieval_cost_usd == Decimal("0.000060")  # 1000 * $0.06 / 1M
    total = await sync_to_async(conversation_cost)(conversation)
    assert total == saved.cost_usd + saved.retrieval_cost_usd


# --- Real retrieval end to end (fake embedder, real Postgres search) -------------------


async def test_default_retriever_searches_the_users_documents(conversation, user):
    question = "What is the notice period?"
    document = await Document.objects.acreate(
        user=user, original_name="contract.pdf", sha256="y", size_bytes=1, status="ready"
    )
    await Chunk.objects.acreate(
        document=document,
        index=0,
        text="The notice period is two months.",
        token_count=8,
        page_start=3,
        page_end=3,
        embedding=fake_vector(question),  # what the fake embedder returns for the question
    )
    await Message.objects.acreate(conversation=conversation, role="user", content=question)
    service, client = make_service(Script(["Two months.", cite(0)]))

    events = await collect(service, conversation)

    found = events[0].sources
    assert [(s.document_name, s.location) for s in found] == [("contract.pdf", "p. 3")]
    assert found[0].vector_rank == 1 and found[0].keyword_rank == 1
    block = client.messages.calls[0]["messages"][-1]["content"][0]
    assert block["content"] == [{"type": "text", "text": "The notice period is two months."}]
    assert isinstance(events[-1], Completed)
    assert events[-1].message.retrieval_tokens > 0


async def test_sources_of_chunks_deleted_during_generation_are_saved_unlinked(conversation):
    await Message.objects.acreate(conversation=conversation, role="user", content="Q")
    # Retrieved, then the document was deleted / re-processed before the answer finished.
    gone = make_source(0, chunk_id=999_999, document_id=999_999)
    service, _ = make_service(Script(["A", cite(0)]), retriever=FakeRetriever(Retrieval([gone])))

    events = await collect(service, conversation)

    assert isinstance(events[-1], Completed)
    row = await MessageSource.objects.aget(message=events[-1].message)
    assert (row.chunk_id, row.document_id, row.cited) == (None, None, True)
    assert row.document_name == "contract.pdf"
