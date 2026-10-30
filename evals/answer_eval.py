"""Answer eval: run every question through the real ChatService and grade the reply.

Deterministic checks (free, exact):
- cited_evidence: at least one citation points at a chunk that contains an evidence quote.
LLM judge (see judge.py):
- correctness vs the reference facts, faithfulness vs the passages, declined or not.

Follow-ups replay their history first, so the model sees its own earlier answers, as in
the app. Each case runs in its own conversation of the eval user.
"""

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from functools import partial

from chat.llm import ChatService, Completed, Failed, Source, SourcesFound, retrieve_sources
from chat.models import Conversation, Message

from .dataset import Case, contains
from .judge import Judge, Passage, Verdict
from .metrics import average

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


@dataclass
class AnswerResult:
    id: str
    category: str
    answerable: bool
    question: str
    answer: str
    status: str
    sources: list[str]  # "document name (location)" in prompt order
    cited: list[int]  # source positions the answer cites
    cited_evidence: bool | None  # None for unanswerable cases
    verdict: dict
    answer_cost_usd: float
    judge_cost_usd: float
    conversation_id: int


@dataclass
class AnswerReport:
    settings: dict
    cases: list[AnswerResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


async def run_case(case: Case, service: ChatService, judge: Judge, user) -> AnswerResult:
    conversation = await Conversation.objects.acreate(user=user, title=f"eval: {case.id}")
    answer_cost = Decimal(0)
    for question in [*case.history, case.question]:
        await Message.objects.acreate(
            conversation=conversation, role=Message.Role.USER, content=question
        )
        sources: list[Source] = []
        message = None
        async for event in service.stream_reply(conversation):
            if isinstance(event, SourcesFound):
                sources = event.sources
            elif isinstance(event, Completed | Failed):
                message = event.message
        answer_cost += sum(
            cost or 0
            for cost in (message.cost_usd, message.retrieval_cost_usd, message.rewrite_cost_usd)
        )

    cited = sorted({c["source"] for c in message.citations})
    cited_evidence = None
    if case.answerable:
        cited_evidence = any(
            contains(sources[position].text, quote) for position in cited for quote in case.evidence
        )
    judgement = await judge.grade(
        case, [Passage(s.title, s.text) for s in sources], message.content
    )
    return AnswerResult(
        id=case.id,
        category=case.category,
        answerable=case.answerable,
        question=case.question,
        answer=message.content,
        status=message.status,
        sources=[s.title for s in sources],
        cited=cited,
        cited_evidence=cited_evidence,
        verdict=judgement.verdict.model_dump(),
        answer_cost_usd=float(answer_cost),
        judge_cost_usd=float(judgement.cost_usd or 0),
        conversation_id=conversation.pk,
    )


async def run_answer_eval(
    cases: list[Case],
    service: ChatService,
    judge: Judge,
    user,
    *,
    concurrency: int = 4,
    on_result: Callable[[AnswerResult], None] | None = None,
) -> list[AnswerResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: Case) -> AnswerResult:
        async with semaphore:
            result = await run_case(case, service, judge, user)
        if on_result:
            on_result(result)
        return result

    return list(await asyncio.gather(*(one(c) for c in cases)))


def make_service(client, queries, **kwargs) -> ChatService:
    """The app's ChatService, with query embeddings served from one batch request."""
    return ChatService(client, retriever=partial(retrieve_sources, embedder=queries), **kwargs)


def all_questions(cases: list[Case]) -> list[str]:
    return [q for c in cases for q in (*c.history, c.question)]


# --- Summary --------------------------------------------------------------------------


def summarize(results: list[AnswerResult]) -> dict:
    def verdict(r: AnswerResult) -> Verdict:
        return Verdict(**r.verdict)

    def metrics(group: list[AnswerResult]) -> dict:
        answerable = [r for r in group if r.answerable]
        unanswerable = [r for r in group if not r.answerable]
        return {
            "n": len(group),
            "correctness": average([CORRECTNESS_SCORE[verdict(r).correctness] for r in group]),
            "faithful": average([verdict(r).faithful for r in group]),
            "cited_evidence": average([r.cited_evidence for r in answerable]),
            # Answerable but the model said "not in the documents".
            "false_decline": average([verdict(r).declined for r in answerable]),
            # Not answerable and the model correctly said so.
            "correct_decline": average([verdict(r).declined for r in unanswerable]),
        }

    categories = sorted({r.category for r in results})
    return {
        "overall": metrics(results),
        "categories": {c: metrics([r for r in results if r.category == c]) for c in categories},
        "cost_usd": {
            "answers": sum(r.answer_cost_usd for r in results),
            "judge": sum(r.judge_cost_usd for r in results),
        },
        "failures": [
            r.id for r in results if verdict(r).correctness != "correct" or not verdict(r).faithful
        ],
    }


def format_summary(summary: dict) -> str:
    columns = ["correctness", "faithful", "cited_evidence", "false_decline", "correct_decline"]
    rows = {"overall": summary["overall"], **summary["categories"]}
    width = max(len(name) for name in rows) + 2
    lines = [" " * width + "".join(f"{c:>16}" for c in columns) + f"{'n':>5}"]
    for name, row in rows.items():
        cells = "".join(f"{'-' if row[c] is None else f'{row[c]:.2f}':>16}" for c in columns)
        lines.append(f"{name:<{width}}{cells}{row['n']:>5}")
    cost = summary["cost_usd"]
    lines += [
        "",
        f"Not fully correct or unfaithful: {', '.join(summary['failures']) or '-'}",
        "",
        f"Cost: answers ${cost['answers']:.4f} + judge ${cost['judge']:.4f} "
        f"= ${cost['answers'] + cost['judge']:.4f}",
    ]
    return "\n".join(lines)
