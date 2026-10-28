"""Retrieval eval: run every question through search() and score the rankings.

Each question is searched once; the vector, keyword and fused (hybrid) rankings all come
from that one call, so the three modes are compared on identical inputs.
"""

from dataclasses import asdict, dataclass, field
from statistics import median

from django.conf import settings

from chat.pricing import estimate_embedding_cost
from documents.embeddings import Embedder, EmbeddingResult, get_embedder
from documents.models import Chunk, Document
from documents.retrieval import SearchHit, search

from .dataset import Case, Dataset, contains
from .metrics import average, hit_at, recall_at, reciprocal_rank, relevant_ranks

MODES = ("hybrid", "vector", "keyword")


@dataclass
class ModeResult:
    chunks: list[str]  # "doc-key#index", in rank order
    relevant_ranks: list[int]
    hits: dict[int, bool]  # k -> hit@k
    recall: float  # recall@top_k
    reciprocal_rank: float


@dataclass
class CaseResult:
    id: str
    category: str
    answerable: bool
    query: str
    modes: dict[str, ModeResult]
    top_distance: float | None  # nearest chunk overall
    relevant_distance: float | None  # nearest relevant chunk


@dataclass
class RetrievalReport:
    settings: dict
    k_values: list[int]
    cases: list[CaseResult]
    unreachable: list[str] = field(default_factory=list)  # quotes split between chunks
    embedding_tokens: int = 0
    embedding_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class BatchedQueries:
    """Embeds all eval questions in one request, then serves them to search() one by one.

    Rate limits count requests (Voyage's free tier allows 3 per minute), and one batch is
    also faster than dozens of round trips. Tokens are counted once, for the batch.
    """

    def __init__(self, embedder: Embedder, queries: list[str]) -> None:
        unique = list(dict.fromkeys(queries))
        result = embedder.embed_queries(unique)
        self.model = result.model
        self.total_tokens = result.total_tokens
        self.vectors = dict(zip(unique, result.vectors, strict=True))

    def embed_query(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vectors=[self.vectors[text]], total_tokens=0, model=self.model)

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        raise NotImplementedError("BatchedQueries only embeds queries")

    def embed_queries(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult([self.vectors[t] for t in texts], 0, self.model)


def k_values(top_k: int) -> list[int]:
    return sorted({1, 3, 5, top_k})


def run_retrieval_eval(
    dataset: Dataset, documents: dict[str, Document], user, *, embedder=None
) -> RetrievalReport:
    top_k, candidates = settings.RETRIEVAL_TOP_K, settings.RETRIEVAL_CANDIDATES
    ks = k_values(top_k)
    labels = {d.pk: key for key, d in documents.items()}
    report = RetrievalReport(
        settings={
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size_tokens": settings.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": settings.CHUNK_OVERLAP_TOKENS,
            "top_k": top_k,
            "candidates": candidates,
            "chunks": Chunk.objects.filter(document__in=documents.values()).count(),
        },
        k_values=ks,
        cases=[],
        unreachable=unreachable_quotes(dataset, documents),
    )
    queries = BatchedQueries(embedder or get_embedder(), [c.question for c in dataset.cases])
    report.embedding_tokens = queries.total_tokens

    for case in dataset.cases:
        # limit=candidates: keep the whole fused list so every k can be scored.
        result = search(user, case.question, limit=candidates, embedder=queries)
        if result.vector_error:
            raise RuntimeError(f"Vector search failed for {case.id}: {result.vector_error}")
        rankings = {
            "hybrid": result.hits,
            "vector": result.vector_hits,
            "keyword": result.keyword_hits,
        }
        relevant_vector = [
            h.vector_distance
            for h in result.vector_hits
            if any(contains(h.chunk.text, e) for e in case.evidence)
        ]
        report.cases.append(
            CaseResult(
                id=case.id,
                category=case.category,
                answerable=case.answerable,
                query=case.question,
                modes={
                    mode: score(hits, case, labels, ks, top_k) for mode, hits in rankings.items()
                },
                top_distance=result.vector_hits[0].vector_distance if result.vector_hits else None,
                relevant_distance=relevant_vector[0] if relevant_vector else None,
            )
        )

    cost = estimate_embedding_cost(settings.EMBEDDING_MODEL, report.embedding_tokens)
    report.embedding_cost_usd = float(cost or 0)
    return report


def score(hits: list[SearchHit], case: Case, labels, ks, top_k) -> ModeResult:
    texts = [h.chunk.text for h in hits]
    return ModeResult(
        chunks=[f"{labels.get(h.chunk.document_id, '?')}#{h.chunk.index}" for h in hits],
        relevant_ranks=relevant_ranks(texts, case.evidence),
        hits={k: hit_at(texts, case.evidence, k) for k in ks},
        recall=recall_at(texts, case.evidence, top_k),
        reciprocal_rank=reciprocal_rank(texts, case.evidence),
    )


def unreachable_quotes(dataset: Dataset, documents: dict[str, Document]) -> list[str]:
    """Quotes no single chunk contains (cut by a chunk boundary): recall can't reach 100%."""
    texts = {
        key: list(Chunk.objects.filter(document=document).values_list("text", flat=True))
        for key, document in documents.items()
    }
    return [
        f"{case.id}: {quote.text}"
        for case in dataset.cases
        for quote in case.evidence
        if not any(contains(text, quote) for text in texts[quote.doc])
    ]


# --- Summary --------------------------------------------------------------------------


def summarize(report: RetrievalReport) -> dict:
    """Aggregate metrics over answerable cases: per mode, and per category for hybrid."""
    answerable = [c for c in report.cases if c.answerable]
    top_k = report.settings["top_k"]

    def metrics(cases, mode):
        results = [c.modes[mode] for c in cases]
        row = {f"hit@{k}": average([r.hits[k] for r in results]) for k in report.k_values}
        row[f"recall@{top_k}"] = average([r.recall for r in results])
        row["mrr"] = average([r.reciprocal_rank for r in results])
        row["n"] = len(results)
        return row

    categories = sorted({c.category for c in answerable})
    return {
        "modes": {mode: metrics(answerable, mode) for mode in MODES},
        "categories": {
            category: metrics([c for c in answerable if c.category == category], "hybrid")
            for category in categories
        },
        "distances": {
            "answerable_relevant": distribution([c.relevant_distance for c in answerable]),
            "answerable_top": distribution([c.top_distance for c in answerable]),
            "unanswerable_top": distribution(
                [c.top_distance for c in report.cases if not c.answerable]
            ),
        },
        "misses": [c.id for c in answerable if not c.modes["hybrid"].hits[top_k]],
    }


def distribution(values) -> dict | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {"min": values[0], "median": median(values), "max": values[-1], "n": len(values)}


def format_summary(report: RetrievalReport, summary: dict) -> str:
    columns = [f"hit@{k}" for k in report.k_values] + [
        f"recall@{report.settings['top_k']}",
        "mrr",
        "n",
    ]

    def table(title, rows):
        width = max(len(name) for name in rows) + 2
        lines = [title, " " * width + "".join(f"{c:>11}" for c in columns)]
        for name, row in rows.items():
            cells = "".join(f"{row[c]:>11}" if c == "n" else f"{_fmt(row[c]):>11}" for c in columns)
            lines.append(f"{name:<{width}}{cells}")
        return "\n".join(lines)

    s = report.settings
    lines = [
        f"Corpus: {s['chunks']} chunks ({s['embedding_model']}, chunk {s['chunk_size_tokens']}"
        f"/{s['chunk_overlap_tokens']} tokens); top_k={s['top_k']}, "
        f"candidates={s['candidates']}",
        "",
        table("Answerable questions by mode:", summary["modes"]),
        "",
        table("Hybrid by category:", summary["categories"]),
        "",
        "Cosine distance of the nearest chunk (lower = closer):",
    ]
    for name, d in summary["distances"].items():
        if d:
            lines.append(
                f"  {name:<22} min {d['min']:.3f}  median {d['median']:.3f}  "
                f"max {d['max']:.3f}  (n={d['n']})"
            )
    lines += ["", f"Missed by hybrid in top {s['top_k']}: {', '.join(summary['misses']) or '-'}"]
    if report.unreachable:
        lines += ["", "Quotes split between chunks (cannot be fully found):"]
        lines += [f"  {q}" for q in report.unreachable]
    lines += [
        "",
        f"Query embeddings: {report.embedding_tokens} tokens, ${report.embedding_cost_usd:.6f}",
    ]
    return "\n".join(lines)


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.2f}"
