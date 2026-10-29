"""Record an answer eval run as a Langfuse experiment.

The cases are synced to a Langfuse dataset, and the run goes through the SDK's
`run_experiment`: it links every item's trace (the chat-reply traces nest inside) to the
dataset item and the run, so runs can be compared side by side in the Langfuse UI.
The grading stays ours (evals/judge.py); Langfuse only stores the verdicts as scores.

Uses the client from chat/tracing.py, which stays the only module importing langfuse.
"""

from collections.abc import Awaitable, Callable

from .answer_eval import CORRECTNESS_SCORE, AnswerResult, summarize
from .dataset import Case

DATASET_NAME = "docqa-answers"
EXPERIMENT_NAME = "DocQA answers"


def item_id(case: Case) -> str:
    # Item ids are global in a Langfuse project, hence the prefix.
    return f"{DATASET_NAME}-{case.id}"


def sync_dataset(client, cases: list[Case]) -> None:
    """Create or update (upsert by id) one dataset item per case."""
    client.create_dataset(
        name=DATASET_NAME,
        description="DocQA eval questions (evals/dataset.toml) with reference facts.",
    )
    for case in cases:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=item_id(case),
            input={"question": case.question, "history": list(case.history)},
            expected_output={"answerable": case.answerable, "facts": list(case.facts)},
            metadata={"case_id": case.id, "category": case.category},
        )


def scores(result: AnswerResult) -> list[dict]:
    """The judge's verdict and the citation check as numeric scores (averageable)."""
    verdict = result.verdict
    unsupported = "; ".join(verdict["unsupported_claims"])
    missing = "; ".join(verdict["missing_facts"])
    items = [
        {
            "name": "correctness",
            "value": CORRECTNESS_SCORE[verdict["correctness"]],
            "comment": verdict["reasoning"] + (f"\nMissing: {missing}" if missing else ""),
        },
        {
            "name": "faithful",
            "value": float(verdict["faithful"]),
            "comment": f"Unsupported: {unsupported}" if unsupported else None,
        },
        {"name": "declined", "value": float(verdict["declined"])},
    ]
    if result.cited_evidence is not None:
        items.append({"name": "cited_evidence", "value": float(result.cited_evidence)})
    return items


def run_summary_scores(results: list[AnswerResult]) -> list[dict]:
    overall = summarize(results)["overall"]
    return [
        {"name": f"avg_{name}", "value": overall[name]}
        for name in ("correctness", "faithful", "cited_evidence", "false_decline")
        if overall[name] is not None
    ]


def run_experiment(
    client,
    cases: list[Case],
    run_one: Callable[[Case], Awaitable[AnswerResult]],
    *,
    run_name: str,
    concurrency: int,
    metadata: dict[str, str],
    on_result: Callable[[AnswerResult], None] | None = None,
) -> tuple[list[AnswerResult], str | None]:
    """Run the cases as one experiment; returns the results (in case order) and its URL."""
    sync_dataset(client, cases)
    by_item = {item_id(c): c for c in cases}
    items = [i for i in client.get_dataset(DATASET_NAME).items if i.id in by_item]
    results: dict[str, AnswerResult] = {}

    async def task(*, item, **kwargs):
        result = await run_one(by_item[item.id])
        results[result.id] = result
        if on_result:
            on_result(result)
        return result.answer

    def evaluator(*, metadata, **kwargs):
        result = results.get(metadata["case_id"])
        return scores(result) if result else []

    def run_evaluator(*, item_results, **kwargs):
        return run_summary_scores(list(results.values()))

    experiment = client.run_experiment(
        name=EXPERIMENT_NAME,
        run_name=run_name,
        data=items,
        task=task,
        evaluators=[evaluator],
        run_evaluators=[run_evaluator],
        max_concurrency=concurrency,
        metadata=metadata,
    )
    # The SDK logs a failed item and goes on; the caller reports the missing ones.
    return [results[c.id] for c in cases if c.id in results], experiment.experiment_url
