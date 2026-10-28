import asyncio
import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from chat.llm import get_chat_service
from chat.models import Conversation
from documents.embeddings import get_embedder
from evals.answer_eval import (
    AnswerReport,
    all_questions,
    format_summary,
    make_service,
    run_answer_eval,
    summarize,
)
from evals.dataset import DATASET_PATH, EVALS_DIR, load_dataset
from evals.ingest import ensure_corpus, get_eval_user
from evals.judge import JUDGE_MODEL, get_judge
from evals.retrieval_eval import BatchedQueries

# Measured on a trial run: answer ~$0.036 (Opus, low effort, ~6K input tokens of
# sources) + judge ~$0.012 (Sonnet, medium effort).
ESTIMATED_COST_PER_TURN = 0.05


class Command(BaseCommand):
    help = (
        "Answer the eval questions with the real ChatService and grade the answers with "
        "an LLM judge. Calls the Claude API: costs real money, so --yes is required."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
        parser.add_argument("--case", action="append", default=[], help="Only these case ids.")
        parser.add_argument("--category", action="append", default=[])
        parser.add_argument("--limit", type=int, help="At most this many cases.")
        parser.add_argument("--concurrency", type=int, default=4)
        parser.add_argument("--judge-model", default=JUDGE_MODEL)
        parser.add_argument("--yes", action="store_true", help="Confirm the API spend.")
        parser.add_argument("--output", type=Path, help="JSON report path.")

    def handle(self, *args, **options):
        dataset = load_dataset(options["dataset"])
        cases = [
            c
            for c in dataset.cases
            if (not options["case"] or c.id in options["case"])
            and (not options["category"] or c.category in options["category"])
        ][: options["limit"]]
        if not cases:
            raise CommandError("No cases selected.")

        turns = sum(1 + len(c.history) for c in cases)
        estimate = turns * ESTIMATED_COST_PER_TURN
        self.stdout.write(f"{len(cases)} cases, {turns} answers: about ${estimate:.2f}.")
        if not options["yes"]:
            self.stdout.write("Dry run. Add --yes to call the API.")
            return

        documents = ensure_corpus(dataset)
        user = get_eval_user()
        # Keep only the latest run's conversations (handy for inspecting answers).
        Conversation.objects.filter(user=user).delete()
        queries = BatchedQueries(get_embedder(), all_questions(cases))
        service = make_service(get_chat_service().client, queries)
        judge = get_judge(options["judge_model"])

        def progress(result):
            mark = "ok" if result.verdict["correctness"] == "correct" else "--"
            self.stdout.write(f"  {mark} {result.id}: {result.verdict['correctness']}")

        results = asyncio.run(
            run_answer_eval(
                cases,
                service,
                judge,
                user,
                concurrency=options["concurrency"],
                on_result=progress,
            )
        )
        report = AnswerReport(
            settings={
                "model": service.model,
                "effort": service.effort,
                "judge_model": judge.model,
                "judge_effort": judge.effort,
                "documents": {key: d.original_name for key, d in documents.items()},
            },
            cases=sorted(results, key=lambda r: [c.id for c in cases].index(r.id)),
        )
        summary = summarize(report.cases)
        self.stdout.write("\n" + format_summary(summary))

        output = options["output"]
        if output is None:
            stamp = timezone.now().strftime("%Y%m%d-%H%M%S")
            output = EVALS_DIR / "results" / f"answers-{stamp}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"summary": summary, **report.to_dict()}, ensure_ascii=False, indent=2)
        )
        self.stdout.write(f"\nReport: {output}")
