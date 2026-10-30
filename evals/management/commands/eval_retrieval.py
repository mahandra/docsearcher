import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from chat.llm import get_chat_service
from evals.dataset import DATASET_PATH, EVALS_DIR, load_dataset
from evals.ingest import ensure_corpus, get_eval_user
from evals.retrieval_eval import format_summary, run_retrieval_eval, summarize


class Command(BaseCommand):
    help = (
        "Measure retrieval quality on the eval set. Ingests the eval corpus for the "
        "'docqa-eval' user if needed (real Voyage calls: costs a fraction of a cent)."
    )

    def add_arguments(self, parser):
        parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
        parser.add_argument(
            "--reingest",
            action="store_true",
            help="Re-chunk and re-embed the corpus (needed after changing CHUNK_* settings).",
        )
        parser.add_argument(
            "--rewrite",
            action="store_true",
            help="Rewrite follow-ups into standalone queries first, as the app does "
            "(one LLM_REWRITE_MODEL call per follow-up: a fraction of a cent).",
        )
        parser.add_argument("--output", type=Path, help="JSON report path.")

    def handle(self, *args, dataset, reingest, rewrite, output, **options):
        rewriter = None
        if rewrite:
            if not settings.LLM_REWRITE_MODEL:
                raise CommandError("--rewrite needs LLM_REWRITE_MODEL")
            rewriter = get_chat_service().rewriter
        data = load_dataset(dataset)
        documents = ensure_corpus(data, reingest=reingest)
        report = run_retrieval_eval(data, documents, get_eval_user(), rewriter=rewriter)
        summary = summarize(report)
        self.stdout.write(format_summary(report, summary))

        if output is None:
            stamp = timezone.now().strftime("%Y%m%d-%H%M%S")
            output = EVALS_DIR / "results" / f"retrieval-{stamp}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"summary": summary, **report.to_dict()}, ensure_ascii=False, indent=2)
        )
        self.stdout.write(f"\nReport: {output}")
