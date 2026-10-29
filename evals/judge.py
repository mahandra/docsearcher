"""LLM-as-judge: grade an answer for correctness (vs reference facts) and faithfulness
(vs the passages the answering model saw).

A second model is used as the grader because "is this answer right?" cannot be checked
with string matching: paraphrases, other languages and partial answers all occur.
Its verdict is constrained to a JSON schema (structured outputs), so it always parses.

This is a development tool with its own Anthropic client; the app itself still talks to
the API only through chat/llm.py.
"""

from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Literal

import anthropic
from django.conf import settings
from pydantic import BaseModel

from chat.pricing import estimate_cost

from .dataset import Case

JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_EFFORT = "medium"
PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.md"


class Verdict(BaseModel):
    reasoning: str  # first: the model writes its rationale before committing to scores
    declined: bool
    correctness: Literal["correct", "partial", "incorrect"]
    missing_facts: list[str]
    faithful: bool
    unsupported_claims: list[str]


@dataclass(frozen=True)
class Passage:
    """A source as the answering model saw it: the title matters, the model may name it."""

    title: str
    text: str


@dataclass(frozen=True)
class Judgement:
    verdict: Verdict
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None


@cache
def load_judge_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_judge_input(case: Case, passages: list[Passage], answer: str) -> str:
    parts = []
    if case.history:
        earlier = "\n".join(f"- {q}" for q in case.history)
        parts.append(f"<earlier_questions>\n{earlier}\n</earlier_questions>")
    parts.append(f"<question>\n{case.question}\n</question>")
    if case.answerable:
        facts = "\n".join(f"- {f}" for f in case.facts)
        parts.append(f"<reference>\nA correct answer states:\n{facts}\n</reference>")
    else:
        parts.append(
            "<reference>\nThe documents do not answer this question. A correct answer "
            "says so and does not invent the information.\n</reference>"
        )
    numbered = "\n\n".join(
        f'<passage index="{i}" title="{p.title}">\n{p.text}\n</passage>'
        for i, p in enumerate(passages, start=1)
    )
    parts.append(f"<passages>\n{numbered or '(none)'}\n</passages>")
    parts.append(f"<answer>\n{answer}\n</answer>")
    return "\n\n".join(parts)


class Judge:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        model: str = JUDGE_MODEL,
        effort: str = JUDGE_EFFORT,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort

    async def grade(self, case: Case, passages: list[Passage], answer: str) -> Judgement:
        response = await self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=load_judge_prompt(),
            messages=[{"role": "user", "content": build_judge_input(case, passages, answer)}],
            output_config={"effort": self.effort},
            output_format=Verdict,
        )
        if response.parsed_output is None:
            # Refusal or max_tokens: the output may not match the schema.
            raise RuntimeError(f"Judge returned no verdict (stop_reason={response.stop_reason})")
        usage = response.usage
        return Judgement(
            verdict=response.parsed_output,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=estimate_cost(response.model, usage.input_tokens, usage.output_tokens),
        )


def get_judge(model: str = JUDGE_MODEL) -> Judge:
    client = anthropic.AsyncAnthropic(
        api_key=settings.ANTHROPIC_API_KEY or None,
        max_retries=settings.LLM_MAX_RETRIES,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        open_telemetry=False,  # see get_chat_service
    )
    return Judge(client, model=model)
