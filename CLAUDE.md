# DocQA — project conventions

Django web app: users upload documents and ask questions answered by an LLM via RAG with citations.
Learning/portfolio project — the author is an experienced Django engineer new to LLM integration.

## Working agreements
- Talk to the user in **Russian**; write all code, comments, commit messages and docs in **English**.
- Build strictly phase by phase (see Roadmap in README.md). Never start the next phase unprompted.
- Before coding a phase: present a plan and wait for approval.
- Small steps, one commit per logical step. After each step explain (in Russian): what was done,
  the LLM concept behind it, and how to verify it.

## Stack
- Python 3.12, Django 5.2, uv (`uv add`, `uv run`), served via **ASGI/uvicorn only** (no wsgi.py).
- PostgreSQL (`pgvector/pgvector` image), Redis (Celery from phase 2), Docker Compose.
- Anthropic Python SDK (`AsyncAnthropic`, SDK 2.x). Frontend: Django templates + HTMX + small vanilla JS.
- pytest + pytest-django + pytest-asyncio (`asyncio_mode = "auto"`), ruff.

## Architecture rules
- **Only `chat/llm.py` imports the Anthropic SDK.** Views talk to the service layer and receive domain
  events (`TextDelta`, `Completed`, `Failed`), never SDK objects.
- System prompts live in `chat/prompts/*.md`, not in Python strings.
- Model pricing lives in `chat/pricing.py`; cost is computed from `usage` and stored per message.
- Model, effort, max_tokens, history window, retries — all configurable via env (`LLM_*`).
- Do not send `temperature`/`top_p`/`top_k` (rejected by the current API version).
- Retries: rely on the SDK's `max_retries` (backoff + retry-after). Never retry after the first
  token was streamed — save a partial answer with status `interrupted` instead.
- Streaming: SSE via `StreamingHttpResponse` + async generator; the browser uses native `EventSource`
  and must close it on `done`/`error` (otherwise it auto-reconnects and re-triggers generation).
  The server only generates for a user message that has no assistant reply yet.
- Everything user-owned is scoped to `request.user` (Django built-in auth).

## Tests
- No real API calls in tests, ever. Mock the client injected into the service layer.
- Tests run against Postgres (`make test` runs inside Docker).

## Commands
`make up | down | migrate | makemigrations | test | lint | fmt | shell | superuser | logs`
Run on the host: `uv run pytest`, `uv run python manage.py ...` (uses DATABASE_URL from .env → localhost:5434).
