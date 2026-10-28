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
- PostgreSQL + pgvector (`pgvector` Python package), Redis + Celery worker, Docker Compose.
- Embeddings: Voyage AI (`voyageai` SDK, model `voyage-4`, 1024 dims) — Anthropic has no embeddings API.
- Parsing: `pypdf` (PDF), `python-docx` (DOCX). Test fixtures are generated with `reportlab`/`python-docx`.
- Anthropic Python SDK (`AsyncAnthropic`, SDK 2.x). Frontend: Django templates + HTMX + small vanilla JS.
- pytest + pytest-django + pytest-asyncio (`asyncio_mode = "auto"`), ruff.

## Architecture rules
- **Only `chat/llm.py` imports the Anthropic SDK** (the dev-only eval judge, `evals/judge.py`,
  has its own client). Views talk to the service layer and receive domain
  events (`SourcesFound`, `TextDelta`, `Cited`, `Completed`, `Failed`), never SDK objects.
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
- **Only `documents/embeddings.py` imports `voyageai`.** Callers use the `Embedder` protocol; errors are
  mapped to `TransientEmbeddingError` (retry) / `EmbeddingAuthError` / `EmbeddingError` (permanent).
- Ingestion pipeline (`documents/tasks.py`): detect type by magic bytes → parse (keep page/heading) →
  chunk (`documents/chunking.py`) → embed with `input_type="document"` → atomically replace chunks.
- Celery tasks must be idempotent (`acks_late`); enqueue only via `transaction.on_commit`
  (`enqueue_processing`). No network calls inside `transaction.atomic()`.
- Embedding dimension is fixed in the schema (`EMBEDDING_DIMENSIONS`); changing model/dims = re-embed all.
- Token counts in chunking are estimates (`CHARS_PER_TOKEN`); billed usage comes from the provider.
- Retrieval (`documents/retrieval.py`): vector + full-text search merged with RRF, always scoped to
  the user's `ready` documents. Queries are embedded with `input_type="query"`. Full-text uses a
  generated `Chunk.search_vector` (english + russian); query words are stemmed per script.
- RAG prompt: retrieved chunks go into the **latest** user turn only, as `search_result` blocks with
  `citations: {enabled: true}` (all-or-nothing per request). Never send `citations` back inside a
  `user` message (400 on API version 2026-10-21). Do not prompt for `[n]` markers — use native
  citations. `ChatService` takes a `retriever`; tests inject a fake one.
- Read citations from the finished text block (`content_block_stop`), not from `citation` deltas:
  the API may send a block's citation before its text.
- `MessageSource` copies document name/location; FKs to Chunk/Document are `SET_NULL` and are
  checked before saving (documents can vanish mid-generation).

## Tests
- No real API calls in tests, ever. Fakes: `tests/fakes.py` (Anthropic, incl. `cite()`),
  `tests/documents/fakes.py` (Voyage). An autouse fixture replaces the query embedder everywhere.
- Async ORM tests need `django_db(transaction=True)` + the `close_async_db_connections` fixture.
- Celery tasks are tested with `task.apply()`; on_commit with `django_capture_on_commit_callbacks`.
- Tests run against Postgres (`make test` runs inside Docker).

## Commands
`make up | down | migrate | makemigrations | test | lint | fmt | shell | superuser | logs`
Run on the host: `uv run pytest`, `uv run python manage.py ...` (uses DATABASE_URL from .env → localhost:5434).
After `uv add`, run `make build` — the Docker image does not see new dependencies otherwise.
Containers run as the host UID (`HOST_UID`/`HOST_GID` exported by the Makefile); use `make`, not raw
`docker compose`, so files in the bind mount are not root-owned.
Changes to `.env` need `docker compose up -d --force-recreate web worker`.
