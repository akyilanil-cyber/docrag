# DocRAG — async document Q&A

Upload a PDF/TXT/MD file, get a task ID back immediately, watch processing progress live, then ask questions answered from the document's own text.

The point of the project is the architecture: slow work (parsing, chunking, embedding) never runs inside the web request.

```
React ──POST /documents──▶ FastAPI ──enqueue──▶ Redis ──▶ Celery worker
  ▲                          │  ▲                              │
  │   SSE /events            │  └──── progress / status ───────┤
  └──────────────────────────┘                                 ▼
                         PostgreSQL + pgvector  ◀──── chunks + embeddings
```

## Stack
FastAPI · Celery · Redis · PostgreSQL + pgvector · React + TypeScript (Vite) · Docker Compose

## Run it
```bash
cp example.env .env        # set POSTGRES_PASSWORD; ANTHROPIC_API_KEY is optional
docker compose up --build
```
UI: http://localhost:5173 · API docs: http://localhost:8000/docs

Without `ANTHROPIC_API_KEY` the app answers **extractively** (returns the best matching passage), so it costs nothing and works offline. With a key it asks the model to answer from the retrieved passages only.

## Design decisions
| Decision | Why |
|---|---|
| Celery + Redis instead of `BackgroundTasks` | Work survives API restarts, retries, scales to many workers. `BackgroundTasks` runs in the web process. |
| `acks_late` + `reject_on_worker_lost` + idempotent task | If a worker dies mid-task the job is redelivered; the pipeline clears old chunks first so nothing is duplicated (tested). |
| Retries only for transient errors | DB outages retry with backoff. Bad input (empty/corrupt file) fails immediately with a clear message instead of retrying forever. |
| SSE instead of WebSocket | Progress is one-way; SSE is simpler and reconnects by itself. |
| pgvector `cosine_distance` | Similarity search runs in the database, not in Python. SQLite fallback exists only so unit tests run without Docker. |
| Deterministic hashing embedder | No model download or API key needed, reproducible tests. It sits behind a tiny `Embedder` protocol so a real model can replace it. Quality is keyword-level, not semantic. |

## Security notes (OWASP Top 10 for LLM Applications, 2025)
- **Prompt injection (LLM01):** document text is sent inside `<document>` tags with a system prompt that marks it as untrusted data. Model output is rendered as plain text and never executed or used in SQL.
- **Unbounded consumption (LLM10):** per-IP rate limit (Redis), question length cap, hard `max_tokens`, request timeout, 10 MB upload cap.
- **Sensitive info (LLM02):** no secrets in code; keys come from environment variables; `.env` is git-ignored.
- **Uploads:** extension whitelist plus content check (`%PDF` magic bytes / UTF-8), random on-disk names (no path traversal), size enforced while streaming, partial files deleted on rejection.
- API container runs as a non-root user; DB and Redis ports bind to localhost only.

## Tests
```bash
cd backend
pip install -r requirements.txt
pytest -q
```
19 tests cover chunking, embeddings, upload validation, the full upload → process → ask flow, idempotency, rate limiting, SSE, and the LLM-failure fallback. Celery runs in eager mode and the database is SQLite, so no Docker is needed for tests.

## Known limits (honest list)
- No authentication or per-user isolation: every visitor sees every document. A multi-tenant version needs auth and per-tenant filtering (see the roadmap).
- Scanned PDFs without a text layer produce no text (no OCR).
- The Postgres/pgvector, Redis and Docker Compose path is written but the automated tests do not exercise it; test it with `docker compose up` before relying on it.
- Embeddings are lexical (hashing trick), not semantic.

## Roadmap
JWT auth with tenant-scoped queries · real embedding model · OCR · answer streaming · CI pipeline
