# AI Judge

A retrieval-augmented generation (RAG) prototype that answers legal questions **only** from a
supplied corpus, in Arabic, with inline source citations. Given a question, it retrieves the most
relevant passages from a legal corpus, feeds them to a local LLM under strict grounding
instructions, and returns an answer whose every claim is tied to a numbered source — or refuses to
answer when the corpus doesn't support one.

> **Status: learning prototype.** This is an educational project, not a legal product. It runs
> entirely against local models and a small sample corpus. It must **not** be used to make, inform,
> or influence real legal decisions. LLM output can be wrong or fabricated even when it cites a
> source — always verify against the primary text.

---

## How it works

```
                 ingestion (one-time)                         query (per request)
   .txt corpus ──► chunk ──► embed ──► Qdrant        question ──► embed ──► vector search
                  (500/50)   (BGE-M3)  (vectors)                          │
                                                                          ▼
                                                        relevance gate (score ≥ 0.5?)
                                                            │no ──────────► refuse
                                                            │yes
                                                            ▼
                                            grounded Arabic prompt (numbered sources)
                                                            │
                                                            ▼
                                                  local LLM (LM Studio)
                                                            │
                                                            ▼
                                       answer + sources + invalid-citation check
```

- **Grounded-only.** The prompt instructs the model to use the numbered sources and nothing else,
  to cite each claim as `[n]`, and to say so plainly when the sources are insufficient.
- **Relevance gate.** If the top retrieval score is below `min_relevance_score` (default `0.5`), the
  API refuses rather than handing the model weak context. This cutoff is a heuristic, not calibrated.
- **Citation validation.** After generation, the backend flags any `[n]` the model cited that
  doesn't map to a real source (`invalid_citations`).

## Tech stack

| Layer      | Choice                                                                 |
| ---------- | ---------------------------------------------------------------------- |
| Backend    | FastAPI · Uvicorn · httpx · pydantic-settings (Python 3.11+)           |
| Frontend   | Angular 21 · Tailwind CSS 4 · `marked` · Vitest                        |
| Vector DB  | Qdrant (cosine distance)                                               |
| Models     | Served locally by **LM Studio** (OpenAI-compatible API)                |

Defaults (see [`backend/app/config.py`](backend/app/config.py)): chat model `google/gemma-4-e4b`,
embedding model `text-embedding-bge-m3` (1024 dims), Qdrant collection `legal_corpus`.

---

## Prerequisites

You must have these running/installed **before** starting the app:

1. **Python 3.11+** and **Node.js 20+ / npm 10+**.
2. **[LM Studio](https://lmstudio.ai/)** with the local server started on `http://localhost:1234`,
   and **both** a chat model and an embedding model loaded. Model names must match
   `backend/app/config.py` (or override them — see Configuration).
3. **[Qdrant](https://qdrant.tech/)** reachable at `http://localhost:6333`, e.g.:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Setup

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest the corpus (one-time, and after any corpus change)

With LM Studio and Qdrant running, from `backend/`:

```bash
python -m app.ingestion.run
```

This loads every `.txt` in [`backend/data/sample_corpus/`](backend/data/sample_corpus/), chunks it
(500 chars, 50 overlap), embeds it, and upserts it into the `legal_corpus` collection. Re-running
overwrites existing points by id.

### 3. Run the backend

From `backend/`:

```bash
uvicorn app.main:app --reload
```

API on `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

### 4. Run the frontend

```bash
cd frontend
npm install
npm start
```

App on `http://localhost:4200` (CORS is preconfigured for exactly this origin).

---

## Configuration

Settings load from environment variables or a `backend/.env` file (see `config.py`). Create
`backend/.env` only if you need to override defaults:

```dotenv
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=google/gemma-4-e4b
LM_STUDIO_EMBEDDING_MODEL=text-embedding-bge-m3
EMBEDDING_DIMENSION=1024
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=legal_corpus
MIN_RELEVANCE_SCORE=0.5
```

> If you change `LM_STUDIO_EMBEDDING_MODEL`, make sure `EMBEDDING_DIMENSION` matches it, then
> **re-ingest** — the Qdrant collection is created with a fixed vector size.

## API

| Method | Path           | Purpose                                                             |
| ------ | -------------- | ------------------------------------------------------------------- |
| `GET`  | `/health`      | Liveness check.                                                     |
| `GET`  | `/dashboard`   | Corpus stats (documents, chunks) + health of Qdrant and LM Studio.  |
| `GET`  | `/resources`   | Every stored chunk, grouped by source document.                     |
| `POST` | `/retrieve`    | Vector search only — returns matching chunks (optional source filter). |
| `POST` | `/ask`         | Full RAG — returns `answer`, `sources`, and `invalid_citations`.    |
| `POST` | `/ask/stream`  | Same as `/ask`, streamed as Server-Sent Events (`stage`/`token`/`done`/`refused`). |
| `POST` | `/chat`        | Raw single-turn LLM call — **no retrieval, no grounding** (debug only). |

Example:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي أركان العقد؟", "limit": 5}'
```

The frontend exposes three pages: **Ask** (`/ask`, default), **Dashboard** (`/dashboard`), and
**Resources** (`/resources`).

## Project layout

```
backend/
  app/
    api/            # FastAPI routers: ask, chat, retrieve, resources, dashboard, health
    ingestion/      # loader → chunker → run (corpus → Qdrant)
    config.py       # settings (env / .env)
    embedding_client.py, llm_client.py   # LM Studio clients
    vector_store.py # Qdrant client
    generation.py   # grounded prompt + citation validation
  data/sample_corpus/   # sample .txt legal overviews
frontend/
  src/app/          # ask, dashboard, resources components
```

## Testing

```bash
cd frontend
npm test          # Vitest
```

---

## Limitations

- Sample corpus is a handful of short, general legal overviews — **not** authoritative law.
- Chunking is naive fixed-size character splitting; it can cut mid-sentence.
- **The relevance threshold is an un-calibrated heuristic, and measurably miscalibrated.**
  See below.
- Prompt and answers are Arabic-only.
- No authentication, rate limiting, persistence beyond Qdrant, or conversational memory.

### The relevance threshold is refusing answerable questions

`min_relevance_score` is `0.5`. Probing the corpus with questions it demonstrably
answers turns up false refusals — the gate fires and the user is told there isn't
enough information, when the answer is sitting in a retrieved document:

| Question | Top score | Where the answer actually is |
|---|---:|---|
| ما المعيار الحاكم في إسناد الحضانة؟ | 0.491 | `family_law_overview` §4 — the section *is* الحضانة / المصلحة الفضلى |
| من هو الحامل حسن النية؟ | 0.444 | `negotiable_instruments_overview` §3 — the section *is* الحامل حسن النية |
| ما حكم تعسف المالك في استعمال حقه؟ | 0.497 | `property_law_overview` §4 — states the rule verbatim |

The two populations aren't cleanly separated at 0.5 in either direction: a genuinely
out-of-corpus question about bankruptcy priority scores **0.484**, above two of the
answerable questions above. Answerable questions land roughly 0.44–0.73 and
out-of-corpus ones roughly 0.26–0.48, and those ranges overlap.

Nothing here has been tuned — 0.5 was a guess, and it is the one number in the
pipeline that has never been calibrated against data. Doing it properly means
probing both populations, plotting the distributions, and picking the separation
point, rather than nudging the constant until a particular question behaves.

Deliberately left at 0.5 for now so the committed eval baseline measures the system
as it actually behaves. `eval/cases.yaml` includes `refused` cases spanning
0.26–0.48 specifically so that moving this threshold shows up as a scorecard change
rather than a silent one.
