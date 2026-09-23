# EchoAgent Roadmap

Living status doc. Updated as work progresses. For deferred / Phase 2 items see `docs/future.md`. For architectural decisions see `docs/architecture.md`.

---

## Current Status

### Fully Implemented

- `VibeIntent` contract: normalization, validation, constraint separation
- `PromptParser`: schema-driven LLM parsing with retry-and-repair loop
- `PlannerAgent`: structured `PlaylistPlan` output with scoring weights and retrieval limits
- `RelationalRetrievalAgent` + `RelationalRetrievalMapper`: SQL hard-filter retrieval from `VibeIntent.hard_constraints`
- `VectorRetrievalAgent`: ChromaDB semantic retrieval using `VibeIntent.semantic_query`
- `CandidateFuser`: merges relational and vector candidates by `track_id`, tracks source origin
- `Reranker`: deterministic scoring using planner-supplied weights, preserves score components per candidate
- `PlaylistBuilderAgent`: final track selection with artist-repeat and genre-concentration controls
- `playlist_graph.py`: full LangGraph orchestration with parallel retrieval fan-out, critic routing, and retry loop
- `graph_state.py`: shared typed state schema (`PlaylistGraphState`)
- Relational DB: SQLAlchemy models, ingestion pipelines, audio features, tags
- Vector DB: ChromaDB collections, sentence-transformer embeddings, semantic search utilities
- Partial test coverage: `test_api`, `test_candidate_fuser`, `test_reranker`, `test_embeddings`, `test_planner_agent`, `test_vector_retrieval_agent`

### Stubs / Incomplete

- **`critic_agent.py`**: the `CriticAgent` class itself is implemented (LLM-backed accept/reason/suggested_adjustments) and verified working standalone in `notebooks/pipeline_tests.ipynb`. It is just not wired into `run_playlist_graph()` — the convenience runner never constructs/passes a `critic_agent`, so `critique_node` always auto-accepts. Wiring it in is Phase 5 of `docs/web_app_full_data_plan.md`.
- **Graph/critic integration tests**: `tests/test_api.py` now covers the mocked FastAPI contract. End-to-end graph integration tests and critic-routing tests remain open.
- **Frontend**: none exists yet. Decided approach: Streamlit calling the FastAPI backend over HTTP (not calling `run_playlist_graph()` directly), so a later React frontend is a drop-in swap. See `docs/web_app_full_data_plan.md`.

### Phase 1 — Backend MVP: implemented and partially verified

`backend/api/main.py` and `backend/api/routes/recommend.py` are both fully implemented (`POST /recommend` calls `run_playlist_graph()`, translates the result to `RecommendResponse`, classifies errors into the four documented failure modes). The `GROQ_MODEL` default bug (below) is fixed. Startup config is now verified against the current subset data paths. `/health` returns `{"status":"ok"}`. A live `/recommend` request reaches the graph/data-loading path, but a clean successful response still needs re-test after the playlist-builder JSON/fallback changes.

- `main.py` uses a `lifespan` context manager to build `LLMClient`/`PlannerAgent`/`PlaylistBuilderAgent` once at startup (stored on `app.state`), reading `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_TEMPERATURE`, `GROQ_MAX_TOKENS`, `DATABASE_URL`, `CHROMA_PERSIST_DIRECTORY` from `.env` / the environment.
- `recommend.py` uses FastAPI `Depends()` reading those `app.state` singletons (so tests can override via `app.dependency_overrides`), builds a fresh `PromptParser` per request, and translates candidates to `PlaylistTrack` via `_to_playlist_track()` — including `_extract_genre()`, which correctly reconciles the `seed_genre` (relational) vs. `genres_csv` (vector) mismatch noted below.
- **Does not yet backfill missing `title`/`artist_name`** for vector-only candidates — see Known Issues, still open.
- LLM client: all four structured agents request JSON mode on every attempt. Provider JSON-validation failures, empty/truncated output, and schema errors enter bounded repairs; there is no plain-text retry. Provider errors such as 429 propagate separately. Builder validation exhaustion retains ranked-track fallback for critic review, exposed via `debug.builder_fallback_used` and `debug.builder_fallback_reason`. See `docs/web_app_full_data_plan.md` for offline JSON/graph test coverage and diagnostic logging.

---

## Known Issues

- There appear to be two copies of the relational DB: `backend/data/music_relational.db` and `database/music_relational.db`. Should be consolidated.
- **Vector-only candidates can be missing `title`/`artist_name`**: known issue, explicitly deferred by the user on 2026-09-23; move on with other work and revisit later. Deferred scope also includes normalizing byte-representation title/artist/album strings and malformed tags. This is unresolved, not a completed fix. `fuse_candidates()` processes retrieved relational candidates first, but it only checks the relational results returned for the request; it does not query SQLite by `track_id` for every vector result. Exact example showing what remains open: SQLite has A, B, C, D; relational retrieval returns A, B; vector retrieval returns B, C. A uses relational metadata. B keeps relational metadata and adds vector score/source. C uses vector metadata because it was absent from the retrieved relational candidates, even if C exists in SQLite. If C's vector metadata only has `track_id` or is missing display fields, the API can return a playlist row without title/artist. `track_id` is sufficient for identity, but not sufficient for display unless the API/front end hydrates by ID before rendering. `EmbeddingManager.track_metadata()` in `backend/data/embeddings.py` only adds `title`/`artist_name` to Chroma metadata `if value:`, so missing ingestion values can become omitted keys. `backend/api/routes/recommend.py`'s `_to_playlist_track()` still reads whatever is in `metadata` as-is. Genre keying (`seed_genre` vs. `genres_csv`) is already reconciled correctly by `_extract_genre()`.
- **Deprecated Groq models**: `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` no longer exist on Groq (confirmed via `client.models.list()`). `notebooks/pipeline_tests.ipynb` has been updated off these. `qwen/qwen3.6-27b` is available but is a reasoning model that wraps output in `<think>` tags by default, which breaks every agent's JSON extraction — avoid it unless called with `reasoning_effort="none"`. `openai/gpt-oss-120b` / `openai/gpt-oss-20b` are clean by default and currently the project's standard.

---

## Next Priorities

`docs/web_app_full_data_plan.md` Phase 0 is complete. Phase 1 (`POST /recommend`, error handling, schemas, startup config, and mocked API tests) is implemented. Startup and `/health` are verified. Resume here:

1. **Re-test `/recommend`** — restart Uvicorn, hit it with a real HTTP request, and confirm the playlist-builder JSON/fallback changes now return a successful response
2. **Metadata enrichment — deferred (2026-09-23)** — revisit missing title/artist fields, byte-representation strings, and malformed tags later; do not treat this as the immediate next task (see Known Issues above)
3. **Build the Streamlit frontend** — `frontend/app.py`, calling `/recommend` over HTTP
4. **Wire `CriticAgent` into `run_playlist_graph()`** — Phase 5, deferred until the above is working end-to-end
5. **Expand tests** — `tests/test_api.py` now has mocked API contract tests passing; graph integration tests and critic behavior tests remain open
