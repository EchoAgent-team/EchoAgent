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
- Partial test coverage: `test_candidate_fuser`, `test_reranker`, `test_embeddings`, `test_planner_agent`, `test_vector_retrieval_agent`

### Stubs / Incomplete

- **`critic_agent.py`**: the `CriticAgent` class itself is implemented (LLM-backed accept/reason/suggested_adjustments) and verified working standalone in `notebooks/pipeline_tests.ipynb`. It is just not wired into `run_playlist_graph()` — the convenience runner never constructs/passes a `critic_agent`, so `critique_node` always auto-accepts. Wiring it in is Phase 5 of `docs/web_app_full_data_plan.md`.
- **`tests/test_api.py`**: placeholder, no real tests yet.
- **Frontend**: none exists yet. Decided approach: Streamlit calling the FastAPI backend over HTTP (not calling `run_playlist_graph()` directly), so a later React frontend is a drop-in swap. See `docs/web_app_full_data_plan.md`.

### Phase 1 — Backend MVP: implemented, not yet verified working

`backend/api/main.py` and `backend/api/routes/recommend.py` are both fully implemented (`POST /recommend` calls `run_playlist_graph()`, translates the result to `RecommendResponse`, classifies errors into the four documented failure modes). The `GROQ_MODEL` default bug (below) is fixed. **Still not yet confirmed working end-to-end via a real HTTP request** — that verification hasn't been done.

- `main.py` uses a `lifespan` context manager to build `LLMClient`/`PlannerAgent`/`PlaylistBuilderAgent` once at startup (stored on `app.state`), reading `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_TEMPERATURE`, `GROQ_MAX_TOKENS`, `DATABASE_URL`, `CHROMA_PERSIST_DIRECTORY` from the environment.
- `recommend.py` uses FastAPI `Depends()` reading those `app.state` singletons (so tests can override via `app.dependency_overrides`), builds a fresh `PromptParser` per request, and translates candidates to `PlaylistTrack` via `_to_playlist_track()` — including `_extract_genre()`, which correctly reconciles the `seed_genre` (relational) vs. `genres_csv` (vector) mismatch noted below.
- **Does not yet backfill missing `title`/`artist_name`** for vector-only candidates — see Known Issues, still open.
- LLM client: after some churn this session (a redundant `GroqLLMClient` module was built, then deleted once it turned out `LLMClient` in `prompt_parser.py` already gained real Groq support via a parallel commit), the project has standardized on `LLMClient(provider="groq", ...)` from `prompt_parser.py`. Its `provider` param now defaults to `"groq"` whenever `api_key` is set (previously silently defaulted to `"hf_api"` — fixed this session).

---

## Known Issues

- There appear to be two copies of the relational DB: `backend/data/music_relational.db` and `database/music_relational.db`. Should be consolidated.
- **Vector-only candidates can be missing `title`/`artist_name`**: open issue. `fuse_candidates()` processes retrieved relational candidates first, but it only checks the relational results returned for the request; it does not query SQLite by `track_id` for every vector result. Exact example showing what remains open: SQLite has A, B, C, D; relational retrieval returns A, B; vector retrieval returns B, C. A uses relational metadata. B keeps relational metadata and adds vector score/source. C uses vector metadata because it was absent from the retrieved relational candidates, even if C exists in SQLite. If C's vector metadata only has `track_id` or is missing display fields, the API can return a playlist row without title/artist. `track_id` is sufficient for identity, but not sufficient for display unless the API/front end hydrates by ID before rendering. `EmbeddingManager.track_metadata()` in `backend/data/embeddings.py` only adds `title`/`artist_name` to Chroma metadata `if value:`, so missing ingestion values can become omitted keys. `backend/api/routes/recommend.py`'s `_to_playlist_track()` still reads whatever is in `metadata` as-is. Genre keying (`seed_genre` vs. `genres_csv`) is already reconciled correctly by `_extract_genre()`.
- **Deprecated Groq models**: `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` no longer exist on Groq (confirmed via `client.models.list()`). `notebooks/pipeline_tests.ipynb` has been updated off these. `qwen/qwen3.6-27b` is available but is a reasoning model that wraps output in `<think>` tags by default, which breaks every agent's JSON extraction — avoid it unless called with `reasoning_effort="none"`. `openai/gpt-oss-120b` / `openai/gpt-oss-20b` are clean by default and currently the project's standard.

---

## Next Priorities

`docs/web_app_full_data_plan.md` Phase 0 is complete. Phase 1 (`POST /recommend`, error handling, schemas) is implemented, and the `GROQ_MODEL` default bug is fixed, but it's **not yet confirmed working end-to-end** — resume here:

1. **Verify `/recommend`** — actually hit it with a real HTTP request to confirm Phase 1 works, not just that it compiles
2. **Fix the title/artist_name gap** — backfill missing metadata for vector-only candidates via relational lookup by `track_id` in `_to_playlist_track()` (see Known Issues above)
3. **Build the Streamlit frontend** — `frontend/app.py`, calling `/recommend` over HTTP
4. **Wire `CriticAgent` into `run_playlist_graph()`** — Phase 5, deferred until the above is working end-to-end
5. **Expand tests** — `tests/test_api.py` real API tests, graph integration tests, critic behavior tests
