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
- **API routes**: `main.py` (FastAPI app) and `routes/health.py` (`GET /health`) are implemented and verified working. `backend/api/schemas.py` defines the full `/recommend` request/response contract (`RecommendRequest`, `PlaylistTrack`, `CriticReport`, `RecommendDebug`, `RecommendResponse`). `routes/recommend.py` itself — actually calling `run_playlist_graph()` and translating the result — is still empty; this is Phase 1 of `docs/web_app_full_data_plan.md`.
- **`tests/test_api.py`**: placeholder, no real tests yet.
- **Frontend**: none exists yet. Decided approach: Streamlit calling the FastAPI backend over HTTP (not calling `run_playlist_graph()` directly), so a later React frontend is a drop-in swap. See `docs/web_app_full_data_plan.md`.

---

## Known Issues

- **`PromptParser.parse()` signature mismatch**: the implementation takes no arguments (uses `self.user_input` set at init), but `playlist_graph.py` calls `parser.parse(state["user_prompt"])` with an argument. Will surface as a `TypeError` during graph execution. Needs resolution before end-to-end testing.
- **`environment.yml` is incomplete**: missing `langgraph`, `transformers`, `torch`, `fastapi`, `uvicorn`, `pytest`, and the Anthropic client library. Needs updating.
- There appear to be two copies of the relational DB: `backend/data/music_relational.db` and `database/music_relational.db`. Should be consolidated.
- **Vector-only candidates can be missing `title`/`artist_name`**: `EmbeddingManager.track_metadata()` in `backend/data/embeddings.py` only adds `title`/`artist_name` to Chroma metadata `if value:` — if either was empty at ingestion time, the key is omitted entirely rather than stored as `None`. A candidate found only via vector retrieval (not present in the relational store) can therefore reach the final playlist with no title or artist to display. Fix should happen before the playlist leaves the backend — e.g. backfill missing fields via a relational lookup by `track_id` for vector-only candidates — rather than pushing the gap onto the API/frontend layer. Also note: genre is keyed differently per source (relational: `seed_genre`, vector metadata: `genres_csv`, pipe-joined) — any fix here should reconcile that too.

---

## Next Priorities

`docs/web_app_full_data_plan.md` Phase 0 (dev environment, backend/frontend startup commands, minimum product flow, API schema) is complete. Current priorities are Phase 1 of that plan:

1. **Wire `POST /recommend`** — implement `backend/api/routes/recommend.py`: call `run_playlist_graph()`, translate the result into a `RecommendResponse`, add error handling (parser failure, no candidates, vector DB unavailable, LLM failure)
2. **Fix the title/artist_name gap** — backfill missing metadata for vector-only candidates via relational lookup by `track_id` before the playlist leaves the backend (see Known Issues above)
3. **Build the Streamlit frontend** — `frontend/app.py`, calling `/recommend` over HTTP
4. **Wire `CriticAgent` into `run_playlist_graph()`** — Phase 5, deferred until the above is working end-to-end
5. **Expand tests** — `tests/test_api.py` real API tests, graph integration tests, critic behavior tests
