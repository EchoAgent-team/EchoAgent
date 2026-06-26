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

- **`critic_agent.py`**: always returns `accept: True`. LLM-backed critique not yet implemented.
- **API routes** (`main.py`, `recommend.py`, `health.py`): files exist but are not wired to the assembled graph.
- **`tests/test_api.py`**: placeholder, no real tests yet.

---

## Known Issues

- **`PromptParser.parse()` signature mismatch**: the implementation takes no arguments (uses `self.user_input` set at init), but `playlist_graph.py` calls `parser.parse(state["user_prompt"])` with an argument. Will surface as a `TypeError` during graph execution. Needs resolution before end-to-end testing.
- **`environment.yml` is incomplete**: missing `langgraph`, `transformers`, `torch`, `fastapi`, `uvicorn`, `pytest`, and the Anthropic client library. Needs updating.
- There appear to be two copies of the relational DB: `backend/data/music_relational.db` and `database/music_relational.db`. Should be consolidated.

---

## Next Priorities

1. **End-to-end graph test** — run `run_playlist_graph()` with real prompts via `langchain_tester.ipynb`, identify and fix integration bugs (start with the `PromptParser.parse()` issue above)
2. **Fix `environment.yml`** — add missing dependencies so the environment is reproducible
3. **Implement `critic_agent.py`** — LLM-backed structured review with `accept / reason / suggested_adjustments` output
4. **Wire API routes** — connect `run_playlist_graph()` to the `POST /recommend` endpoint
5. **Expand tests** — graph integration tests, API contract tests, critic behavior tests
