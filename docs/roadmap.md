# EchoAgent Roadmap

Living status doc. Updated as work progresses. For deferred items see `docs/future.md`. For architectural decisions see `docs/architecture.md`.

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
- `PlaylistBuilderAgent`: model-guided selection/order, unique-track validation, and a candidate pool that expands to the requested size (up to the planner limit of 50); uses available unique tracks when retrieval is short
- `playlist_graph.py`: full LangGraph orchestration with parallel retrieval fan-out, critic routing, and retry loop
- `graph_state.py`: shared typed state schema (`PlaylistGraphState`)
- Relational DB: SQLAlchemy models, ingestion pipelines, audio features, tags
- Vector DB: ChromaDB collections, sentence-transformer embeddings, semantic search utilities
- `CriticAgent`: active in the graph runner; reviews playlists and sends feedback and the previous plan into replanning
- Final rejection handling: API returns HTTP 422 with `playlist_rejected`, reason, and retry count instead of returning rejected tracks
- JSON reliability and focused offline verification: 85 tests passed; real-graph API tests use scripted external I/O (see [testing guide](testing.md))

### Remaining / Deferred

- **Live acceptance**: the latest live requests failed at the planner and builder with Groq 429 quota errors, surfaced as API 502. A fully accepted live playlist on the latest code remains unverified.
- **Frontend**: not implemented; the next product milestone is Streamlit calling FastAPI over HTTP.
- **Deferred by agreement**: application-level rate-limit backoff and per-agent budgets; strict exclusion filtering; metadata enrichment and cleanup. These are not completed fixes or immediate prerequisites for frontend work.

### Phase 1 — Backend MVP: implemented and partially verified

The API, critic integration, bounded JSON repair, rejection handling, and offline graph/API integration tests are implemented. Startup and `/health` were verified. On 2026-09-23, the focused offline suite passed 85 tests. Latest live runs reached `plan` and `build_playlist`, respectively, but Groq rejected requests exceeding the account's reported 8,000 TPM allowance. JSON repair does not handle provider quota failures; the API currently maps them to HTTP 502. See [testing and troubleshooting](testing.md) for commands, evidence, and limits of verification.

- `main.py` uses a `lifespan` context manager to build `LLMClient`/`PlannerAgent`/`PlaylistBuilderAgent` once at startup (stored on `app.state`), reading `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_TEMPERATURE`, `GROQ_MAX_TOKENS`, `DATABASE_URL`, `CHROMA_PERSIST_DIRECTORY` from `.env` / the environment.
- `recommend.py` uses FastAPI `Depends()` reading those `app.state` singletons (so tests can override via `app.dependency_overrides`), builds a fresh `PromptParser` per request, and translates candidates to `PlaylistTrack` via `_to_playlist_track()` — including `_extract_genre()`, which correctly reconciles the `seed_genre` (relational) vs. `genres_csv` (vector) mismatch noted below.
- **Does not yet backfill missing `title`/`artist_name`** for vector-only candidates — see Known Issues, still open.
- LLM client: all four structured agents request JSON mode on every attempt. Provider JSON-validation failures, empty/truncated output, and schema errors enter bounded repairs; there is no plain-text retry. Provider errors such as 429 propagate separately. Builder validation exhaustion retains ranked-track fallback for critic review, exposed via `debug.builder_fallback_used` and `debug.builder_fallback_reason`. See `docs/web_app_full_data_plan.md` for offline JSON/graph test coverage and diagnostic logging.

---

## Known Issues

- There appear to be two copies of the relational DB: `backend/data/music_relational.db` and `database/music_relational.db`. Should be consolidated.
- **Vector-only candidates can be missing `title`/`artist_name`**: known issue, explicitly deferred by the user on 2026-09-23; move on with other work and revisit later. Deferred scope also includes normalizing byte-representation title/artist/album strings and malformed tags. This is unresolved, not a completed fix. `fuse_candidates()` processes retrieved relational candidates first, but it only checks the relational results returned for the request; it does not query SQLite by `track_id` for every vector result. Exact example showing what remains open: SQLite has A, B, C, D; relational retrieval returns A, B; vector retrieval returns B, C. A uses relational metadata. B keeps relational metadata and adds vector score/source. C uses vector metadata because it was absent from the retrieved relational candidates, even if C exists in SQLite. If C's vector metadata only has `track_id` or is missing display fields, the API can return a playlist row without title/artist. `track_id` is sufficient for identity, but not sufficient for display unless the API/front end hydrates by ID before rendering. `EmbeddingManager.track_metadata()` in `backend/data/embeddings.py` only adds `title`/`artist_name` to Chroma metadata `if value:`, so missing ingestion values can become omitted keys. `backend/api/routes/recommend.py`'s `_to_playlist_track()` still reads whatever is in `metadata` as-is. Genre keying (`seed_genre` vs. `genres_csv`) is already reconciled correctly by `_extract_genre()`.
- **Groq quota failures**: latest live runs with `GROQ_MAX_TOKENS=4096` exceeded the reported 8,000 TPM allowance (5,836 + 2,492 and 4,329 + 4,015). The API returns 502 for upstream 429. Restarting the server does not reset quota; rate-limit recovery remains deferred.
- **Strict exclusions**: scoring penalties and critic instructions do not guarantee deterministic exclusion filtering. Enforcement is deferred.
- **Model configuration**: the API default is `openai/gpt-oss-120b`, overridable with `GROQ_MODEL`. Model availability and account limits must be checked for the configured provider; no model choice guarantees valid output on every call.

---

## Next Priorities

1. **Phase 2: Streamlit frontend MVP** — `frontend/app.py` calling `/recommend`; prompt input, loading state, results, examples, and a debug panel. Handle structured 422 rejection details and string-valued provider errors.
2. **Complete live acceptance verification when quota permits** — use the testing guide; do not equate offline fixture coverage with a successful live run.
3. **Continue the full-dataset track** according to `web_app_full_data_plan.md`, without making full-data processing a prerequisite for the subset UI.

Critic wiring and offline graph/API tests are completed, not future tasks. Metadata enrichment, exclusion enforcement, and rate-limit improvements remain parked in [future work](future.md).
