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
- JSON reliability and focused offline verification: 89 tests passed (85 before the 2026-09-30 rate-limit tests); real-graph API tests use scripted external I/O (see [testing guide](testing.md))

### Remaining / Deferred

- **Live acceptance**: as of 2026-09-30 rate limiting and JSON failures no longer block a live run (no 429s, no repairs; see [architecture](architecture.md#rate-limits-and-model-allocation)), but the critic rejected all three passes for the test prompt, so a fully accepted live playlist is still unverified. The blocker has shifted from quota to retrieval/ranking quality.
- **Frontend**: Phase 2 MVP built 2026-09-28 (`frontend/app.py`, `.streamlit/config.toml`) — prompt input, example-vibe chips, track cards with match-strength bars, debug drawer, native theme. Calls FastAPI over HTTP via `FRONTEND_BACKEND_URL`. Not yet exercised against a real accepted `/recommend` response; only checked with headless `AppTest` smoke runs and mocked data.
- **Done 2026-09-30**: bounded 429 retry in `LLMClient` (API now returns 429, not 502), token-shrinking changes, and a light/heavy model split. **Still deferred by agreement**: Groq Dev Tier upgrade, full per-agent token budgets, strict exclusion filtering, metadata enrichment and cleanup.

### Phase 1 — Backend MVP: implemented and partially verified

The API, critic integration, bounded JSON repair, rejection handling, and offline graph/API integration tests are implemented. Startup and `/health` were verified. On 2026-09-23, the focused offline suite passed 85 tests. Latest live runs reached `plan` and `build_playlist`, respectively, but Groq rejected requests exceeding the account's reported 8,000 TPM allowance. JSON repair does not handle provider quota failures; the API currently maps them to HTTP 502. See [testing and troubleshooting](testing.md) for commands, evidence, and limits of verification.

- `main.py` uses a `lifespan` context manager to build two `LLMClient`s at startup (light: `GROQ_MODEL_LIGHT`, default `openai/gpt-oss-20b`, for parser + planner; heavy: `GROQ_MODEL`, default `openai/gpt-oss-120b`, for builder + critic) plus `PlannerAgent`/`PlaylistBuilderAgent`/`CriticAgent`, stored on `app.state`. It also reads `GROQ_API_KEY`, `GROQ_TEMPERATURE`, `GROQ_MAX_TOKENS`, `GROQ_REASONING_EFFORT` (default `low`), `LOG_LEVEL` (default `INFO`), `DATABASE_URL`, `CHROMA_PERSIST_DIRECTORY`. The route now also depends on `get_critic_agent` and passes the critic into `run_playlist_graph()` explicitly.
- `recommend.py` uses FastAPI `Depends()` reading those `app.state` singletons (so tests can override via `app.dependency_overrides`), builds a fresh `PromptParser` per request, and translates candidates to `PlaylistTrack` via `_to_playlist_track()` — including `_extract_genre()`, which correctly reconciles the `seed_genre` (relational) vs. `genres_csv` (vector) mismatch noted below.
- **Does not yet backfill missing `title`/`artist_name`** for vector-only candidates — see Known Issues, still open.
- LLM client: all four structured agents request JSON mode on every attempt. Provider JSON-validation failures, empty/truncated output, and schema errors enter bounded repairs; there is no plain-text retry. Provider errors such as 429 propagate separately. Builder validation exhaustion retains ranked-track fallback for critic review, exposed via `debug.builder_fallback_used` and `debug.builder_fallback_reason`. See `docs/web_app_full_data_plan.md` for offline JSON/graph test coverage and diagnostic logging.

---

## Known Issues

- There appear to be two copies of the relational DB: `backend/data/music_relational.db` and `database/music_relational.db`. Should be consolidated.
- **Vector-only candidates can be missing `title`/`artist_name`**: known issue, explicitly deferred by the user on 2026-09-23; move on with other work and revisit later. Deferred scope also includes normalizing byte-representation title/artist/album strings and malformed tags. This is unresolved, not a completed fix. `fuse_candidates()` processes retrieved relational candidates first, but it only checks the relational results returned for the request; it does not query SQLite by `track_id` for every vector result. Exact example showing what remains open: SQLite has A, B, C, D; relational retrieval returns A, B; vector retrieval returns B, C. A uses relational metadata. B keeps relational metadata and adds vector score/source. C uses vector metadata because it was absent from the retrieved relational candidates, even if C exists in SQLite. If C's vector metadata only has `track_id` or is missing display fields, the API can return a playlist row without title/artist. `track_id` is sufficient for identity, but not sufficient for display unless the API/front end hydrates by ID before rendering. `EmbeddingManager.track_metadata()` in `backend/data/embeddings.py` only adds `title`/`artist_name` to Chroma metadata `if value:`, so missing ingestion values can become omitted keys. `backend/api/routes/recommend.py`'s `_to_playlist_track()` still reads whatever is in `metadata` as-is. Genre keying (`seed_genre` vs. `genres_csv`) is already reconciled correctly by `_extract_genre()`.
- **Groq free-tier quota (mitigated, not removed)**: every free chat model has 8K TPM and 200K tokens/day. The 2026-09-30 changes cut a request to roughly 13k tokens total (~8.5k on the 120b bucket), which is still near the per-minute limit and allows only ~20-25 requests/day per model. Bounded 429 retry is in place; sustained use needs the Dev Tier (postponed) or further trimming.
- **Strict exclusions**: scoring penalties and critic instructions do not guarantee deterministic exclusion filtering. Enforcement is deferred.
- **Model configuration**: parser/planner default to `openai/gpt-oss-20b` (`GROQ_MODEL_LIGHT`); builder/critic default to `openai/gpt-oss-120b` (`GROQ_MODEL`). The per-model TPM bucket assumption is unconfirmed, and 20b's schema-following and judgment quality are untested at scale. `qwen/qwen3.8-27b` (docs say 3.8; CLAUDE.md says 3.6) still breaks JSON extraction with thinking output.
- **Temporary testing caps (remove later)**: `POOL_SIZE = 5` in `playlist_builder.py` and `TESTING_PLAYLIST_SIZE_CAP = 5` in `planner_agent.py` keep requests inside the free tier; playlists are at most 5 tracks until these are lifted.
- **Critic rejects the test prompt three times**: "late-night rainy city drive, no metal music" ended in `playlist_rejected` (tracks unrelated, low energy, inappropriate genres). Unknown whether replanning changes retrieval meaningfully or whether a 5-track pool is too small for a fair review. Next step is to dump the tracks and critic report for that run.
- **`echoagent-env` can drift below `environment.yml`'s floor versions.** Hit 2026-09-28: a stale `streamlit==1.8.0` (floor is `>=1.36.0`) crashed on import with a protobuf `TypeError: Descriptors cannot be created directly`. Fixing it (`pip install --upgrade streamlit`) pulled in `starlette>=0.46.0`, which then broke the then-installed `fastapi==0.115.9` (needs `starlette<0.46.0`) — `APIRouter()` failed with `unexpected keyword argument 'on_startup'`. Resolved by upgrading `fastapi` too (→0.141.1). If a teammate hits the protobuf error, check `pip check` for a starlette/fastapi mismatch before assuming it's protobuf alone.

---

## Next Priorities

1. **Diagnose the critic rejection** for the rainy-drive prompt (dump the 5 tracks and full critic report; check whether replans change retrieval), then complete live acceptance verification — use the testing guide; do not equate offline fixture coverage with a successful live run. This is the shared blocker for confirming both Phase 1 and the Phase 2 frontend actually work end-to-end.
2. **Run the Phase 2 frontend against a live backend** — `frontend/app.py` has a working first build (prompt input, examples, results, debug panel; structured 422 and string-valued errors are already handled), but it has never rendered a real accepted playlist. Start `uvicorn` + `streamlit run frontend/app.py` together and click through it once quota allows.
3. **Continue the full-dataset track** according to `web_app_full_data_plan.md`, without making full-data processing a prerequisite for the subset UI.

Critic wiring and offline graph/API tests are completed, not future tasks. Metadata enrichment, exclusion enforcement, and rate-limit improvements remain parked in [future work](future.md).
