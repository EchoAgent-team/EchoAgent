# EchoAgent Web App And Full Dataset Plan

This plan assumes a two-person team pair-programming with coding agents, with limited frontend experience. The goal is to build a polished web demo first, while processing the full Million Song Dataset into compact runtime artifacts for the recommendation pipeline.

## High-Level Recommendation

Build a responsive web app / PWA first, not native Android or iOS.

Recommended stack:

- Backend: FastAPI + existing LangGraph pipeline
- Frontend (MVP): Streamlit, calling the FastAPI backend over HTTP (not calling `run_playlist_graph()` directly) — same client-agnostic `/recommend` contract a future React frontend would use. Fast to build given limited frontend experience, and defers the fully-custom UI work without requiring any backend changes later.
- Frontend (post-MVP, deferred): Next.js or Vite React + Tailwind CSS + shadcn/ui, swapped in once the product is validated. Only the frontend changes — FastAPI/`schemas.py` stay as-is since Streamlit already exercises the same API contract.
- Hosting: Vercel for frontend (once on React); Render, Fly.io, Railway, or a small VM for backend
- Data: SQLite locally at first, then Postgres if needed; Chroma, Qdrant, or FAISS for vectors

Native mobile should wait until there is evidence that users want the product enough to justify app-store workflows, device testing, and mobile-specific UI work.

## Execution Strategy

Run two parallel tracks:

- Product track: build the backend API and web app against the current subset.
- Data track: process the full MSD-derived dataset into final relational tables and embeddings.

Do not wait for the full 1M-song processing pipeline before building the app. The app should work on the current subset first, then swap to the full processed dataset later through configuration.

## Phase 0: Project Reset And Scope Lock

**Status: Complete.**

Goal: make the repo runnable and define the app contract.

Tasks:

- [x] Fix `environment.yml` / dependency setup — see `environment.yml`.
- [x] Decide one local command for backend startup — `uvicorn backend.api.main:app --reload --port 8000`. Verified working: `GET /health` returns `{"status": "ok"}`.
- [x] Decide one local command for frontend startup — `streamlit run frontend/app.py` (not yet built). Streamlit will call the FastAPI backend over HTTP rather than calling `run_playlist_graph()` directly, so a later React frontend is a drop-in swap with no backend changes — see "High-Level Recommendation" above.
- [x] Define the minimum product flow (confirmed):
  1. User enters a vibe prompt in the frontend.
  2. Frontend calls `POST /recommend` with `{"prompt": "..."}`.
  3. Backend runs `run_playlist_graph()` and translates the result into a `RecommendResponse` (see `backend/api/schemas.py`).
  4. UI shows the playlist: title, artist, album/year, genre/tags, score. **No per-track "why"** — corrected from the original sketch below. `PlaylistBuilderAgent` only produces a playlist-level `rationale`/`energy_arc`, not a per-track explanation, so there is nothing to show per track beyond score.
  5. Optional debug panel shows parsed intent, planner weights, retrieval counts, and critic result (critic result will be `null` until Phase 5 wires `CriticAgent` into the graph runner).
- [x] Decide what counts as MVP done (confirmed):
  - 10–20 track playlist (matches `PlaylistPlan.playlist_size`, default 20, planner range 5–50)
  - No playback required
  - Spotify export/linking deferred (Phase 7)
  - Acceptable demo latency — no fixed number pinned; to be judged once Streamlit is actually calling `/recommend`.

Deliverables:

- [x] Working local dev environment
- [x] `.env.example`
- [x] Updated README setup instructions
- [x] Agreed API request/response schema — `backend/api/schemas.py`

## Phase 1: Backend MVP

**Status: Implemented and partially verified; live `/recommend` still has an LLM-output blocker.** See `docs/roadmap.md` Known Issues / Next Priorities for the exact blocker.

Goal: expose the existing LangGraph pipeline through a clean API.

Tasks:

- [x] Create FastAPI app in `backend/api/main.py` — `lifespan` context manager builds `LLMClient`/`PlannerAgent`/`PlaylistBuilderAgent` once at startup onto `app.state`, reading `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_TEMPERATURE`, `GROQ_MAX_TOKENS`, `DATABASE_URL`, `CHROMA_PERSIST_DIRECTORY` from `.env` / the environment.
- [x] Add `GET /health` — verified locally with `{"status":"ok"}`.
- [x] Add `POST /recommend` — `backend/api/routes/recommend.py`, via FastAPI `Depends()` reading the `app.state` singletons (so tests can override them without touching real Groq/DB).
- [x] Add request/response models in `backend/api/schemas.py`.
- [x] Wrap `run_playlist_graph()` behind the recommendation endpoint — builds a fresh `PromptParser` per request (reuses the shared `PlannerAgent`/`PlaylistBuilderAgent`).
- [x] Add error handling for parser failure, no candidates, vector DB unavailable, LLM failure — `_classify_error()` maps exception types/messages to the right HTTP status (422/503/502/500), empty playlist → 404.
- [x] Return both user-facing and debug fields — `_to_playlist_track()` + `_build_debug()`.

Final critic rejection: a non-empty playlist rejected after the allowed attempts returns HTTP `422` with `detail.code = "playlist_rejected"`, a user-facing `message`, the critic's `reason`, and `retry_count`. Rejected tracks are not returned. An empty playlist retains HTTP `404`. Accepted playlists retain HTTP `200`. Clients should handle both string-valued error details and this structured rejection detail.

API response shape: defined in `backend/api/schemas.py` (`RecommendResponse`) — that file is the canonical contract, not this doc. No per-track `"why"` field (no code path produces one). `_extract_genre()` in `recommend.py` correctly reconciles `seed_genre` (relational) vs. `genres_csv` (vector) into one `genre` field.

Open issue: `_to_playlist_track()` does **not** yet backfill missing `title`/`artist_name` for vector-only candidates. `fuse_candidates()` processes retrieved relational candidates first, but it checks only the relational candidate list returned for that request; it does not query SQLite by `track_id` for every vector result. Exact example showing what remains open: SQLite has A, B, C, D; relational retrieval returns A, B; vector retrieval returns B, C. A uses relational metadata. B keeps relational metadata and adds vector score/source. C uses vector metadata because it was absent from the retrieved relational candidates, even if C exists in SQLite. If C's vector metadata only has `track_id` or is missing display fields, the API can return a playlist row without title/artist. This is open for later; `track_id` is sufficient for identity, but not sufficient for display unless the API/front end hydrates by ID before rendering.

Startup/subset configuration is fixed for the current repo layout: `.env` and `.env.example` point to `DATABASE_URL=sqlite:///database/music_relational.db` and `CHROMA_PERSIST_DIRECTORY=database/chroma_db`, matching the files that exist under `echoagent/database/`. `main.py` now explicitly loads the repo `.env` file. Verified in `echoagent-env`: `GROQ_API_KEY` is present, both dataset paths resolve, and `backend.api.main:app` imports cleanly.

The `GROQ_MODEL` default bug (was `"gpt-oss-120b"`, an invalid Groq model ID) is fixed — `main.py` now defaults to `"openai/gpt-oss-120b"`. `/recommend` has now been exercised via a real HTTP request: the request reaches the LangGraph pipeline and loads data, but live completion is still blocked by the final LLM playlist-selection step returning/triggering invalid JSON behavior. A fallback to ranked candidates was added in `build_playlist_node()` so this should be re-tested after restarting Uvicorn.

Deliverables:

- [ ] `POST /recommend` works locally — implemented and exercised via curl; currently reaches the graph but still needs a clean successful response after the playlist-builder JSON/fallback changes are re-tested.
- [x] `tests/test_api.py` has real API tests — covers `/health`, `/recommend` success mapping/debug output, empty playlist `404`, prompt parse `422`, and request validation `422` with graph/LLM dependencies mocked.
- [x] Backend can start with the current subset config — `.env` points to `database/music_relational.db` and `database/chroma_db`; `/health` passes and the live `/recommend` request reaches the graph/data-loading path. Full successful `/recommend` response remains open under the deliverable above.

### JSON reliability and offline integration verification

- Parser, planner, builder, and critic request JSON mode on every attempt. Provider JSON-validation failures, empty content, truncation, malformed JSON, and invalid fields enter the agent's bounded repair loop (three attempts by default). JSON mode is never disabled for a repair.
- Provider errors such as HTTP 429 propagate through the existing API error mapping; they do not trigger JSON repairs or the builder's ranked-track fallback. Rate-limit backoff and per-agent budget changes remain deferred.
- Builder validation exhaustion retains the ranked-track fallback for critic review. `debug.builder_fallback_used` and `debug.builder_fallback_reason` describe the final builder pass and reset when a later pass succeeds.
- INFO logs from `backend.agents.json_output` include agent, attempt, model, finish reason, and available token usage. Repair warnings include agent, attempt, and error type. Configure that logger at INFO to inspect completion diagnostics. Raw prompts and responses are not logged by these diagnostics.
- `tests/test_json_reliability.py` exercises the real LLM client response handling and the API-to-LangGraph path, scripting only external model transport and retrieval data. It covers acceptance, feedback-driven replanning, final rejection, empty results, bounded retries, and fallback visibility.
- Run offline checks with `python -m pytest tests/test_json_reliability.py tests/test_api.py tests/test_playlist_builder.py tests/test_candidate_fuser.py tests/test_reranker.py -q`.
- Live Groq/database smoke testing remains a separate, pending step. Exclusion enforcement and metadata enrichment remain deferred.

## Phase 2: Frontend MVP

Goal: build a simple but polished web UI.

Screens:

- Main recommendation screen:
  - prompt input
  - generate button
  - loading state
  - playlist result list
- Track result cards/table:
  - title
  - artist
  - album/year
  - genre/tags
  - score or match badge
- Debug drawer:
  - parsed intent
  - planner weights
  - retrieval counts
  - critic result
- Saved examples:
  - "late-night rainy city drive"
  - "warm nostalgic indie autumn walk"
  - "high-energy gym rhythm, no sad songs"

Frontend guidance:

- Keep the interface closer to a tool/dashboard than a flashy landing page.
- Avoid complex animations.
- Prioritize clarity: prompt in, playlist out.
- Make it responsive for mobile browsers.
- Keep the first version as one primary screen with a debug drawer.

Deliverables:

- Usable web UI
- Mobile-friendly layout
- Demo prompts
- Frontend talks to local backend

## Phase 3: Full MSD Processing Pipeline

Goal: process the full dataset into only the data EchoAgent actually needs.

The app should consume final artifacts, not raw MSD files or temporary ingestion outputs.

Runtime artifacts to keep:

- Relational database:
  - `tracks`
  - `artists`
  - `albums`
  - `audio_features`
  - compact `lyrics` or retrieval text fields
  - tags / seed genres
  - source IDs
- Vector store:
  - one embedding per searchable track document
  - metadata needed for retrieval and hydration:
    - `track_id`
    - title
    - artist
    - genre/tags
    - source IDs

Artifacts not needed in final runtime storage:

- raw MSD HDF5 files
- raw MusicBrainz dump
- duplicate MXM text files
- intermediate joins
- temporary CSVs/parquets
- full verbose documents if compact fields are enough

Pipeline stages:

1. Raw ingest
   - read MSD metadata/audio features
   - read Last.fm/tagtraum tags
   - read MXM BoW matches
2. Normalize IDs
   - canonical `track_id`
   - map external IDs
   - dedupe tracks/artists/albums
3. Build relational DB
   - compact SQLite/Postgres schema
   - indexes on genre, year, energy, tempo, danceability, artist
4. Build retrieval text
   - v1: BoW-derived text where available
   - fallback: title + artist terms + tags + seed genre for tracks without BoW
5. Generate embeddings
   - use `all-MiniLM-L6-v2` initially
   - 384-dimensional vectors
   - batch processing
   - checkpoint progress every N tracks
6. Build vector index
   - Chroma initially if easiest
   - consider Qdrant or FAISS later for production
7. Validate
   - row counts
   - missing-field report
   - sample retrieval tests
   - storage report

Deliverables:

- `database/music_relational_full.db`
- `database/chroma_full/` or equivalent vector store
- Processing report with:
  - number of tracks
  - number with BoW
  - number with tags
  - number embedded
  - final storage size

## Phase 4: Swap App To Full Dataset

Goal: make the web app work with the full processed dataset.

Tasks:

- Add config flags:
  - `DATABASE_URL`
  - `CHROMA_PERSIST_DIRECTORY`
  - `EMBEDDING_MODEL`
- Run backend against full DB locally.
- Test prompts across genres/eras.
- Add retrieval fallback behavior:
  - if relational retrieval is too narrow, broaden constraints
  - if vector results are weak, increase top-k
  - if few BoW embeddings exist, use metadata/tag embeddings
- Profile latency.

Deliverables:

- App works with full dataset.
- Full-data smoke test suite.
- Latency notes.

## Phase 5: Critic And Quality Layer

Goal: make the agentic layer visible and useful.

Tasks:

- Wire `CriticAgent` into graph state.
- Let critic reject playlists for:
  - exclusion violations
  - repeated artists
  - genre collapse
  - mismatch with prompt
- Feed critic suggestions back into planner.
- Add deterministic guardrails before/after critic:
  - no excluded artists
  - no duplicate tracks
  - max artist repeats
- Show critic result in frontend debug drawer.

Deliverables:

- Critic participates in graph execution.
- Critic tests.
- Visible quality trace in UI.

## Phase 6: Deployment

Goal: public demo.

Recommended deployment:

- Frontend: Vercel
- Backend: Render, Fly.io, Railway, or a small VM
- Data:
  - subset demo: SQLite + Chroma on disk
  - full demo: persistent volume or hosted VM
  - production-ish: Postgres + Qdrant

Tasks:

- Containerize backend.
- Add production env vars.
- Add CORS.
- Add basic rate limiting.
- Add logging.
- Add a demo mode with canned prompts if LLM quota fails.

Deliverables:

- Public URL
- Stable demo
- README with architecture diagram
- Portfolio-ready screenshots

## Phase 7: Optional Spotify Export

Goal: useful integration without making Spotify the core product.

Tasks:

- Resolve generated tracks to Spotify search results by title + artist.
- Return Spotify links.
- Later, create playlist in the user's Spotify account via OAuth.

Important caution: do not build the core product around streaming playback or paid Spotify-dependent access until platform rules are reviewed carefully.

## Storage Estimate

The local repo currently shows this pattern:

- Current app-facing `database/` folder: about 18 MB
- Current raw/working `data/` folder: about 9.7 GB
- Current relational DB: about 9.9 MB for 10,000 tracks
- Current Chroma DB: about 7.6 MB for 2,350 embeddings

This gap is expected. The final deployed runtime dataset can be much smaller than the raw processing workspace if only relational tables and vector indexes are retained.

### Scenario A: Embed Only MXM / Lyrics-Matched Tracks

If embeddings are generated only for tracks with MXM/lyrics-derived text, the embedded subset will likely be a few hundred thousand tracks rather than the full 1M.

Estimated final storage:

- Relational DB for 1M tracks: 1-3 GB
- Vector DB for roughly 200k-300k tracks: 1-2 GB
- Indexes/metadata overhead: 0.5-1 GB

Estimated total: 3-6 GB

### Scenario B: Embed All 1M Tracks With Compact Metadata/Tag Retrieval Text

Using `all-MiniLM-L6-v2`, which produces 384-dimensional vectors:

- Relational DB: 1-3 GB
- Raw float32 vectors alone: about 1.5 GB
- Vector index + Chroma/Qdrant metadata overhead: likely 4-7 GB
- Extra indexes/cache: 1-2 GB

Estimated total: 6-12 GB

### Scenario C: Larger 768-Dimensional Embeddings

If a larger embedding model is used, vector storage approximately doubles.

Estimated total: 10-18 GB

### Practical Budget

Recommended final app-facing dataset budget:

- Minimum comfortable target: 10-15 GB
- Safer allocation: 25 GB

Recommended processing workspace:

- Raw MSD plus side datasets and temporary artifacts can require much more space.
- Plan for 300-500 GB free workspace during full processing.
- The final deployed artifact should be much smaller after cleanup.

## Suggested Milestone Order

1. Backend API on current subset.
2. Web UI on current subset.
3. Full-data processing pipeline.
4. Swap app to full dataset.
5. Critic + quality loop.
6. Deploy public demo.
7. Spotify export / monetization experiments.

This ordering gives the team a demoable product early while the heavier data work runs in parallel.
