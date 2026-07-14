# EchoAgent Web App And Full Dataset Plan

This plan assumes a two-person team pair-programming with coding agents, with limited frontend experience. The goal is to build a polished web demo first, while processing the full Million Song Dataset into compact runtime artifacts for the recommendation pipeline.

## High-Level Recommendation

Build a responsive web app / PWA first, not native Android or iOS.

Recommended stack:

- Backend: FastAPI + existing LangGraph pipeline
- Frontend: Next.js or Vite React
- UI: Tailwind CSS + shadcn/ui
- Hosting: Vercel for frontend; Render, Fly.io, Railway, or a small VM for backend
- Data: SQLite locally at first, then Postgres if needed; Chroma, Qdrant, or FAISS for vectors

Native mobile should wait until there is evidence that users want the product enough to justify app-store workflows, device testing, and mobile-specific UI work.

## Execution Strategy

Run two parallel tracks:

- Product track: build the backend API and web app against the current subset.
- Data track: process the full MSD-derived dataset into final relational tables and embeddings.

Do not wait for the full 1M-song processing pipeline before building the app. The app should work on the current subset first, then swap to the full processed dataset later through configuration.

## Phase 0: Project Reset And Scope Lock

Goal: make the repo runnable and define the app contract.

Tasks:

- Fix `environment.yml` / dependency setup.
- Decide one local command for backend startup.
- Decide one local command for frontend startup.
- Define the minimum product flow:
  - user enters vibe prompt
  - backend returns playlist
  - UI shows tracks, artists, metadata, score/reason
  - optional debug panel shows parsed intent, plan, retrieval counts
- Decide what counts as MVP done:
  - 10 to 20 track playlist
  - acceptable demo latency
  - no playback required
  - Spotify export/linking can be deferred

Deliverables:

- Working local dev environment
- `.env.example`
- Updated README setup instructions
- Agreed API request/response schema

## Phase 1: Backend MVP

Goal: expose the existing LangGraph pipeline through a clean API.

Tasks:

- Create FastAPI app in `backend/api/main.py`.
- Add `GET /health`.
- Add `POST /recommend`.
- Add request/response models in `backend/api/schemas.py`.
- Wrap `run_playlist_graph()` behind the recommendation endpoint.
- Add error handling for:
  - parser failure
  - no candidates
  - vector DB unavailable
  - LLM failure
- Return both user-facing and debug fields.

Suggested API response shape:

```json
{
  "prompt": "late-night rainy city drive",
  "playlist": [
    {
      "track_id": "TRABC123",
      "title": "Example Track",
      "artist_name": "Example Artist",
      "album_title": "Example Album",
      "year": 2007,
      "genre": "indie",
      "score": 0.82,
      "why": "Matches the low-energy rainy-city vibe."
    }
  ],
  "debug": {
    "intent": {},
    "plan": {},
    "relational_candidate_count": 100,
    "vector_candidate_count": 100,
    "fused_candidate_count": 143
  }
}
```

Deliverables:

- `POST /recommend` works locally.
- `tests/test_api.py` has real API tests.
- Backend can run with the current subset data.

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
