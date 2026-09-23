# Future Directions

Ideas and deferred features that are intentionally out of scope for the current MVP. These are parked here so they don't clutter the active roadmap or architecture docs.

## Playback Integration (post-MVP)

The `TrackDocument` schema already includes a `playback` field with optional `spotify_url` and `preview_url` entries, so the data contract is ready. The actual integration work is deferred.

- **Spotify**: Link generated playlists to real Spotify tracks via the Spotify API. Allow users to send the assembled playlist directly to their Spotify account.
- **Apple Music**: Same idea, different target.
- `backend/data/ingest_spotify.py` and `ingest_genius.py` both exist as deferred ingestion modules. Spotify is kept optional (playback hydration only, not a primary retrieval source).

## User Feedback Loop (post-MVP)

Allow users to react to generated playlists (thumbs up/down per track, regenerate with adjusted preferences) and feed that signal back into future retrievals. This implies some form of session or user state storage, which is out of scope for the local-first MVP.

## Optional Final Selector Agent

An LLM agent that takes the top 30–50 deterministically ranked candidates and chooses/orders the final 15–20 tracks. This would sit between `rerank` and `build_playlist` in the graph. Currently `PlaylistBuilderAgent` handles this, but a dedicated selector with explicit playlist-curation reasoning could improve quality. Worth revisiting after the pipeline is stable and tested.

## Enrichment Agent

`backend/agents/enrichment_agent.py` is a planned (but deferred) module for external enrichment when local recall is too low. For example, if a prompt retrieves very few matching candidates, the enrichment agent could query external APIs to augment the local corpus before reranking. Not part of v1.

## Database Migration

The current relational store uses SQLite. The SQLAlchemy models are written to be compatible with PostgreSQL. Migrating is a deployment-time decision, not a code rewrite.

## UI / Interactive Demo — active next milestone

The Streamlit MVP is Phase 2 of [the web-app plan](web_app_full_data_plan.md), not a deferred feature. It will call `/recommend` over HTTP. Playback, export, and a later React UI remain separate future work.

## End-to-End Playback: Prompt → Hear Music

Full user-facing flow: user types a prompt in a browser, EchoAgent builds a playlist, and music plays — all in one session.

**Architecture:**

```
Browser (prompt input + Spotify iframe player)
    │ POST /recommend
    ▼
FastAPI backend
    ├── /recommend →  LangGraph pipeline (existing playlist_graph.py)
    │                     ↓ playlist (title + artist metadata)
    └── /spotify   →  Spotify module (new)
                          1. Search each track by title + artist → Spotify URI
                          2. Create playlist in user's Spotify account
                          3. Return playlist URL/URI to frontend
                               ↓
                     Embedded Spotify iframe player in browser
```

The playback routes below are proposals; `/recommend` already exists and returns playlist metadata, not playback.

**Layers to build:**

| Module | Purpose |
|--------|---------|
| `backend/spotify/resolver.py` | Search Spotify API for title+artist → Spotify URI |
| `backend/spotify/player.py` | Create playlist in user account, optionally start playback |
| `backend/api/main.py` | FastAPI endpoints: `POST /recommend`, `POST /spotify/create` |
| `frontend/index.html` | Single-page UI: prompt input + embedded Spotify iframe player |

**Playback behavior by tier:**
- Spotify Free: playlist created in account; iframe shows 30-second previews
- Spotify Premium: full songs play in iframe or on any active Spotify device

**Suggested build order:**
1. `play.py` — single CLI script: run pipeline → resolve tracks → open Spotify playlist in browser
2. FastAPI endpoints wrapping the same flow
3. Minimal HTML frontend with embedded iframe player

**Notes:**
- Track resolution uses title + artist search (no ID mapping needed with current MSD data)
- The `TrackDocument` schema already has a `playback.spotify_url` field ready for this
- `backend/data/ingest_spotify.py` exists as a deferred module; Spotify is playback-only here, not a retrieval source
- Requires a Spotify Developer app (`client_id` + `client_secret`) and OAuth for playlist creation
- Full playback control (start/pause/skip from code) requires Spotify Premium

## Verification Still Open

API contract tests and real-graph fixture tests, including critic acceptance/rejection, feedback, bounded retries, and JSON repairs, are implemented. The focused suite passed 85 tests; see [testing guide](testing.md).

Still open:

- Successful live Groq/database acceptance on the latest code (latest runs hit provider quota).
- Retrieval correctness baselines using `docs/specs/02_prompt_taxonomy.md`.
- Full-dataset coverage and latency evaluation.

## Explicitly Deferred Backend Work

User decision reaffirmed 2026-09-23; do not treat these as prerequisites for frontend development:

- **Groq quota handling:** application-level bounded 429 backoff and smaller per-agent token budgets. Current upstream 429 failures become API 502. JSON repairs do not solve token-per-minute limits.
- **Strict exclusions:** deterministic removal of excluded artists/genres. Existing score penalties and critic instructions are not a guarantee.
- **Metadata enrichment and cleanup:** look up missing vector-only track metadata by `track_id` in the relational DB, and normalize byte-representation strings and malformed tags. Previously documented and intentionally postponed; it is unresolved, not fixed. This local-data repair is separate from the optional external enrichment agent above.
