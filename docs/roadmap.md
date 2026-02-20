# EchoAgent Roadmap (v1)

## Final State (Must Hold)

"Agentic intent-driven playlist system using local semantic retrieval and constraint-aware ranking, with Spotify used only for playback hydration."

## Scope Guardrails

- Use local/open datasets for core retrieval corpus.
- Treat Spotify as optional link hydration only (not primary ingestion).
- Keep parser output contract fixed to: `semantic_query`, `hard_constraints`, `soft_preferences`, `exclusions`.
- Prioritize deterministic retrieval and ranking over tool-heavy orchestration.

## 6-Week Delivery Plan

### Week 1: Prompt -> VibeIntent Contract

Outcome: parser emits stable `VibeIntent` structure with hard/soft/exclusion separation.

- Add and align `docs/specs/vibe_intent.md`.
- Add and align `backend/agents/vibe_intent.py`.
- Update `backend/agents/prompt_parser.py` to emit `VibeIntent` consistently.
- Update `docs/specs/01_prompt_decomposition_spec.md` to remove routing language.
- Update `docs/specs/02_prompt_taxonomy.md` to map categories into hard vs soft vs exclusions.

### Week 2: Open Metadata Ingestion + TrackDocument Schema

Outcome: `TrackDocument`-first storage model with null-safe optional fields.

- Add `docs/specs/track_document.md`.
- Add `backend/data/ingest_open_dataset.py`.
- Refactor `backend/data/db.py` from Song/AudioFeature/Lyric/Album tables to TrackDocument-first schema.
- Defer `backend/data/ingest_spotify.py` and `backend/data/ingest_genius.py`.
- Load 5k-10k tracks from open datasets.

### Week 3: Generic Track-Text Embeddings + Hybrid Retrieval

Outcome: hybrid retrieval using SQL hard filters + vector similarity on track text.

- Refactor `backend/data/embeddings.py` from lyrics/art collections to generic track-text collections.
- Add retrieval validation notebook `notebooks/retrieval_sanity.ipynb`.
- Validate retrieval quality on representative prompts.

### Week 4: Pipeline Assembly With Validation + Reranking

Outcome: end-to-end retrieval pipeline with schema-safe parser output and constraint-aware reranking.

- Add `backend/agents/intent_adapter.py`.
- Add `backend/agents/reranker.py`.
- Add `backend/agents/prompts.json`.
- Update `backend/agents/prompt_parser.py` to strict JSON schema validation and retry repair loop.
- Keep `backend/agents/enrichment_agent.py` deferred.

### Week 5: API Integration (TrackDocument Response)

Outcome: API returns ranked `TrackDocument` responses with optional playback hydration.

- Implement `backend/api/main.py`.
- Implement `backend/api/schemas.py`.
- Implement `backend/api/routes/recommend.py`.
- Implement `backend/api/routes/health.py`.
- Implement `tests/test_api.py`.

### Week 6: UI + Documentation Alignment

Outcome: messaging and docs accurately reflect local retrieval architecture.

- Update `README.md` wording to local retrieval first, Spotify links optional.
- Update `docs/roadmap.md` completion state and remaining gaps.
- Verify final-state sentence appears in docs unchanged.

## GitHub TODOs (Copy Into Issues or Projects)

### Epic: Week 1 - Prompt -> VibeIntent

- [ ] `W1-01` Define `VibeIntent` schema and examples in `docs/specs/vibe_intent.md`.
- [ ] `W1-02` Implement `VibeIntent` normalize/validate contract in `backend/agents/vibe_intent.py`.
- [ ] `W1-03` Ensure `backend/agents/prompt_parser.py` always returns the 4-key contract.
- [ ] `W1-04` Remove parser-routing language from `docs/specs/01_prompt_decomposition_spec.md`.
- [ ] `W1-05` Map taxonomy categories to hard/soft/exclusion logic in `docs/specs/02_prompt_taxonomy.md`.

### Epic: Week 2 - TrackDocument + Open Data

- [ ] `W2-01` Author `TrackDocument` spec in `docs/specs/track_document.md`.
- [ ] `W2-02` Build open-data ingestion in `backend/data/ingest_open_dataset.py`.
- [ ] `W2-03` Refactor relational schema to TrackDocument-first in `backend/data/db.py`.
- [ ] `W2-04` Add null-safe ingestion paths for missing lyrics/audio features.
- [ ] `W2-05` Mark `backend/data/ingest_spotify.py` as deferred and keep it non-blocking.
- [ ] `W2-06` Mark `backend/data/ingest_genius.py` as deferred and keep it non-blocking.
- [ ] `W2-07` Ingest and validate 5k-10k tracks from open datasets.

### Epic: Week 3 - Embeddings + Retrieval

- [ ] `W3-01` Convert embedding pipeline to track text fields in `backend/data/embeddings.py`.
- [ ] `W3-02` Add SQL hard-filter pass before vector search.
- [ ] `W3-03` Add retrieval sanity notebook at `notebooks/retrieval_sanity.ipynb`.
- [ ] `W3-04` Define retrieval metrics and baseline prompt set for acceptance.

### Epic: Week 4 - Intent Adapter + Reranker

- [ ] `W4-01` Implement intent-to-query translation in `backend/agents/intent_adapter.py`.
- [ ] `W4-02` Implement constraint-aware ranking in `backend/agents/reranker.py`.
- [ ] `W4-03` Add parser prompt templates in `backend/agents/prompts.json`.
- [ ] `W4-04` Upgrade parser validation and repair loop in `backend/agents/prompt_parser.py`.
- [ ] `W4-05` Keep `backend/agents/enrichment_agent.py` deferred behind explicit flag.

### Epic: Week 5 - API

- [ ] `W5-01` Wire FastAPI app entrypoint in `backend/api/main.py`.
- [ ] `W5-02` Define request/response models in `backend/api/schemas.py`.
- [ ] `W5-03` Implement `POST /recommend` in `backend/api/routes/recommend.py`.
- [ ] `W5-04` Implement `GET /health` in `backend/api/routes/health.py`.
- [ ] `W5-05` Add API contract tests in `tests/test_api.py`.

### Epic: Week 6 - Docs + Positioning

- [ ] `W6-01` Update `README.md` to emphasize local retrieval and Spotify hydration only.
- [ ] `W6-02` Add architecture diagram updates if needed.
- [ ] `W6-03` Verify wording consistency across docs and API responses.

## File Goals (What Each File Is For)

| File | Goal |
| --- | --- |
| `backend/agents/prompt_parser.py` | Parse natural-language prompts into validated `VibeIntent` JSON. |
| `backend/agents/vibe_intent.py` | Define the intent contract object, normalization rules, and validation logic. |
| `backend/agents/playlist_ranker.py` | Assemble/rank final candidates using intent-aware scoring and diversity controls. |
| `backend/agents/enrichment_agent.py` | Deferred module for external enrichment when local recall is insufficient. |
| `backend/agents/intent_adapter.py` | Translate `VibeIntent` fields into SQL filters and vector query inputs. |
| `backend/agents/reranker.py` | Apply post-retrieval reranking using hard constraints and soft preference boosts. |
| `backend/agents/prompts.json` | Store parser prompt templates and parser output guidance text. |
| `backend/data/db.py` | Own relational schema and query helpers around `TrackDocument` storage. |
| `backend/data/embeddings.py` | Build/store/query track-text embeddings for semantic retrieval. |
| `backend/data/ingest_open_dataset.py` | Ingest open datasets and map rows into `TrackDocument` records. |
| `backend/data/ingest_spotify.py` | Deferred ingestion module; only used for optional playback-link hydration workflows. |
| `backend/data/ingest_genius.py` | Deferred ingestion module; not part of core retrieval path. |
| `backend/api/main.py` | FastAPI app bootstrap and route registration. |
| `backend/api/schemas.py` | API input/output schemas, especially `TrackDocument` response contract. |
| `backend/api/routes/recommend.py` | Recommendation endpoint that executes parser -> retrieval -> reranking pipeline. |
| `backend/api/routes/health.py` | Health and readiness endpoint for service checks. |
| `tests/test_api.py` | API contract and regression tests for recommendation and health endpoints. |
| `docs/specs/01_prompt_decomposition_spec.md` | Parser decomposition rules for converting prompt text into `VibeIntent`. |
| `docs/specs/02_prompt_taxonomy.md` | Taxonomy of prompt classes and their mapping into hard/soft/exclusion fields. |
| `docs/specs/vibe_intent.md` | Canonical schema and semantics for `VibeIntent`. |
| `docs/specs/track_document.md` | Canonical schema for retrieval units returned by the API. |
| `docs/roadmap.md` | Execution plan, milestones, and GitHub-tracking checklist. |
| `README.md` | Project overview and architecture narrative aligned to current scope. |
