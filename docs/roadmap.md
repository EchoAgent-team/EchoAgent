# Spotify Vibe Agent — Project Roadmap

**Goal:**  
Build an agentic, multimodal playlist recommendation system that converts free‑form “vibe” prompts into structured attributes and retrieves songs using audio, text, image, and cultural metadata.

**Timeline:** ~6 weeks (≈1.5 months)  
**Team:** 2 engineers (shared ownership)

---

## Week 0 — Project Setup (Lightweight)
**Outcome:** Repo is ready; scope is aligned.

- Finalize project scope and constraints
- Decide core stack (Python, FastAPI, vector DB, SQL DB)
- Create repo structure
- Set up environment, linting, and basic CI (optional)

---

## Week 1 — Prompt → Attribute System Design
**Outcome:** Clear specification for how natural language becomes machine‑usable signals.

- Define supported prompt types (mood, visual, lyrical, structural, hybrid)
- Design attribute taxonomy (mood, energy, instruments, themes, visuals, era, constraints)
- Create prompt → attribute mapping spec (design document)
- Define attribute routing (which attributes query which data sources)
- Build a **rule‑based prompt parser prototype**
- Write example prompts and expected structured outputs

**Deliverables**
- `docs/prompt_to_attribute_spec.md`
- `backend/agents/prompt_parser.py` (rule‑based stub)

---

## Week 2 — Data Acquisition & Schema Implementation
**Outcome:** A clean, queryable multimodal music dataset.

- Implement relational schema:
  - Songs
  - AudioFeatures
  - Lyrics
  - AlbumArt
  - Tags
- Set up SQL database (Postgres / SQLite for dev)
- Spotify API ingestion (metadata + audio features)
- Lyrics ingestion (e.g. Genius scraping/API)
- Tag ingestion (e.g. Last.fm or similar)
- Store album art URLs or images

**Deliverables**
- `backend/data/ingest_spotify.py`
- `backend/data/ingest_genius.py`
- `backend/data/db.py`
- Populated DB with ~1–2k diverse tracks

---

## Week 3 — Embeddings & Retrieval Foundations
**Outcome:** Semantic similarity search across modalities.

- Generate text embeddings for lyrics
- Generate image embeddings for album art
- Store embeddings in vector DB (Chroma / FAISS / Pinecone)
- Implement similarity search utilities
- Prototype cross‑modal retrieval logic in notebooks

**Deliverables**
- `backend/data/embeddings.py`
- Vector DB populated
- `notebooks/prototype_similarity.ipynb`

---

## Week 4 — Agentic Pipeline Assembly
**Outcome:** End‑to‑end reasoning and retrieval pipeline.

- Replace rule‑based parser with **LLM‑powered prompt parser**
- Add JSON schema validation + fallback logic
- Implement attribute routing logic
- Build playlist ranking and fusion logic
- Optional: enrichment agent to fetch missing data on demand

**Deliverables**
- `backend/agents/prompt_parser.py` (LLM version)
- `backend/agents/playlist_ranker.py`
- `backend/agents/enrichment_agent.py`

---

## Week 5 — API & System Integration
**Outcome:** System is usable via a clean backend interface.

- Build FastAPI endpoints:
  - `/recommend`
  - `/health`
- Connect parser → retrieval → ranking → response
- Add logging and basic error handling
- Write API tests

**Deliverables**
- `backend/api/main.py`
- `backend/api/routes/recommend.py`
- `tests/test_api.py`

---

## Week 6 — UI + Deployment
**Outcome:** Interactive demo of the system.

- Decide UI framework (Streamlit / Gradio)
- Build simple prompt input + playlist output UI
- Connect UI to FastAPI backend
- Deploy (Hugging Face Spaces / cloud VM)
- Write final documentation

**Deliverables**
- `ui/` implementation
- Live demo
- Updated `README.md`

---

## Final State
- Agentic system that:
  - Parses natural‑language vibe prompts
  - Routes intent to multimodal retrieval
  - Returns coherent, explainable playlists
- Modular architecture (parser, retrieval, ranking, API)
- Clean data + embedding infrastructure

---