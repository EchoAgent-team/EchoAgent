# EchoAgent

EchoAgent is a music discovery and playlist generation project for translating free-form **vibe prompts** into structured retrieval, hybrid search, and playlist assembly logic. The project is being built as a portfolio piece focused on ML/AI systems design, retrieval, data engineering, and agentic orchestration rather than as a thin wrapper around a single LLM call.

## 🎧 Motivation

The goal is to let a user describe a playlist the way people naturally do, with language such as atmosphere, context, energy, exclusions, and emotional cues, and convert that into a system that can retrieve and assemble relevant tracks. The current design separates language understanding from deterministic retrieval and ranking so that ambiguous prompt interpretation can be handled by an LLM while database queries, filtering, and playlist construction remain system-controlled and inspectable.

Example prompts:

- "Late-night train ride through a rainy city, introspective but not depressing."
- "Warm, nostalgic indie for an autumn walk, low energy, no acoustic ballads."
- "High-energy gym tracks with strong rhythm and minimal lyrical focus."

## ⚙️ System Overview

EchoAgent is currently shaped as a hybrid retrieval system with a typed intent layer in front of relational and vector search. The main flow is: parse the prompt, normalize it into a structured contract, retrieve candidate tracks from two complementary stores, then fuse and rank them into a playlist.

```mermaid
flowchart TD
    A[User vibe prompt] --> B[PromptParser]
    B --> C[VibeIntent]
    C --> D[Relational mapper]
    C --> E[Semantic query]
    D --> F[Relational DB retrieval]
    E --> G[Vector DB retrieval]
    F --> H[Candidate fusion]
    G --> H
    H --> I[Ranking and playlist assembly]
    I --> J[Playlist output]
```

This diagram reflects the current architectural intent more accurately than the previous README because it centers the existing parser, typed intent object, mapper, and vector retrieval wrapper that are already present in the repository. The retrieval and ranking path is deliberately system-controlled, while the prompt interpretation edge is where LLM-based reasoning is currently most useful. 

## 🏗 Current architecture

The repository already contains the core pieces needed for a hybrid retrieval stack. It includes a prompt parser, a typed intent contract, a relational retrieval mapping layer, a vector retrieval layer, local data and embedding utilities, documentation, notebooks, and tests. 

### 🪄 Prompt understanding

`PromptParser` converts a natural-language prompt into a validated `VibeIntent` object with four main fields: `semantic_query`, `hard_constraints`, `soft_preferences`, and `exclusions`. It uses a schema-driven prompt and validation flow, including retry-and-repair behavior when the model output is invalid JSON.

`VibeIntent` acts as the typed contract between language understanding and retrieval/ranking logic. It normalizes values, canonicalizes certain fields, and checks for contradictions such as overlap between hard constraints, soft preferences, and exclusions. 

```mermaid
flowchart LR
    A[Free-form prompt] --> B[Schema-constrained LLM parsing]
    B --> C[Raw JSON output]
    C --> D[Validation and repair]
    D --> E[VibeIntent]
    E --> F[Hard constraints]
    E --> G[Soft preferences]
    E --> H[Exclusions]
    E --> I[Semantic query]
```

### ⛓️ Relational retrieval

The relational database stores artist, album, and track metadata together with audio features such as tempo, danceability, energy, loudness, mode, and duration, plus seed genre labels, top tags, and lyrics bag-of-words data. This store is implemented with SQLAlchemy models and query helpers that support structured filtering over year, genre, tempo, energy, and danceability ranges. 

The relational retrieval mapper translates `VibeIntent.hard_constraints` into deterministic database filters. It already supports mappings for artist, genre, year or era, tempo, energy, and danceability, including both bucketed values such as `low` or `medium` and explicit numeric ranges. 

### ↗️ Vector retrieval

The vector side uses ChromaDB and sentence-transformer embeddings for track text retrieval, with utilities for embedding generation, metadata extraction, upsert, and semantic search. The embedding layer supports persisted Chroma collections and query-time semantic search using `semantic_query` text derived from the prompt. 

The current vector retrieval wrapper already returns cleaned candidate objects and includes a LangGraph-style node interface, which makes it a natural building block for a graph-based orchestration layer. 

## 📦 Data stores

EchoAgent uses two complementary local data stores.

```mermaid
flowchart LR
    A[Metadata and audio features] --> B[Relational DB]
    C[Tags and genre labels] --> B
    D[Lyrics BoW and retrieval text] --> E[Embedding pipeline]
    C --> E
    E --> F[Chroma vector DB]
    B --> G[Structured filtering]
    F --> H[Semantic retrieval]
```

### Relational DB

- Artist, album, and track metadata. 
- Audio features including tempo, danceability, energy, loudness, mode, and duration.
- Seed genre labels and top tag metadata. 
- Lyrics bag-of-words storage for downstream use.

### Vector DB

- Persisted Chroma collections for track text embeddings and legacy lyrics embeddings. 
- Sentence-transformer-based embedding generation and semantic query support. 
- Metadata attached to vector entries for downstream retrieval and filtering. 

## 🧩 Design principles

A central design decision in this project is to be selective about where agentic reasoning is useful. Prompt interpretation is inherently ambiguous and benefits from LLM-based parsing, while retrieval mapping and database querying are better handled as deterministic system components. 

The planned orchestration direction is therefore not "everything is an agent," but rather a mixed system of LLM-driven and deterministic nodes connected through LangGraph. That allows the project to showcase agentic workflow design without sacrificing traceability or engineering rigor. [

## 📈 LangGraph workflow

The orchestration layer is implemented in `backend/agents/playlist_graph.py`. The workflow is:

```mermaid
flowchart TD
    A[User prompt] --> B[parse_intent node]
    B --> C[relational_mapper node]
    C --> D[relational_retrieval node]
    B --> E[vector_retrieval node]
    D --> F[fusion_ranking node]
    E --> F
    F --> G[playlist_builder node]
    G --> H[critic node optional]
    H --> I[playlist output]
```

This graph reflects the implemented workflow: one stateful LangGraph pipeline combining parsed intent, deterministic retrieval, semantic retrieval, candidate fusion, scoring, and playlist assembly. The critic routes back to the planner on rejection, capped at `max_retries`.

## ✅ Project status

What is implemented:

- Schema-driven prompt parsing into structured intent objects.
- Typed `VibeIntent` contract with normalization and validation.
- `PlannerAgent` producing a structured `PlaylistPlan` with scoring weights and retrieval limits.
- Relational database models, query helpers, and deterministic intent-to-filter mapping.
- Chroma-backed embedding management and semantic retrieval utilities.
- `CandidateFuser` merging relational and vector candidates by track ID.
- Deterministic `Reranker` executing planner-supplied weights with per-candidate score components.
- `PlaylistBuilderAgent` with artist-repeat and genre-concentration controls.
- Full LangGraph orchestration graph with parallel retrieval, critic routing, and retry loop.
- Shared `PlaylistGraphState` schema across all nodes.

What is actively being built next:

- End-to-end graph testing and integration bug fixes.
- `CriticAgent` with LLM-backed structured review (currently a stub that always accepts).
- API route wiring to connect the graph to the FastAPI endpoints.

## 📁 Repository structure

```
EchoAgent/
├── backend/                    # Core application backend
│   ├── agents/                 # LangGraph nodes and agents
│   ├── api/                    # FastAPI application
│   │   └── routes/             # API route handlers
│   ├── data/                   # Data management and ingestion
│   └── utils/                  # Utility modules
│
├── database/                   # Persisted data stores
│   └── chroma_db/              # ChromaDB vector store
│
├── docs/                       # Documentation
│   ├── architecture.md         # Design decisions and rationale
│   ├── future.md               # Deferred and Phase 2 ideas
│   ├── roadmap.md              # Current status and next priorities
│   └── specs/                  # Detailed interface specifications
│
├── notebooks/                  # Jupyter notebooks for exploration and testing
│
└── tests/                      # Test suite
```

This structure reflects the current emphasis of the repository: prompt understanding modules, retrieval modules, local data infrastructure, documentation, and early test coverage.

## 🛠 Tech stack

- Python for the core system and data pipeline. 
- SQLAlchemy-backed relational storage over SQLite today, with the schema written in a way that can also support PostgreSQL. 
- ChromaDB plus sentence-transformer embeddings for semantic retrieval. 
- Transformer-based LLM prompting for schema-constrained prompt parsing. 
- LangGraph as the planned orchestration layer for the multi-step retrieval and playlist workflow. 
- FastAPI and an interactive front end as the planned serving layer. 

## 🛣️ Near-term roadmap

- Test the full LangGraph pipeline end-to-end and resolve integration bugs.
- Implement `CriticAgent` with LLM-backed structured review and retry logic.
- Wire FastAPI routes to the assembled graph for an initial API endpoint.
- Expand test coverage: graph integration tests, API contract tests, critic behavior.

## Future directions
Phase 2:
- Live list of tracks which the user can send to their audio service of choice (Spotify, Apple Music, etc.)
- User feedback loop for refining playlist results over time.

## Notes

The public README focuses on architecture, current progress, and next steps. The current state of the project is best described as a partially implemented hybrid retrieval system with a clear path toward LangGraph-based orchestration and playlist generation. 
