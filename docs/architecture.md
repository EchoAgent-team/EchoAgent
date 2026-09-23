# EchoAgent Architecture

## Core Design Principle

EchoAgent uses agents for subjective interpretation and tradeoff decisions, while deterministic retrieval and ranking tools execute those decisions over structured relational and vector data.

The motivation: prompt interpretation is inherently ambiguous and benefits from LLM reasoning. But retrieval mapping, database querying, scoring, and playlist construction are better handled as deterministic, inspectable system components. Every LLM call in the system has a specific job that code genuinely cannot do well — everything else stays in code.

## System Shape

```text
User prompt
  ↓
PromptParser (LLM — intent agent)
  ↓
VibeIntent
  ↓
PlannerAgent (LLM — decides taste strategy / scoring weights)
  ↓
RelationalRetrievalAgent + VectorRetrievalAgent (parallel, deterministic)
  ↓
CandidateFuser (deterministic)
  ↓
Reranker (deterministic — executes planner weights)
  ↓
PlaylistBuilderAgent (LLM — selects and orders final tracks)
  ↓
CriticAgent (LLM — accepts or suggests adjustments)
  ↓
Accept → output / Reject → Planner (bounded retries) / Exhausted rejection → API 422
```

LangGraph nodes: `parse_intent → plan → retrieve_relational + retrieve_vector → fuse_candidates → rerank → build_playlist → critique`

## Where LLMs Are Used and Why

### PromptParser (Intent Agent)
Natural language is inherently ambiguous. The same word ("dark", "chill", "heavy") means different things across contexts and users. Code cannot reliably extract hard constraints, soft preferences, and exclusions from free-form text. The LLM outputs a structured `VibeIntent` object; deterministic retrieval consumes structured intent; planner, builder, and critic also receive the original prompt.

### PlannerAgent (Taste Strategy)
The planner reads the prompt and `VibeIntent` and decides the scoring strategy: how many tracks, what weight to give semantic similarity vs. relational matches vs. soft preferences, how strict to be on diversity, and retrieval limits. These are subjective tradeoffs — code has no good way to decide that a "rainy night" prompt should weight semantic similarity at 0.55 while a "90s grunge" prompt should weight relational genre matching more heavily. The LLM produces a `PlaylistPlan` dataclass; the code executes it.

### PlaylistBuilderAgent
After deterministic ranking, the builder selects and orders tracks. The pool contains up to `max(20, playlist_size)` unique valid track IDs. If fewer tracks exist, a copy of the plan limits the builder to the available count; the original target remains in graph state for critic review. Empty pools skip the builder model call. Artist diversity is model-guided, not a deterministic guarantee.

### CriticAgent
The critic reviews the completed playlist against the original prompt and plan. It can accept or suggest concrete adjustments (e.g. "increase semantic weight", "raise genre-concentration penalty"). This gives the system an agentic feedback loop without making the ranking opaque. The runner supplies a critic by default using the planner's LLM client, or accepts an explicit critic. A hard cap of `max_retries` prevents infinite loops.

## Why Not Pure LLM Ranking

Several reasons:

**Scale.** Retrieving 100–500 candidates then asking an LLM to rank them is slow, expensive, and unreliable. LLMs are not designed for large list ranking.

**Consistency.** The same prompt can produce different orderings across calls unless very tightly constrained.

**Debuggability.** A score like `semantic_similarity: 0.38, relational_match: 0.20, exclusion_penalty: -0.30` is inspectable and testable. "The LLM liked it" is not.

**Data fit.** Audio features, vector distances, SQL match flags, genre scores, and tag weights are structured numeric data. Code handles this better than language models.

The LLM decides *what to optimize for*. Code does the optimization.

## VibeIntent as the Interface Contract

`VibeIntent` is the stable boundary between language understanding and retrieval logic. It has exactly four keys: `semantic_query`, `hard_constraints`, `soft_preferences`, `exclusions`. Deterministic retrieval uses this structured contract. Later LLM agents also receive the original prompt; they do not need to know the SQL or Chroma implementation.

This boundary makes each layer independently testable and replaceable.

## Two-Store Retrieval Architecture

EchoAgent uses two complementary retrieval stores that run in parallel:

**Relational DB (SQLite/SQLAlchemy)**
- Stores artist, album, track metadata, audio features, genre labels, top tags.
- Used for deterministic hard-filter retrieval: genre, era, tempo range, energy level, artist constraints.
- Maps directly from `VibeIntent.hard_constraints` via `RelationalRetrievalMapper`.

**Vector DB (ChromaDB + sentence-transformers)**
- Stores track-text embeddings derived from BoW lyrics and retrieval text.
- Used for semantic similarity: finds tracks whose lyrical content and vibe descriptions match the `semantic_query`.
- Returns candidates with vector distance scores.

Candidates from both stores are merged by `track_id` in `CandidateFuser`. Tracks appearing in both get a boost; origin is tracked per candidate.

## Scoring and Ranking

`CandidateFuser` computes `retrieval_score` from the relational-source weight and reciprocal vector rank. `Reranker` then computes:

```text
score = retrieval_score
      + semantic_weight * semantic_score
      + soft_preference_weight * soft_match_score
      - exclusion_penalty * exclusion_match
```

Score components are preserved in `ranking_debug`. The plan also carries novelty and diversity controls, but this deterministic reranker does not implement all of them as score terms. Artist-repeat guidance is provided to the builder. Exclusion penalties are not hard filtering; deterministic exclusion enforcement is deferred.

## Retry Logic

Two retry mechanisms are separate:

- **JSON repair:** each structured agent has up to three generation attempts by default. JSON mode stays enabled. Provider JSON-validation failures, empty content, token-limit truncation, malformed JSON, and invalid fields enter repair. Groq provider errors such as 429 propagate through the API error mapper instead.
- **Critic replanning:** accepted reports end the graph. Rejection with retries remaining returns to `plan`, which includes the previous plan, critic reason, and suggested adjustments in the planner input. `retry_count` counts replans, not reviews; the default `max_retries=2` allows an initial playlist plus two revisions.

An exhausted critic rejection ends graph execution with the rejected state, but the API withholds those tracks and returns HTTP 422 with `detail.code = "playlist_rejected"`, a message, reason, and retry count. Empty playlists retain HTTP 404.

Builder validation exhaustion falls back to ranked candidates for critic review. `debug.builder_fallback_used` and `debug.builder_fallback_reason` describe the final pass and reset after a successful later build. Provider failures are not converted into that fallback. A Groq 429 currently becomes API 502; application-level quota backoff and per-agent budgets are deferred.

Completion diagnostics use request-local context to log agent, attempt, model, finish reason, and available token usage without storing per-request state on shared clients. See [testing and troubleshooting](testing.md) for logging configuration and offline integration coverage. The latest live tests were quota-blocked; offline success does not establish live acceptance.
