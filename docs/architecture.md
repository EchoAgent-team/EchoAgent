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
Accept → output   /   Reject → back to Planner (capped at max_retries)
```

LangGraph nodes: `parse_intent → plan → retrieve_relational + retrieve_vector → fuse_candidates → rerank → build_playlist → critique`

## Where LLMs Are Used and Why

### PromptParser (Intent Agent)
Natural language is inherently ambiguous. The same word ("dark", "chill", "heavy") means different things across contexts and users. Code cannot reliably extract hard constraints, soft preferences, and exclusions from free-form text. The LLM outputs a structured `VibeIntent` object; downstream code never touches raw text again.

### PlannerAgent (Taste Strategy)
The planner reads the prompt and `VibeIntent` and decides the scoring strategy: how many tracks, what weight to give semantic similarity vs. relational matches vs. soft preferences, how strict to be on diversity, and retrieval limits. These are subjective tradeoffs — code has no good way to decide that a "rainy night" prompt should weight semantic similarity at 0.55 while a "90s grunge" prompt should weight relational genre matching more heavily. The LLM produces a `PlaylistPlan` dataclass; the code executes it.

### PlaylistBuilderAgent
After deterministic ranking, the builder makes final selection and ordering decisions — which tracks to include, how to shape energy flow, final artist diversity. This is another taste judgment call better suited to an LLM than a fixed heuristic.

### CriticAgent
The critic reviews the completed playlist against the original prompt and plan. It can accept or suggest concrete adjustments (e.g. "increase low-energy weight", "reduce genre concentration"). This gives the system an agentic feedback loop without making the ranking opaque. A hard cap of `max_retries` prevents infinite loops.

## Why Not Pure LLM Ranking

Several reasons:

**Scale.** Retrieving 100–500 candidates then asking an LLM to rank them is slow, expensive, and unreliable. LLMs are not designed for large list ranking.

**Consistency.** The same prompt can produce different orderings across calls unless very tightly constrained.

**Debuggability.** A score like `semantic_similarity: 0.38, relational_match: 0.20, exclusion_penalty: -0.30` is inspectable and testable. "The LLM liked it" is not.

**Data fit.** Audio features, vector distances, SQL match flags, genre scores, and tag weights are structured numeric data. Code handles this better than language models.

The LLM decides *what to optimize for*. Code does the optimization.

## VibeIntent as the Interface Contract

`VibeIntent` is the stable boundary between language understanding and retrieval logic. It has exactly four keys: `semantic_query`, `hard_constraints`, `soft_preferences`, `exclusions`. Nothing downstream touches raw text. Nothing upstream knows about SQL or Chroma.

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

`Reranker` computes a single `final_score` per candidate using planner-supplied weights:

```
final_score = (semantic_weight × vector_similarity)
            + (relational_weight × relational_match_flag)
            + (soft_preference_weight × soft_match_score)
            + (novelty_weight × novelty_bonus)
            - (exclusion_penalty × exclusion_match)
            - (artist_repeat_penalty × artist_repeat_count)
            - (genre_concentration_penalty × genre_concentration)
```

Score components are preserved on each candidate for traceability and critic review.

## Retry Logic

The critic routes to one of two outcomes:
- `accept: True` → end of graph, return playlist
- `accept: False` + `retry_count < max_retries` → route back to `plan` node with `suggested_adjustments` in state
- `accept: False` + retries exhausted → return best-effort playlist

The planner can read `suggested_adjustments` on retry to modify weights. `max_retries` defaults to 2.
