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
After deterministic ranking, the builder selects and orders tracks. The pool contains up to `max(POOL_SIZE, playlist_size)` unique valid track IDs (`POOL_SIZE` is temporarily 5, and the planner's `playlist_size` is temporarily clamped to 5; see [Rate limits and model allocation](#rate-limits-and-model-allocation)). If fewer tracks exist, a copy of the plan limits the builder to the available count; the original target remains in graph state for critic review. Empty pools skip the builder model call. Artist diversity is model-guided, not a deterministic guarantee.

### CriticAgent
The critic reviews the completed playlist against the original prompt and plan. It can accept or suggest concrete adjustments (e.g. "increase semantic weight", "raise genre-concentration penalty"). This gives the system an agentic feedback loop without making the ranking opaque. The runner accepts an explicit critic; if none is passed it builds one from the planner's LLM client. The API always passes one explicitly so the critic runs on the heavy model (see below). A hard cap of `max_retries` prevents infinite loops.

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

Builder validation exhaustion falls back to ranked candidates for critic review. `debug.builder_fallback_used` and `debug.builder_fallback_reason` describe the final pass and reset after a successful later build. Provider failures are not converted into that fallback. `LLMClient` retries a Groq 429 up to 3 times, honoring `retry-after` (capped at 20s); if it still fails, the API returns 429 and the frontend shows a rate-limit message. See the next section.

Completion diagnostics use request-local context to log agent, attempt, model, finish reason, and available token usage without storing per-request state on shared clients. See [testing and troubleshooting](testing.md) for logging configuration and offline integration coverage. The latest live tests were quota-blocked; offline success does not establish live acceptance.

## Rate limits and model allocation

Decisions from 2026-09-30, after a frontend run hit Groq 429s repeatedly. Free tier only for now; upgrading to Groq's Dev Tier was explicitly postponed.

### Diagnosis

Not a Groq outage and not a model defect: one `/recommend` request spent more tokens than the free tier's 8,000 tokens/minute (TPM) per model. Contributing factors:

- The builder sent 20 candidates as pretty-printed JSON (roughly 3.6k tokens per call), and every JSON-repair retry resent the whole pool.
- A critic rejection replans and rebuilds, so the builder and critic calls repeat per loop.
- Groq counts requested tokens (prompt + `max_completion_tokens`) against TPM at request time, not only tokens actually used.
- `openai/gpt-oss-*` reasoning tokens come out of the same `max_completion_tokens` budget, so a reasoning-heavy answer can be cut off mid-JSON. This is the likely cause of the repeated `json_validate_failed`; it was not confirmed directly. Once reasoning effort dropped and the pool shrank, the failures stopped.
- `RateLimitError` is not a `ValueError`, so it slipped past the agents' repair loops and killed the whole request (the frontend then showed a timeout message).

### Decisions and reasons

| Decision | Reason / alternatives considered |
| --- | --- |
| Compact JSON (no `indent`) for the builder pool and critic playlist | Pure whitespace savings, no behavior change. |
| Keep `top_tags` in the slimmed candidate | Dropping it would have cut tokens but erodes MVP scope (tags are the taste signal the builder and critic use). Rejected. |
| `POOL_SIZE` 20 → 10 → **5 (temporary, testing)** | Smaller pool = fewer tokens per builder/critic call. 5 is for free-tier testing only. |
| Planner `playlist_size` clamped to 5 by `TESTING_PLAYLIST_SIZE_CAP` (**temporary**) | The builder uses `max(POOL_SIZE, playlist_size)`, so lowering `POOL_SIZE` alone still let a 15-track plan through. The clamp runs after the planner's own 5–50 validation, so invalid planner output is still rejected. Remove both temporary caps for real use. |
| `reasoning_effort="low"` for gpt-oss (env `GROQ_REASONING_EFFORT`) | Stops reasoning from eating the completion budget. Sent only to `openai/gpt-oss*` models. `GROQ_MAX_TOKENS` left at 1024; observed completions are 90-310 tokens. |
| Did not trim the repair prompt | Considered dropping the previous bad output from it. On `json_validate_failed` there is no previous output, so it saved almost nothing. |
| 429 retry lives in `LLMClient`, not in the agents' repair loops | A 429 is a transport problem, not bad JSON; it must not consume a repair attempt or trigger the builder fallback. 3 attempts, wait = `retry-after` header (default 2s x attempt), capped at 20s. After that the API returns 429 (was 502) and `frontend/app.py` maps it to a friendly message. |
| Two Groq clients: **`gpt-oss-20b`** for PromptParser + PlannerAgent, **`gpt-oss-120b`** for PlaylistBuilderAgent + CriticAgent | Free-tier chat models all share the same limits (30 RPM, 1K requests/day, 8K TPM, 200K tokens/day per model), so a cheaper model doesn't cost less; it gives a second, separate TPM bucket (assumed per-model, which Groq's error message suggests but we have not confirmed). Parser and planner are structured-JSON tasks on small inputs. The critic was moved to 120b because it judges taste and its reject decision triggers an expensive replan. Env: `GROQ_MODEL_LIGHT` (default `openai/gpt-oss-20b`), `GROQ_MODEL` (default `openai/gpt-oss-120b`). |
| Models not used | `qwen/qwen3.8-27b` (preview; emits thinking output that breaks JSON extraction), `openai/gpt-oss-safeguard-20b` (preview; safety-classifier variant), llama models (listed on the models page but absent from the free rate-limit table; CLAUDE.md records them as deprecated). |
| INFO logging on by default in the API (`LOG_LEVEL` overrides) | uvicorn's log level does not reach our own loggers; without `logging.basicConfig` the per-call token/finish-reason lines were invisible. |

### Observed result (first run after the changes)

Prompt "late-night rainy city drive, no metal music", 47s total, no 429s, no JSON repairs, every call `finish_reason=stop`. Tokens per call: parser ~750, planner ~1.3-1.6k, builder ~1.3k (was ~3.6k), critic ~1.5k. The 120b bucket used about 8.5k tokens across three passes, i.e. right at the TPM limit; the 200K/day cap allows roughly 20-25 full requests per day on 120b. The critic rejected all three passes ("tracks are unrelated, low energy, genre inappropriate"), so the remaining failure is retrieval/ranking quality, not rate limiting.

### Naming note

The `agent=` field in `LLM completion` and `LLM repair` log lines is just the caller name passed to `generate_json`. It is a retry counter for a call site, not an agent. Renaming the field (e.g. `caller`) was discussed and not done.

