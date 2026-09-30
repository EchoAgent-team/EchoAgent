# Testing and Troubleshooting

Status updated 2026-09-23. The focused offline suite last passed **85 tests**, plus two unittest subtests. This is not a claim that the entire repository suite or live Groq/database workflow passed.

## Offline regression suite

From the repository root:

```bash
conda activate echoagent-env
python -m pytest tests/test_json_reliability.py tests/test_api.py tests/test_playlist_builder.py tests/test_candidate_fuser.py tests/test_reranker.py -q
```

The suite makes no real Groq or database requests. API contract tests mock graph results. Integration tests in `test_json_reliability.py` run the real API, LangGraph, agents, fusion, and ranking, replacing only external model transport and retrieval data with fixtures.

Coverage includes:

- JSON mode remains enabled across parser, planner, builder, critic, and repair attempts.
- Empty, truncated, malformed, provider-rejected JSON and invalid fields trigger bounded repair attempts (three total attempts per agent by default).
- Provider 429 errors are retried inside `LLMClient` (up to 3 attempts, `time.sleep` patched in tests); they never become JSON repairs or ranked-track fallbacks, and exhausted retries surface as API 429.
- Immediate acceptance, rejection followed by feedback-driven replanning, and exhausted critic retries.
- Final rejection returns 422; empty candidates return 404.
- Builder fallback is reviewed by the critic and appears in debug output; flags reset after a successful later pass.
- Larger playlists, insufficient unique candidates, fusion, and ranking.

The last run also reported existing Starlette/AnyIO and SQLAlchemy deprecation warnings; these did not fail the suite.

## Live check

Configure `.env` and start the backend:

```bash
uvicorn backend.api.main:app --reload --port 8000
```

In a second terminal:

```bash
curl -sS -i 'http://127.0.0.1:8000/health'
curl -sS -i 'http://127.0.0.1:8000/recommend' -H 'Content-Type: application/json' -d '{"prompt":"late-night rainy city drive"}'
```

Use plain URLs, not pasted Markdown link syntax. Live calls use the configured Groq account and local dataset. A fully accepted live playlist with the latest code remains unverified; the latest two attempts were blocked by token-per-minute quota errors.

| Outcome | HTTP response |
| --- | --- |
| Accepted playlist | 200 with `playlist` and `debug` |
| Final critic rejection | 422 with structured `detail`: `code`, `message`, `reason`, `retry_count`; no playlist |
| Empty playlist | 404 with string `detail` |
| Prompt parsing exhausted | 422 with string `detail` |
| Invalid request body | 422 with FastAPI validation details (a list) |
| Planner/critic output validation exhausted | 502 with string `detail` |
| Groq rate limit (429) after retries | 429 with string `detail`; the frontend shows a rate-limit message |
| Other Groq failure | 502 with string `detail` describing the upstream failure |
| Missing LLM configuration or unavailable database | 503 |
| Unexpected application failure | 500 |

Builder validation exhaustion uses ranked candidates instead of immediately returning an error. The critic still reviews that fallback; acceptance/rejection determines the API outcome. On successful responses, `debug.builder_fallback_used` and `debug.builder_fallback_reason` describe the final builder pass.

## Latest live failures

With `GROQ_MAX_TOKENS=4096`, the supplied logs showed:

| Failed node | Used tokens | Requested tokens | Reported TPM limit |
| --- | ---: | ---: | ---: |
| `plan` | 5,836 | 2,492 | 8,000 |
| `build_playlist` | 4,329 | 4,015 | 8,000 |

In both cases used plus requested exceeded the reported allowance. Groq returned 429 and suggested waiting about 2.5 seconds; the API returned 502. These are observed account/model limits from those requests, not universal Groq limits. The tracebacks describe provider failures, not JSON validation failures or server-startup failures.

`GROQ_MAX_TOKENS` sets a completion ceiling per call, not a total workflow allowance. Increasing it can help output truncation but does not increase the account quota. Several agents and critic retries share that quota. Restarting Uvicorn does not reset it. Waiting the provider-specified interval permits another attempt but does not guarantee the entire multi-call workflow will finish within quota.

As of 2026-09-30, bounded 429 retry (3 attempts, `retry-after` honored, 20s cap) is implemented and token volume was cut (pool and playlist size capped at 5 for testing, compact JSON, `reasoning_effort=low`); parser/planner run on `gpt-oss-20b` and builder/critic on `gpt-oss-120b`. Per-agent token budgets remain deferred. Strict exclusion enforcement and metadata enrichment are also deferred; see [future work](future.md).

## Diagnostics

`backend.agents.json_output` emits INFO completion records with agent, attempt, model, finish reason, and available token usage. Enable INFO for that logger through the Python logging configuration (for example, Uvicorn's `--log-config`); Uvicorn access logs alone do not enable application INFO logs. Repair warnings include agent, attempt, and error type. These diagnostics do not log raw prompts/responses; the API's existing exception logger can include the request prompt and upstream error.

- `finish_reason=length`: generation reached the token ceiling; handled as a repairable output failure.
- `json_validate_failed`: provider rejected generated JSON; repair retains JSON mode.
- `Loading weights: 100%`: normal model loading, not an error.
- `resource_tracker: ... leaked semaphore ... at shutdown`: a separate multiprocessing cleanup warning seen after Ctrl+C. The supplied log does not identify the originating component, and this warning did not cause the earlier quota failures.

Offline verification is complete for the selected suite. Successful live acceptance remains pending; frontend development can proceed with the current response contract and documented limitations.
