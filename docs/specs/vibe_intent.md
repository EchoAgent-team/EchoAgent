# VibeIntent Specification

## Purpose

`VibeIntent` is the stable contract between natural-language prompts and the
retrieval pipeline.

It replaces prompt-to-attribute routing as the parser's primary output.
Routing decisions happen later, after intent normalization.

## Top-Level Schema

The parser must return exactly this shape:

```json
{
  "semantic_query": "string",
  "hard_constraints": {},
  "soft_preferences": {},
  "exclusions": {}
}
```

## Field Definitions

- `semantic_query`:
  - A compact normalized text query used for vector retrieval.
  - Includes key vibe words (mood, scene, instrumentation, context).

- `hard_constraints`:
  - Explicit requirements that should be enforced where possible.
  - Typical keys:
    - `genres_include: string[]`
    - `instruments_include: string[]`
    - `energy: "low" | "medium" | "high"`
    - `tempo_bpm_min: number`
    - `tempo_bpm_max: number`
    - `instrumental_only: boolean`
    - `era: string[]`

- `soft_preferences`:
  - Signals used for ranking/boosting, not strict filtering.
  - Typical keys:
    - `moods: string[]`
    - `themes: string[]`
    - `energy: "low" | "medium" | "high"`
    - `tempo_bpm: number`
    - `tags: string[]`

- `exclusions`:
  - Explicit negatives and avoid rules.
  - Typical keys:
    - `genres_exclude: string[]`
    - `artists_exclude: string[]`
    - `tags_exclude: string[]`
    - `vocal_content: "no_vocals"`

## Normalization Rules

- Top-level keys are always present.
- Nested objects may be empty (`{}`) when no signal is available.
- Do not invent missing values.
- Explicit user language maps to `hard_constraints` or `exclusions`.
- Ambiguous language defaults to `soft_preferences`.

## Constraint Separation Rules

- Hard constraints are deterministic retrieval filters.
- Soft preferences are reranking signals.
- Exclusions always encode "must avoid" logic.

## Example

Prompt:

`Late-night neon city drive, synth-heavy, medium energy, no EDM.`

Output:

```json
{
  "semantic_query": "late-night neon city drive synth-heavy medium energy no edm",
  "hard_constraints": {
    "instruments_include": ["synth"],
    "energy": "medium"
  },
  "soft_preferences": {
    "themes": ["driving", "night_city"]
  },
  "exclusions": {
    "genres_exclude": ["edm"]
  }
}
```
