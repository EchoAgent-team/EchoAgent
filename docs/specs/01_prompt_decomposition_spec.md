# Prompt Decomposition Specification (VibeIntent)

## Purpose

This document defines how natural-language prompts are decomposed into
`VibeIntent`, the stable intent contract for the playlist retrieval system.

The parser does not pick tracks or call data sources. It only converts free-form
language into a strict, machine-usable intent representation.

This spec is the human-readable companion to
`vibe_intent.md`.

---

## Scope

This specification covers:
- Meaning of each `VibeIntent` section
- Ambiguity handling
- Constraint separation (hard vs soft vs exclusion)

This document does **not** define:
- Database schemas
- Retrieval algorithms
- Ranking formulas
- API or UI behavior

---

## Output Contract Overview

The parser must output a JSON object with exactly these top-level keys:

- `semantic_query`
- `hard_constraints`
- `soft_preferences`
- `exclusions`

```json
{
  "semantic_query": "string",
  "hard_constraints": {},
  "soft_preferences": {},
  "exclusions": {}
}
```

---

## Constraint Separation Rules

- Put explicit, enforceable requirements in `hard_constraints`.
- Put ambiguous or preference-like cues in `soft_preferences`.
- Put explicit “avoid” instructions in `exclusions`.
- Keep routing and datasource logic out of parser output.

---

## Ambiguity Rules

- If language is explicit and measurable, treat it as hard.
- If language is metaphorical or uncertain, treat it as soft.
- Do not invent unsupported fields or values.
- If a nested field is unknown, leave it out of the nested object.

---

## Design Principles

- Conservative parsing
- No hallucination
- Deterministic output
- Compiler-style behavior

---

## Example

Prompt:

`Late-night neon city drive, synth-heavy, medium energy, no EDM.`

Expected `VibeIntent`:

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

---

## Success Criteria

- Output always uses the 4-key `VibeIntent` shape.
- Hard constraints, soft preferences, and exclusions are separated consistently.
- No unsupported/guessed values are invented.
