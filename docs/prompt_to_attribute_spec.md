# Prompt → Attribute Spec (v0.1)

## Goal
Convert free-form user prompts into a structured query object that downstream retrieval can execute deterministically.

The LLM acts as a *parser/compiler*: it outputs structured attributes only.
It does NOT choose songs, query APIs, or rank results.

---

## Output Schema (v0.1)

The prompt parser must output JSON with exactly these fields:

```json
{
  "mood": [],
  "themes": [],
  "instruments": [],
  "visuals": [],
  "energy": null,
  "tempo_hint": null,
  "era": null,
  "constraints": []
}

```