# Prompt Decomposition Specification

## Purpose

This document defines how natural-language user prompts are decomposed into structured attributes for the playlist generation system.

The goal of prompt decomposition is **not** to generate music or playlists directly, but to convert free-form human language into a **strict, machine-readable intent representation** that downstream systems (retrieval, filtering, ranking) can operate on deterministically.

This document is the **human-readable design companion** to `prompt_schema.json`, which serves as the machine-enforced contract.

---

## Scope

This specification covers:
- The semantic meaning of each output field
- How ambiguity is handled
- How each field is intended to be used downstream (routing)
- The distinction between hard constraints and soft preferences

This document does **not** define:
- Database schemas
- Retrieval algorithms
- Ranking formulas
- API or UI behavior

---

## Output Contract Overview

The prompt parser must output a JSON object that matches the schema defined in `prompt_schema.json`.

All keys must be present in the output.
Unknown or unspecified information must be represented using `null` (for scalar fields) or `[]` (for list fields).

The fields are:

- `mood`
- `themes`
- `energy`
- `tempo_hint_bpm`
- `instruments`
- `visuals`
- `genres`
- `era`
- `constraints`
- `negative_constraints`
- `confidence`

---

## Field Definitions and Interpretation

### mood
**Type:** list of strings  

Represents the emotional tone or affective quality of the music.

Examples:
- nostalgic
- calm
- melancholic
- euphoric

Notes:
- Mood is often implicit and metaphorical.
- This field should favor semantic interpretation over literal keywords.

---

### themes
**Type:** list of strings  

Represents narrative or conceptual ideas conveyed by the music.

Examples:
- road_trip
- heartbreak
- solitude
- night_city
- nature

Notes:
- Themes often describe *what the music is about* rather than how it feels.
- Multiple themes may coexist.

---

### energy
**Type:** string or null  
**Allowed values:** low, medium, high  

Represents perceived intensity or dynamism of the music.

Notes:
- Energy is qualitative, not numeric.
- If the prompt implies intensity vaguely, treat as a soft preference.
- If unclear, use `null`.

---

### tempo_hint_bpm
**Type:** number or null  

Represents an approximate tempo in beats per minute, only when explicitly implied.

Notes:
- This is a hint, not a command.
- Do not infer BPM unless clearly stated.

---

### instruments
**Type:** list of strings  

Represents instruments that should be present or emphasized.

Examples:
- guitar
- piano
- synth
- strings

---

### visuals
**Type:** list of strings  

Represents imagery or visual scenes the music evokes.

Examples:
- dusk
- neon_city
- fog
- sunset
- forest

---

### genres
**Type:** list of strings  

Represents musical genres or stylistic categories.

Examples:
- indie_rock
- jazz
- ambient
- techno

---

### era
**Type:** list of strings  

Represents a historical period or decade.

Examples:
- 80s
- 90s
- early_2000s

---

### constraints
**Type:** list of strings  

Represents additional requirements or preferences.

Examples:
- instrumental_only
- minimal_vocals
- acoustic_focus

---

### negative_constraints
**Type:** list of strings  

Represents explicit exclusions.

Examples:
- no_edm
- avoid_love_songs
- no_vocals

---

### confidence
**Type:** string or null  
**Allowed values:** low, medium, high  

Represents how confident the parser is in the extracted attributes.

---

## Hard vs Soft Interpretation

Explicit, unambiguous instructions are treated as hard constraints.
Vague or metaphorical language is treated as soft preferences.

---

## Field → Retrieval Routing (Conceptual)

- Vector search: mood, themes, visuals
- Filters: genres, instruments, tempo_hint_bpm, energy (explicit)
- Exclusions: negative_constraints
- Ranking: constraints (soft), energy (vague), era, confidence

---

## Design Principles

- Conservative parsing
- No hallucination
- Deterministic output
- Compiler-style behavior

---

## Success Criteria

A prompt is successfully parsed if:
- Output matches schema exactly
- All keys are present
- No unsupported information is introduced
- Ambiguity is handled using nulls and confidence