# Prompt Taxonomy

## Purpose

This document defines a taxonomy of natural-language prompt types supported by the prompt parser.

The goal of the taxonomy is to:
- Ensure broad coverage of real user intent styles
- Provide a structured test set for validating the parser
- Prevent overfitting the parser to a narrow class of prompts

This taxonomy is **descriptive**, not prescriptive.  
Prompts may span multiple categories simultaneously.

---

## Mapping Rule

Categories are mapped into `VibeIntent` sections, not legacy flat attributes.

- `semantic_query`: shared summary for retrieval text matching.
- `hard_constraints`: explicit, enforceable requirements.
- `soft_preferences`: preference-like cues for ranking boosts.
- `exclusions`: explicit avoid/negative instructions.

---

## Category 1: Mood-Only Prompts

Prompts that primarily describe emotional tone without explicit context or constraints.

**Examples**
- "nostalgic and calm"
- "melancholic but warm"
- "euphoric late-night feeling"
- "soft and introspective"

**Primary mapping**
- `semantic_query`
- `soft_preferences.moods`

---

## Category 2: Scene / Visual Prompts

Prompts that describe an imagined visual or cinematic setting.

**Examples**
- "driving at dusk on an empty highway"
- "neon city lights in the rain"
- "foggy forest at sunrise"
- "snow falling outside a train window"

**Primary mapping**
- `semantic_query`
- `soft_preferences.themes`
- optional `soft_preferences.moods`

---

## Category 3: Activity / Context Prompts

Prompts that specify an activity or situational context.

**Examples**
- "coding at 3am"
- "reading on a rainy afternoon"
- "long train ride alone"
- "late-night studying with focus"

**Primary mapping**
- `semantic_query`
- `soft_preferences.themes`
- optional `soft_preferences.energy`

---

## Category 4: Theme / Narrative Prompts

Prompts that focus on lyrical or conceptual subject matter.

**Examples**
- "songs about home"
- "music about heartbreak and healing"
- "themes of solitude and reflection"
- "songs about escape and freedom"

**Primary mapping**
- `semantic_query`
- `soft_preferences.themes`
- optional `soft_preferences.moods`

---

## Category 5: Instrument-Focused Prompts

Prompts that explicitly mention instrumentation.

**Examples**
- "acoustic guitar and soft vocals"
- "piano-driven with strings"
- "synth-heavy soundscape"
- "minimal drums and bass"

**Primary mapping**
- `hard_constraints.instruments_include`
- `semantic_query`
- optional `soft_preferences`

---

## Category 6: Genre / Era Prompts

Prompts that specify musical style, genre, or historical period.

**Examples**
- "90s indie rock"
- "80s synth pop"
- "modern ambient"
- "jazz with a classic feel"

**Primary mapping**
- `hard_constraints.genres_include`
- `hard_constraints.era`
- `semantic_query`

---

## Category 7: Energy / Tempo Prompts

Prompts that describe intensity, pace, or rhythmic drive.

**Examples**
- "high-energy workout music"
- "slow and minimal"
- "fast-paced but not aggressive"
- "medium energy, steady rhythm"

**Primary mapping**
- explicit energy/BPM -> `hard_constraints`
- approximate/qualitative energy/BPM -> `soft_preferences`
- always keep a compact `semantic_query`

---

## Category 8: Constraint-Heavy Prompts

Prompts with explicit requirements or exclusions.

**Examples**
- "instrumental only, no vocals"
- "no EDM, no big drops"
- "short tracks only"
- "avoid love songs"

**Primary mapping**
- positive strict rules -> `hard_constraints`
- avoid rules -> `exclusions`
- optional preference hints -> `soft_preferences`

---

## Category 9: Mixed / Complex Prompts

Prompts combining multiple intent types.

**Examples**
- "nostalgic road trip at dusk with guitars"
- "late-night city rain, minimal vocals, medium energy"
- "dreamy but unsettling, slow tempo, synth-focused"

**Primary mapping**
- split signals across all four `VibeIntent` sections as needed
- do not force the prompt into one category

---

## Usage Notes

- Prompts may map to multiple categories simultaneously.
- The parser should not attempt to force a prompt into a single category.
- Categories exist to guide testing and evaluation, not parsing logic.

---

## Success Criteria

- Taxonomy covers prompts that exercise each `VibeIntent` section.
- Mixed prompts demonstrate correct hard/soft/exclusion separation.
- The taxonomy can seed deterministic fixture-based parser tests.

---

**End of taxonomy.**
