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

## Category 1: Mood-Only Prompts

Prompts that primarily describe emotional tone without explicit context or constraints.

**Examples**
- "nostalgic and calm"
- "melancholic but warm"
- "euphoric late-night feeling"
- "soft and introspective"

**Primary fields**
- mood

**Secondary fields**
- energy (if implied)
- confidence

---

## Category 2: Scene / Visual Prompts

Prompts that describe an imagined visual or cinematic setting.

**Examples**
- "driving at dusk on an empty highway"
- "neon city lights in the rain"
- "foggy forest at sunrise"
- "snow falling outside a train window"

**Primary fields**
- visuals
- themes

**Secondary fields**
- mood
- energy

---

## Category 3: Activity / Context Prompts

Prompts that specify an activity or situational context.

**Examples**
- "coding at 3am"
- "reading on a rainy afternoon"
- "long train ride alone"
- "late-night studying with focus"

**Primary fields**
- themes
- mood

**Secondary fields**
- energy
- constraints

---

## Category 4: Theme / Narrative Prompts

Prompts that focus on lyrical or conceptual subject matter.

**Examples**
- "songs about home"
- "music about heartbreak and healing"
- "themes of solitude and reflection"
- "songs about escape and freedom"

**Primary fields**
- themes

**Secondary fields**
- mood
- confidence

---

## Category 5: Instrument-Focused Prompts

Prompts that explicitly mention instrumentation.

**Examples**
- "acoustic guitar and soft vocals"
- "piano-driven with strings"
- "synth-heavy soundscape"
- "minimal drums and bass"

**Primary fields**
- instruments

**Secondary fields**
- constraints
- genres

---

## Category 6: Genre / Era Prompts

Prompts that specify musical style, genre, or historical period.

**Examples**
- "90s indie rock"
- "80s synth pop"
- "modern ambient"
- "jazz with a classic feel"

**Primary fields**
- genres
- era

**Secondary fields**
- mood
- energy

---

## Category 7: Energy / Tempo Prompts

Prompts that describe intensity, pace, or rhythmic drive.

**Examples**
- "high-energy workout music"
- "slow and minimal"
- "fast-paced but not aggressive"
- "medium energy, steady rhythm"

**Primary fields**
- energy
- tempo_hint_bpm (only if numeric or explicit)

**Secondary fields**
- mood

---

## Category 8: Constraint-Heavy Prompts

Prompts with explicit requirements or exclusions.

**Examples**
- "instrumental only, no vocals"
- "no EDM, no big drops"
- "short tracks only"
- "avoid love songs"

**Primary fields**
- constraints
- negative_constraints

**Secondary fields**
- confidence

---

## Category 9: Mixed / Complex Prompts

Prompts combining multiple intent types.

**Examples**
- "nostalgic road trip at dusk with guitars"
- "late-night city rain, minimal vocals, medium energy"
- "dreamy but unsettling, slow tempo, synth-focused"

**Primary fields**
- mood
- themes
- visuals
- instruments

**Secondary fields**
- energy
- constraints
- confidence

---

## Usage Notes

- Prompts may map to multiple categories simultaneously.
- The parser should not attempt to force a prompt into a single category.
- Categories exist to guide testing and evaluation, not parsing logic.

---

## Success Criteria

The taxonomy is considered sufficient if:
- Every field in `prompt_schema.json` appears in at least one category
- Both simple and complex prompts are represented
- The taxonomy can be used to construct a test suite of 20–30 prompts

---

**End of taxonomy.**