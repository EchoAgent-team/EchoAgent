from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .vibe_intent import VibeIntent


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class RetrievalLimits:
    """Per-source candidate caps passed to the retrieval workers."""
    n_vector: int = 100
    n_relational: int = 100

    def validate(self) -> None:
        for attr in ("n_vector", "n_relational"):
            v = getattr(self, attr)
            if not isinstance(v, int) or not (5 <= v <= 500):
                raise ValueError(f"retrieval_limits.{attr} must be an int between 5 and 500, got {v!r}")


@dataclass
class PlaylistPlan:
    """
    Structured output of the Planner Agent.

    Defines the scoring strategy and retrieval parameters that the downstream
    deterministic nodes (candidate_fuser, reranker, playlist_builder) will execute.

    The LLM decides the *taste strategy*; the code executes the scoring.
    """

    # --- playlist sizing ---
    playlist_size: int = 20

    # --- scoring weights (should sum ~1.0) ---
    semantic_weight: float = 0.45        # vector/semantic similarity
    relational_weight: float = 0.25      # SQL / relational hard-match
    soft_preference_weight: float = 0.20 # tag / genre / audio-feature soft match
    novelty_weight: float = 0.10         # discovery / less-familiar boost

    # --- diversity / exclusion penalties ---
    artist_repeat_penalty: float = 0.20
    genre_concentration_penalty: float = 0.15
    exclusion_penalty: float = 0.30      # subtracted when a track matches an exclusion rule

    # --- retrieval config ---
    retrieval_limits: RetrievalLimits = field(default_factory=RetrievalLimits)

    # --- search strategy ---
    broaden_if_low_recall: bool = True
    diversity_strictness: str = "medium"   # "low" | "medium" | "high"

    # --- traceability ---
    rationale: str = ""

    # class-level constants (not dataclass fields)
    _WEIGHT_FIELDS = (
        "semantic_weight",
        "relational_weight",
        "soft_preference_weight",
        "novelty_weight",
    )
    _DIVERSITY_LEVELS = frozenset({"low", "medium", "high"})

    def validate(self) -> None:
        """Raise ValueError / TypeError if any field is out of contract."""
        if not isinstance(self.playlist_size, int) or not (5 <= self.playlist_size <= 50):
            raise ValueError(f"playlist_size must be an int in [5, 50], got {self.playlist_size!r}")

        for key in self._WEIGHT_FIELDS:
            v = getattr(self, key)
            if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{key} must be a float in [0.0, 1.0], got {v!r}")

        total = sum(float(getattr(self, k)) for k in self._WEIGHT_FIELDS)
        if not (0.8 <= total <= 1.2):
            raise ValueError(f"Primary weights should sum close to 1.0 (got {total:.3f}). "
                "Adjust semantic_weight + relational_weight + soft_preference_weight + novelty_weight.")

        for key in ("artist_repeat_penalty", "genre_concentration_penalty", "exclusion_penalty"):
            v = getattr(self, key)
            if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"{key} must be a float in [0.0, 1.0], got {v!r}")

        if self.diversity_strictness not in self._DIVERSITY_LEVELS:
            raise ValueError(
                f"diversity_strictness must be one of {set(self._DIVERSITY_LEVELS)}, "
                f"got '{self.diversity_strictness}'")

        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("rationale must be a non-empty string.")

        self.retrieval_limits.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------

class PlannerAgent:
    """
    Planner Agent — decides *how* to build the playlist, not what's in it.

    Input:
        user_prompt (str)       — original natural-language request
        intent (VibeIntent)     — structured output from PromptParser

    Output:
        PlaylistPlan            — scoring weights, retrieval limits, diversity config

    The agent calls the LLM once (with retries on validation failure) to fill in
    the PlaylistPlan JSON, then validates and returns a typed PlaylistPlan object.
    """

    # -----------------------------------------------------------------------
    # Schema shown to the LLM
    # -----------------------------------------------------------------------
    _OUTPUT_SCHEMA: Dict[str, Any] = {
        "playlist_size": "int: target number of tracks (5–50)",
        "semantic_weight": "float [0.0–1.0]: weight for vector/semantic similarity",
        "relational_weight": "float [0.0–1.0]: weight for SQL/relational hard-match score",
        "soft_preference_weight": "float [0.0–1.0]: weight for tag/genre/audio-feature soft-match",
        "novelty_weight": "float [0.0-1.0]: discovery boost — higher when user wants something new",
        "artist_repeat_penalty": "float [0.0–1.0]: penalize repeated artists in the playlist",
        "genre_concentration_penalty": "float [0.0–1.0]: penalize genre collapse",
        "exclusion_penalty": "float [0.0–1.0]: subtracted from final_score when a track matches an exclusion rule",
        "retrieval_limits": {
            "vector": "int (50–200): max candidates from the vector store",
            "relational": "int (50–200): max candidates from the relational store",
        },
        "broaden_if_low_recall": "bool: widen retrieval if fewer than half of limits are returned",
        "diversity_strictness": "string: exactly 'low', 'medium', or 'high'",
        "rationale": "string: 1–2 sentence plain-English explanation of the weight choices",
    }

    _OUTPUT_RULES = [
        "Output JSON ONLY — no markdown fences, no prose, no extra keys.",
        "All float weights must be in [0.0, 1.0].",
        "semantic_weight + relational_weight + soft_preference_weight + novelty_weight must sum to ~1.0.",
        "playlist_size must be an integer between 5 and 50.",
        "retrieval_limits.vector and retrieval_limits.relational must each be between 50 and 200.",
        "diversity_strictness must be exactly one of: 'low', 'medium', 'high'.",
        "broaden_if_low_recall must be a JSON boolean (true or false, not a string).",
        "rationale must be a non-empty string.",
        "Raise semantic_weight when the prompt is mood/vibe/atmosphere-driven.",
        "Raise relational_weight when the prompt has explicit hard constraints (genre, era, tempo, instrument).",
        "Raise novelty_weight when the prompt signals discovery ('something new', 'underground', 'hidden gems', 'surprise me').",
        "Lower novelty_weight when the prompt signals familiarity ('classics', 'my favorites', 'familiar songs').",
        "Set diversity_strictness to 'high' if the prompt explicitly mentions variety or avoiding repetition.",
        "Set diversity_strictness to 'low' if the prompt is narrow/specific (single genre, single artist style).",
        "Set exclusion_penalty higher (0.4–0.8) when the VibeIntent exclusions dict is non-empty; keep it low (0.1–0.2) when exclusions are absent.",
    ]

    def __init__(self, llm_client: Any, max_retries: int = 3) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries
        self._system_prompt = self._build_system_prompt()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def plan(self, user_prompt: str, intent: VibeIntent) -> PlaylistPlan:
        """
        Run the Planner Agent.

        Args:
            user_prompt: The original natural-language request from the user.
            intent:      The VibeIntent produced by PromptParser.

        Returns:
            A validated PlaylistPlan.

        Raises:
            ValueError: If all retries are exhausted without a valid plan.
        """
        user_message = self._build_user_message(user_prompt, intent)

        last_err: Optional[str] = None
        last_raw: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            if attempt == 1:
                system_prompt = self._system_prompt
            else:
                system_prompt = self._build_repair_prompt(
                    error_message=last_err or "unknown_validation_error",
                    last_raw=last_raw or "",)

            raw = self.llm_client.generate(
                system_prompt=system_prompt,
                user_input=user_message,)
            last_raw = raw

            try:
                plan = self._validate_and_parse_output(raw)
                return plan
            except (ValueError, TypeError) as exc:
                last_err = str(exc)
                continue

        raise ValueError(
            f"PlannerAgent failed to produce a valid PlaylistPlan after "
            f"{self.max_retries} attempts. Last error: {last_err}")

    # -----------------------------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        schema_str = json.dumps(self._OUTPUT_SCHEMA, indent=2, ensure_ascii=True)
        rules_str = "\n".join(f"- {r}" for r in self._OUTPUT_RULES)

        return (
            "You are an expert playlist strategy planner for a music recommendation system.\n"
            "You will receive a user's music request and a structured VibeIntent object.\n"
            "Your job is to decide the optimal playlist strategy by choosing scoring weights "
            "and retrieval parameters. You are NOT choosing the tracks — the deterministic "
            "pipeline does that. You are deciding the *taste policy*.\n\n"
            "STRICT OUTPUT REQUIREMENTS:\n"
            "1) Output JSON ONLY (no markdown, no prose).\n"
            "2) Output must match the schema keys EXACTLY.\n"
            "3) Include ALL keys. DO NOT add extra keys.\n"
            "4) Float weights must be in [0.0, 1.0] and the four primary weights must sum ~1.0.\n"
            "5) diversity_strictness must be exactly one of: 'low', 'medium', 'high'.\n"
            "6) rationale must be a non-empty explanation of your weight choices.\n\n"
            f"OUTPUT RULES:\n{rules_str}\n\n"
            "OUTPUT SCHEMA (return this object, filled in):\n"
            f"{schema_str}\n"
        )

    def _build_repair_prompt(self, error_message: str, last_raw: str) -> str:
        schema_str = json.dumps(self._OUTPUT_SCHEMA, indent=2, ensure_ascii=True)
        return (
            "You are a playlist strategy planner for a music recommendation system.\n"
            "Your previous output was INVALID due to the following error:\n"
            f"{error_message}\n\n"
            "PREVIOUS OUTPUT (for reference):\n"
            f"{last_raw}\n\n"
            "Fix the output. STRICT REQUIREMENTS:\n"
            "- Output JSON ONLY (no markdown, no prose).\n"
            "- Return ONLY the corrected JSON object. No other text.\n"
            "- Include ALL required keys; DO NOT add extra keys.\n"
            "- All float weights must be in [0.0, 1.0].\n"
            "- The four primary weights (semantic, relational, soft_preference, novelty) must sum ~1.0.\n"
            "- diversity_strictness must be exactly one of: 'low', 'medium', 'high'.\n"
            "- broaden_if_low_recall must be a JSON boolean.\n"
            "- rationale must be a non-empty string.\n\n"
            "OUTPUT SCHEMA (return this object, filled in):\n"
            f"{schema_str}\n"
        )

    def _build_user_message(self, user_prompt: str, intent: VibeIntent) -> str:
        """Serialize the prompt + VibeIntent into the user turn sent to the LLM."""
        intent_dict = {
            "semantic_query": intent.semantic_query,
            "hard_constraints": intent.hard_constraints,
            "soft_preferences": intent.soft_preferences,
            "exclusions": intent.exclusions,
        }
        intent_str = json.dumps(intent_dict, indent=2, ensure_ascii=True)

        return (
            f'USER PROMPT:\n"{user_prompt}"\n\n'
            f"PARSED VIBE INTENT:\n{intent_str}\n\n"
            "Based on the above, output the PlaylistPlan JSON."
        )

    # -----------------------------------------------------------------------
    # Parsing and validation
    # -----------------------------------------------------------------------

    def _extract_json_block(self, text: str) -> str:
        text = (text or "").strip()

        # strip ```json fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in LLM output.")
        return text[start: end + 1]

    def _validate_and_parse_output(self, generated_text: str) -> PlaylistPlan:
        json_str = self._extract_json_block(generated_text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("LLM output must be a JSON object (dict).")

        required_keys = {
            "playlist_size",
            "semantic_weight",
            "relational_weight",
            "soft_preference_weight",
            "novelty_weight",
            "artist_repeat_penalty",
            "genre_concentration_penalty",
            "exclusion_penalty",
            "retrieval_limits",
            "broaden_if_low_recall",
            "diversity_strictness",
            "rationale",
        }
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Missing required keys in LLM output: {sorted(missing)}")

        # --- coerce numeric fields (LLM sometimes returns strings) ---
        for float_key in (
            "semantic_weight",
            "relational_weight",
            "soft_preference_weight",
            "novelty_weight",
            "artist_repeat_penalty",
            "genre_concentration_penalty",
            "exclusion_penalty",
        ):
            try:
                data[float_key] = float(data[float_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{float_key} could not be coerced to float: {exc}") from exc

        try:
            data["playlist_size"] = int(data["playlist_size"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"playlist_size could not be coerced to int: {exc}") from exc

        # --- retrieval_limits ---
        rl_raw = data.get("retrieval_limits")
        if not isinstance(rl_raw, dict):
            raise ValueError("retrieval_limits must be a JSON object.")
        try:
            rl = RetrievalLimits(
                n_vector=int(rl_raw.get("vector", 100)),
                n_relational=int(rl_raw.get("relational", 100)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"retrieval_limits values could not be coerced to int: {exc}") from exc

        # --- assemble and validate ---
        plan = PlaylistPlan(
            playlist_size=data["playlist_size"],
            semantic_weight=data["semantic_weight"],
            relational_weight=data["relational_weight"],
            soft_preference_weight=data["soft_preference_weight"],
            novelty_weight=data["novelty_weight"],
            artist_repeat_penalty=data["artist_repeat_penalty"],
            genre_concentration_penalty=data["genre_concentration_penalty"],
            exclusion_penalty=data["exclusion_penalty"],
            retrieval_limits=rl,
            broaden_if_low_recall=bool(data["broaden_if_low_recall"]),
            diversity_strictness=str(data["diversity_strictness"]).lower().strip(),
            rationale=str(data.get("rationale", "")).strip(),
        )
        plan.validate()
        return plan


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def plan_playlist(
    user_prompt: str,
    intent: VibeIntent,
    llm_client: Any,
    max_retries: int = 3,
) -> PlaylistPlan:
    """Functional wrapper around PlannerAgent for quick use in the pipeline."""
    return PlannerAgent(llm_client=llm_client, max_retries=max_retries).plan(
        user_prompt=user_prompt,
        intent=intent,
    )
