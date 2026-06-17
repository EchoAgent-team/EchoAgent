"""
Playlist Builder Agent for EchoAgent.

Two-stage pipeline:
  1. Code  — trims ranked_candidates to a manageable pool (top POOL_SIZE by score)
             and slims each candidate to the fields the LLM needs.
  2. LLM   — selects playlist_size tracks from the pool and orders them,
             choosing an energy arc that fits the original prompt context.

The LLM has final selection power (pool → playlist_size), so it can make
curatorial trade-offs (artist variety, genre spread, energy shape) that
deterministic scoring alone cannot capture.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set

from backend.agents.vibe_intent import VibeIntent
from backend.agents.planner_agent import PlaylistPlan


POOL_SIZE = 50
DEFAULT_ENERGY_ARC = (
    "start with medium-energy tracks, build toward high energy in the middle, "
    "then wind down with calmer low-energy tracks at the end"
)


# ---------------------------------------------------------------------------
# Candidate slimming helpers
# ---------------------------------------------------------------------------

def _energy_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        return value.strip().lower()
    try:
        e = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if e < 0.4:
        return "low"
    if e < 0.7:
        return "medium"
    return "high"


def _top_tags(top_tags_json: Any, n: int = 10) -> List[str]:
    if not isinstance(top_tags_json, dict):
        return []
    sorted_tags = sorted(top_tags_json.items(), key=lambda kv: kv[1], reverse=True)
    return [tag for tag, _ in sorted_tags[:n]]


def _slim_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a full ranked candidate to just the fields the LLM needs."""
    metadata = candidate.get("metadata") or {}

    seed_genre = metadata.get("seed_genre")
    if not seed_genre:
        genres_csv = metadata.get("genres_csv", "")
        if genres_csv:
            seed_genre = genres_csv.split("|")[0]

    return {
        "track_id": candidate.get("track_id"),
        "title": metadata.get("title"),
        "artist_name": metadata.get("artist_name"),
        "seed_genre": seed_genre,
        "top_tags": _top_tags(metadata.get("top_tags_json")),
        "energy": _energy_bucket(metadata.get("energy")),
        "score": round(float(candidate.get("score", 0.0)), 4),
    }


def _artist_repeat_limit(plan: PlaylistPlan) -> int:
    """
    Convert the plan's penalty + strictness into a concrete max-tracks-per-artist
    integer for the LLM prompt. A float penalty is too abstract for an LLM to
    reason about directly.
    """
    if plan.diversity_strictness == "high" or plan.artist_repeat_penalty >= 0.3:
        return 1
    if plan.diversity_strictness == "low" or plan.artist_repeat_penalty <= 0.1:
        return 3
    return 2


# ---------------------------------------------------------------------------
# PlaylistBuilderAgent
# ---------------------------------------------------------------------------

class PlaylistBuilderAgent:
    """
    Playlist Builder Agent — selects and orders the final playlist from
    a pre-scored, pre-trimmed candidate pool.

    Input:
        user_prompt (str)             — original natural-language request
        intent (VibeIntent)           — structured intent from PromptParser
        playlist_plan (PlaylistPlan)  — scoring strategy from PlannerAgent
        pool (List[Dict])             — slimmed top-POOL_SIZE candidates

    Output:
        List[str]                     — ordered track_ids (length = playlist_size)
    """

    _OUTPUT_SCHEMA: Dict[str, Any] = {
        "selected_track_ids": (
            "list of strings: ordered track IDs for the final playlist, "
            "length must equal playlist_size"
        ),
        "energy_arc": (
            "string: description of the energy arc you chose "
            "(e.g. 'builds gradually to a peak then winds down') "
            "and why it fits the prompt"
        ),
        "rationale": (
            "string: 1-2 sentences on key artist/genre/energy decisions made"
        ),
    }

    _OUTPUT_RULES = [
        "Output JSON ONLY — no markdown fences, no prose, no extra keys.",
        "selected_track_ids must be a JSON array of strings.",
        "Every track_id in selected_track_ids must come from the provided candidate list — do NOT invent IDs.",
        "selected_track_ids must contain exactly playlist_size entries.",
        "No duplicate track_ids.",
        "Order tracks so the energy sequence reflects the energy_arc you describe.",
        (
            "Energy arc default: start medium, build toward high in the middle, end low. "
            "Deviate from this default when the prompt clearly implies a different journey "
            "(e.g. 'late night wind-down' → gentle throughout; "
            "'morning run' → steady build; 'pre-game hype' → high from the start)."
        ),
        "Respect artist_repeat_limit: do not include more than that many tracks from the same artist.",
        "Prefer genre variety across the playlist unless the prompt is narrow or single-style.",
        "energy_arc and rationale must be non-empty strings.",
    ]

    def __init__(self, llm_client: Any, max_retries: int = 3) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries
        self._system_prompt = self._build_system_prompt()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def build(
        self,
        user_prompt: str,
        intent: VibeIntent,
        playlist_plan: PlaylistPlan,
        pool: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Select and order the final playlist from the candidate pool.

        Args:
            user_prompt:   Original natural-language request.
            intent:        VibeIntent from PromptParser.
            playlist_plan: PlaylistPlan from PlannerAgent.
            pool:          Slimmed top-POOL_SIZE candidates (already sorted by score).

        Returns:
            Ordered list of track_ids, length == playlist_plan.playlist_size.

        Raises:
            ValueError: If all retries are exhausted without a valid playlist.
        """
        pool_ids: Set[str] = {c["track_id"] for c in pool if c.get("track_id")}
        user_message = self._build_user_message(user_prompt, intent, playlist_plan, pool)

        last_err: Optional[str] = None
        last_raw: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            if attempt == 1:
                system_prompt = self._system_prompt
            else:
                system_prompt = self._build_repair_prompt(
                    error_message=last_err or "unknown_validation_error",
                    last_raw=last_raw or "",
                )

            raw = self.llm_client.generate(
                system_prompt=system_prompt,
                user_input=user_message,
            )
            last_raw = raw

            try:
                track_ids = self._validate_and_parse_output(
                    raw,
                    pool_ids=pool_ids,
                    playlist_size=playlist_plan.playlist_size,
                )
                return track_ids
            except (ValueError, TypeError) as exc:
                last_err = str(exc)
                continue

        raise ValueError(
            f"PlaylistBuilderAgent failed to produce a valid playlist after "
            f"{self.max_retries} attempts. Last error: {last_err}"
        )

    # -----------------------------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        schema_str = json.dumps(self._OUTPUT_SCHEMA, indent=2, ensure_ascii=True)
        rules_str = "\n".join(f"- {r}" for r in self._OUTPUT_RULES)
        return (
            "You are an expert music playlist curator for a recommendation system.\n"
            "You will receive a user's music request, their structured intent, the playlist "
            "strategy, and a pool of pre-scored candidate tracks.\n"
            "Your job is to select the final tracks and arrange them into an ordered playlist.\n\n"
            "ENERGY ARC GUIDANCE:\n"
            f"Default arc: {DEFAULT_ENERGY_ARC}.\n"
            "Before applying the default, read the user's original prompt carefully. "
            "If the prompt implies a different emotional or energy journey, "
            "choose an arc that fits that context instead. "
            "State your chosen arc and your reasoning in the energy_arc field.\n\n"
            "STRICT OUTPUT REQUIREMENTS:\n"
            "1) Output JSON ONLY (no markdown, no prose).\n"
            "2) Output must match the schema keys EXACTLY.\n"
            "3) All track_ids must come from the provided candidate list — never invent IDs.\n"
            "4) Length of selected_track_ids must equal playlist_size exactly.\n"
            "5) energy_arc and rationale must be non-empty strings.\n\n"
            f"OUTPUT RULES:\n{rules_str}\n\n"
            "OUTPUT SCHEMA (return this object, filled in):\n"
            f"{schema_str}\n"
        )

    def _build_repair_prompt(self, error_message: str, last_raw: str) -> str:
        schema_str = json.dumps(self._OUTPUT_SCHEMA, indent=2, ensure_ascii=True)
        return (
            "You are a music playlist curator for a recommendation system.\n"
            "Your previous output was INVALID due to the following error:\n"
            f"{error_message}\n\n"
            "PREVIOUS OUTPUT (for reference):\n"
            f"{last_raw}\n\n"
            "Fix the output. STRICT REQUIREMENTS:\n"
            "- Output JSON ONLY (no markdown, no prose).\n"
            "- Return ONLY the corrected JSON object. No other text.\n"
            "- selected_track_ids must be a JSON array of strings.\n"
            "- Every track_id must come from the provided candidate list — do NOT invent IDs.\n"
            "- selected_track_ids must contain exactly playlist_size entries, no duplicates.\n"
            "- energy_arc and rationale must be non-empty strings.\n\n"
            "OUTPUT SCHEMA (return this object, filled in):\n"
            f"{schema_str}\n"
        )

    def _build_user_message(
        self,
        user_prompt: str,
        intent: VibeIntent,
        playlist_plan: PlaylistPlan,
        pool: List[Dict[str, Any]],
    ) -> str:
        intent_dict = {
            "semantic_query": intent.semantic_query,
            "soft_preferences": intent.soft_preferences,
            "exclusions": intent.exclusions,
        }
        strategy_dict = {
            "playlist_size": playlist_plan.playlist_size,
            "diversity_strictness": playlist_plan.diversity_strictness,
            "artist_repeat_limit": _artist_repeat_limit(playlist_plan),
            "planner_rationale": playlist_plan.rationale,
        }
        return (
            f'USER PROMPT:\n"{user_prompt}"\n\n'
            f"VIBE INTENT:\n{json.dumps(intent_dict, indent=2, ensure_ascii=True)}\n\n"
            f"PLAYLIST STRATEGY:\n{json.dumps(strategy_dict, indent=2, ensure_ascii=True)}\n\n"
            f"CANDIDATE POOL ({len(pool)} tracks, ordered by relevance score):\n"
            f"{json.dumps(pool, indent=2, ensure_ascii=True)}\n\n"
            f"Select exactly {playlist_plan.playlist_size} tracks and order them. "
            "Output the PlaylistBuilder JSON."
        )

    # -----------------------------------------------------------------------
    # Parsing and validation
    # -----------------------------------------------------------------------

    def _extract_json_block(self, text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in LLM output.")
        return text[start: end + 1]

    def _validate_and_parse_output(
        self,
        generated_text: str,
        pool_ids: Set[str],
        playlist_size: int,
    ) -> List[str]:
        json_str = self._extract_json_block(generated_text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("LLM output must be a JSON object (dict).")

        track_ids = data.get("selected_track_ids")
        if not isinstance(track_ids, list):
            raise ValueError("selected_track_ids must be a JSON array.")

        if len(track_ids) != playlist_size:
            raise ValueError(
                f"selected_track_ids has {len(track_ids)} entries, "
                f"expected exactly {playlist_size}."
            )

        seen: Set[str] = set()
        for tid in track_ids:
            if not isinstance(tid, str):
                raise ValueError(
                    f"All track_ids must be strings, got {type(tid).__name__}: {tid!r}"
                )
            if tid not in pool_ids:
                raise ValueError(
                    f"track_id {tid!r} is not in the candidate pool. "
                    "Only use track_ids from the provided list."
                )
            if tid in seen:
                raise ValueError(f"Duplicate track_id in selected_track_ids: {tid!r}")
            seen.add(tid)

        for field in ("energy_arc", "rationale"):
            v = data.get(field)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"{field} must be a non-empty string.")

        return track_ids


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def build_playlist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node interface for PlaylistBuilderAgent.

    Code path:
      1. Trim ranked_candidates to top POOL_SIZE.
      2. Slim each candidate to LLM-friendly fields.

    LLM path:
      3. PlaylistBuilderAgent selects + orders playlist_size tracks.

    Code path:
      4. Reconstruct full candidate dicts in the LLM's chosen order.
    """
    ranked: List[Dict[str, Any]] = state.get("ranked_candidates", [])
    plan: PlaylistPlan = state["playlist_plan"]
    intent: VibeIntent = state["intent"]
    agent: PlaylistBuilderAgent = state["playlist_builder_agent"]

    pool_full = ranked[:POOL_SIZE]
    pool_slim = [_slim_candidate(c) for c in pool_full]

    ordered_ids = agent.build(
        user_prompt=state["user_prompt"],
        intent=intent,
        playlist_plan=plan,
        pool=pool_slim,
    )

    id_to_candidate = {c["track_id"]: c for c in pool_full}
    playlist = [id_to_candidate[tid] for tid in ordered_ids if tid in id_to_candidate]

    return {"playlist": playlist}
