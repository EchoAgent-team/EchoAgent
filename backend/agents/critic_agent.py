"""
Critic Agent for EchoAgent.

Reviews the generated playlist against the original prompt and PlaylistPlan,
then returns a CriticReport: accept, reason, and suggested_adjustments
for the planner to incorporate on a retry.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from backend.agents.vibe_intent import VibeIntent
from backend.agents.planner_agent import PlaylistPlan


def _slim_track(track: Dict[str, Any], position: int) -> Dict[str, Any]:
    metadata = track.get("metadata") or {}
    genre = metadata.get("seed_genre") or (metadata.get("genres_csv") or "").split("|")[0] or None
    tags_raw = metadata.get("top_tags_json") or {}
    top_tags = [t for t, _ in sorted(tags_raw.items(), key=lambda kv: kv[1], reverse=True)[:5]] if isinstance(tags_raw, dict) else []
    return {
        "position": position + 1,
        "track_id": track.get("track_id"),
        "title": metadata.get("title"),
        "artist": metadata.get("artist_name"),
        "genre": genre,
        "tags": top_tags,
        "energy": metadata.get("energy"),
    }


class CriticAgent:
    """
    Critic Agent - evaluates whether the playlist satisfies the user's request.

    Returns a CriticReport with accept/reason/suggested_adjustments.
    Suggested adjustments are PlaylistPlan field overrides the planner can use on retry.
    """

    _OUTPUT_SCHEMA = {
        "accept": "bool: true if the playlist satisfies the request, false if it needs a retry",
        "reason": "string: 1-2 sentence explanation of your decision",
        "suggested_adjustments": (
            "object: PlaylistPlan fields to change on retry — empty {} if accept=true. "
            "Allowed keys: semantic_weight, relational_weight, soft_preference_weight, "
            "novelty_weight, artist_repeat_penalty, genre_concentration_penalty, "
            "exclusion_penalty (all floats in [0.0, 1.0]), diversity_strictness ('low'|'medium'|'high')"
        ),
    }

    _OUTPUT_RULES = [
        "Output JSON ONLY — no markdown fences, no prose, no extra keys.",
        "accept must be a JSON boolean (true or false).",
        "reason must be a non-empty string.",
        "suggested_adjustments must be a JSON object ({} if accept=true).",
        "Only include keys in suggested_adjustments that need to change.",
        "If you change any of the four weights (semantic, relational, soft_preference, novelty), "
        "adjust all four so they still sum to ~1.0.",
        "All float adjustments must be in [0.0, 1.0].",
        "diversity_strictness must be exactly 'low', 'medium', or 'high'.",
        # when to accept
        "Accept if: the playlist vibe broadly fits the request, hard constraints are respected, "
        "and no excluded artists/genres appear.",
        "Do NOT reject for minor taste differences — only reject clear violations.",
        # adjustment guidance
        "If exclusions were violated: raise exclusion_penalty (0.6–0.9).",
        "If the vibe is off: raise semantic_weight, lower relational_weight.",
        "If too many tracks from one artist: raise artist_repeat_penalty, set diversity_strictness='high'.",
        "If hard constraints were ignored: raise relational_weight.",
    ]

    def __init__(self, llm_client: Any, max_retries: int = 3) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries
        self._system_prompt = self._build_system_prompt()

    def critique(
        self,
        user_prompt: str,
        intent: VibeIntent,
        playlist_plan: PlaylistPlan,
        playlist: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate the playlist and return a CriticReport dict."""
        user_message = self._build_user_message(user_prompt, intent, playlist_plan, playlist)

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

            raw = self.llm_client.generate(system_prompt=system_prompt, user_input=user_message)
            last_raw = raw

            try:
                return self._validate_and_parse_output(raw)
            except (ValueError, TypeError) as exc:
                last_err = str(exc)
                continue

        raise ValueError(
            f"CriticAgent failed after {self.max_retries} attempts. Last error: {last_err}"
        )

    def _build_system_prompt(self) -> str:
        schema_str = json.dumps(self._OUTPUT_SCHEMA, indent=2, ensure_ascii=True)
        rules_str = "\n".join(f"- {r}" for r in self._OUTPUT_RULES)
        return (
            "You are a quality-control critic for a music recommendation system.\n"
            "You will receive a user's music request, their structured intent, the playlist "
            "strategy that was used, and the resulting playlist.\n"
            "Your job is to decide whether the playlist satisfies the request, and if not, "
            "provide specific adjustments for the planner to use on a retry.\n\n"
            "EVALUATION PRIORITIES:\n"
            "1. Exclusion violations — any excluded artist or genre present → reject.\n"
            "2. Hard constraint coverage — explicit constraints clearly ignored → reject.\n"
            "3. Vibe/mood fit — does the playlist feel like the right response?\n"
            "4. Diversity — is artist/genre spread appropriate for diversity_strictness?\n\n"
            "STRICT OUTPUT REQUIREMENTS:\n"
            "1) Output JSON ONLY (no markdown, no prose).\n"
            "2) Output must match the schema keys EXACTLY.\n"
            "3) reason must be a non-empty string.\n"
            "4) suggested_adjustments must be {} if accept=true.\n\n"
            f"OUTPUT RULES:\n{rules_str}\n\n"
            "OUTPUT SCHEMA (return this object, filled in):\n"
            f"{schema_str}\n"
        )

    def _build_repair_prompt(self, error_message: str, last_raw: str) -> str:
        schema_str = json.dumps(self._OUTPUT_SCHEMA, indent=2, ensure_ascii=True)
        return (
            "You are a quality-control critic for a music recommendation system.\n"
            "Your previous output was INVALID due to the following error:\n"
            f"{error_message}\n\n"
            "PREVIOUS OUTPUT (for reference):\n"
            f"{last_raw}\n\n"
            "Fix the output. STRICT REQUIREMENTS:\n"
            "- Output JSON ONLY (no markdown, no prose).\n"
            "- accept must be a JSON boolean.\n"
            "- reason must be a non-empty string.\n"
            "- suggested_adjustments must be a JSON object ({} if accept=true).\n"
            "- All float values must be in [0.0, 1.0].\n"
            "- diversity_strictness must be exactly 'low', 'medium', or 'high'.\n\n"
            "OUTPUT SCHEMA (return this object, filled in):\n"
            f"{schema_str}\n"
        )

    def _build_user_message(
        self,
        user_prompt: str,
        intent: VibeIntent,
        playlist_plan: PlaylistPlan,
        playlist: List[Dict[str, Any]],
    ) -> str:
        intent_dict = {
            "semantic_query": intent.semantic_query,
            "hard_constraints": intent.hard_constraints,
            "soft_preferences": intent.soft_preferences,
            "exclusions": intent.exclusions,
        }
        strategy_dict = {
            "playlist_size": playlist_plan.playlist_size,
            "semantic_weight": playlist_plan.semantic_weight,
            "relational_weight": playlist_plan.relational_weight,
            "soft_preference_weight": playlist_plan.soft_preference_weight,
            "novelty_weight": playlist_plan.novelty_weight,
            "artist_repeat_penalty": playlist_plan.artist_repeat_penalty,
            "genre_concentration_penalty": playlist_plan.genre_concentration_penalty,
            "exclusion_penalty": playlist_plan.exclusion_penalty,
            "diversity_strictness": playlist_plan.diversity_strictness,
            "planner_rationale": playlist_plan.rationale,
        }
        slim_playlist = [_slim_track(t, i) for i, t in enumerate(playlist)]

        return (
            f'USER PROMPT:\n"{user_prompt}"\n\n'
            f"PARSED VIBE INTENT:\n{json.dumps(intent_dict, indent=2, ensure_ascii=True)}\n\n"
            f"PLAYLIST STRATEGY USED:\n{json.dumps(strategy_dict, indent=2, ensure_ascii=True)}\n\n"
            f"RESULTING PLAYLIST ({len(slim_playlist)} tracks):\n"
            f"{json.dumps(slim_playlist, indent=2, ensure_ascii=True)}\n\n"
            "Evaluate the playlist and output the CriticReport JSON."
        )

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

    def _validate_and_parse_output(self, generated_text: str) -> Dict[str, Any]:
        json_str = self._extract_json_block(generated_text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("LLM output must be a JSON object (dict).")

        if missing := {"accept", "reason", "suggested_adjustments"} - set(data.keys()):
            raise ValueError(f"Missing required keys: {sorted(missing)}")

        if not isinstance(data["accept"], bool):
            raise ValueError(f"accept must be a JSON boolean, got {type(data['accept']).__name__}")

        if not isinstance(data["reason"], str) or not data["reason"].strip():
            raise ValueError("reason must be a non-empty string.")

        if not isinstance(data["suggested_adjustments"], dict):
            raise ValueError("suggested_adjustments must be a JSON object.")

        _FLOAT_KEYS = {
            "semantic_weight", "relational_weight", "soft_preference_weight",
            "novelty_weight", "artist_repeat_penalty", "genre_concentration_penalty",
            "exclusion_penalty",
        }
        adj = data["suggested_adjustments"]
        for key, val in adj.items():
            if key == "diversity_strictness":
                if val not in {"low", "medium", "high"}:
                    raise ValueError(f"diversity_strictness must be 'low', 'medium', or 'high', got {val!r}")
            elif key in _FLOAT_KEYS:
                try:
                    adj[key] = float(val)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"suggested_adjustments.{key} must be a float: {exc}") from exc
                if not (0.0 <= adj[key] <= 1.0):
                    raise ValueError(f"suggested_adjustments.{key} must be in [0.0, 1.0], got {adj[key]}")
            else:
                allowed = sorted(_FLOAT_KEYS | {"diversity_strictness"})
                raise ValueError(f"suggested_adjustments has unsupported key {key!r}. Allowed: {allowed}")

        return {
            "accept": data["accept"],
            "reason": data["reason"].strip(),
            "suggested_adjustments": adj,
        }


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def critique_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agent: CriticAgent = state.get("critic_agent")  # type: ignore[assignment]

    if agent is None:
        return {
            "critic_report": {
                "accept": True,
                "reason": "no critic_agent in state — auto-accepted",
                "suggested_adjustments": {},
            },
            "retry_count": state.get("retry_count", 0) + 1,
        }

    report = agent.critique(
        user_prompt=state["user_prompt"],
        intent=state["intent"],
        playlist_plan=state["playlist_plan"],
        playlist=state.get("playlist", []),
    )

    return {
        "critic_report": report,
        "retry_count": state.get("retry_count", 0) + 1,
    }