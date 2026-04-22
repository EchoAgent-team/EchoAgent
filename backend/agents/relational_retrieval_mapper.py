"""
Utilities for translating VibeIntent objects into deterministic retrieval inputs.

This module is the bridge between:
- language understanding (`VibeIntent`)
- relational retrieval (`backend.data.db.query_tracks`)
- later vector retrieval (via semantic_query)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .vibe_intent import VibeIntent


ENERGY_BUCKETS: Dict[str, Tuple[float, float]] = {
    "low": (0.0, 0.4),
    "medium": (0.4, 0.7),
    "high": (0.7, 1.0),
}

DANCEABILITY_BUCKETS: Dict[str, Tuple[float, float]] = {
    "low": (0.0, 0.4),
    "medium": (0.4, 0.7),
    "high": (0.7, 1.0),
}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _pick_first_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()

    return None


def _extract_min_max(value: Any) -> Tuple[Optional[float], Optional[float]]:
    """
    Accepts either:
    - {"min": x, "max": y}
    - [x, y]
    - (x, y)

    Returns:
        (min_value, max_value)
    """
    if isinstance(value, dict):
        return value.get("min"), value.get("max")

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]

    return None, None


def _apply_bucket_or_range(
    filters: Dict[str, Any],
    raw_value: Any,
    bucket_map: Dict[str, Tuple[float, float]],
    min_key: str,
    max_key: str,
) -> None:
    """
    Handles either bucket labels like 'low'/'medium'/'high'
    or explicit numeric ranges like {'min': 0.3, 'max': 0.8}.
    """
    if isinstance(raw_value, str):
        bucket = bucket_map.get(raw_value.strip().lower())
        if bucket:
            filters[min_key], filters[max_key] = bucket
        return

    min_val, max_val = _extract_min_max(raw_value)
    if min_val is not None:
        filters[min_key] = min_val
    if max_val is not None:
        filters[max_key] = max_val


def vibe_intent_to_relational_filters(intent: VibeIntent) -> Dict[str, Any]:
    """
    Convert VibeIntent.hard_constraints into the filter dictionary expected by
    backend.data.db.query_tracks(...).

    Supported output keys:
        artist_id
        min_year
        max_year
        seed_genre
        min_energy
        max_energy
        min_danceability
        max_danceability
        min_tempo
        max_tempo
    """
    hard = intent.hard_constraints or {}
    filters: Dict[str, Any] = {}

    # Exact artist match, if provided directly.
    if "artist_id" in hard:
        artist_id = _pick_first_string(hard.get("artist_id"))
        if artist_id:
            filters["artist_id"] = artist_id

    # Genre mapping:
    # Prefer seed_genre if present.
    # Fall back to first value from genres_include / genre / genres.
    if "seed_genre" in hard:
        seed_genre = _pick_first_string(hard.get("seed_genre"))
        if seed_genre:
            filters["seed_genre"] = seed_genre
    else:
        genre_candidates = (
            hard.get("genres_include")
            or hard.get("genre")
            or hard.get("genres")
        )
        seed_genre = _pick_first_string(genre_candidates)
        if seed_genre:
            filters["seed_genre"] = seed_genre

    # Era/year mapping:
    # Supports:
    # - era: {"min": 1990, "max": 1999}
    # - year: {"min": 1990, "max": 1999}
    # - year_min / year_max direct keys
    if "era" in hard:
        min_year, max_year = _extract_min_max(hard["era"])
        if min_year is not None:
            filters["min_year"] = int(min_year)
        if max_year is not None:
            filters["max_year"] = int(max_year)

    if "year" in hard:
        min_year, max_year = _extract_min_max(hard["year"])
        if min_year is not None:
            filters["min_year"] = int(min_year)
        if max_year is not None:
            filters["max_year"] = int(max_year)

    if "year_min" in hard and hard["year_min"] is not None:
        filters["min_year"] = int(hard["year_min"])
    if "year_max" in hard and hard["year_max"] is not None:
        filters["max_year"] = int(hard["year_max"])

    # Tempo mapping:
    # Supports:
    # - tempo_bpm: {"min": 100, "max": 140}
    # - tempo: {"min": 100, "max": 140}
    # - min_tempo / max_tempo direct keys
    if "tempo_bpm" in hard:
        min_tempo, max_tempo = _extract_min_max(hard["tempo_bpm"])
        if min_tempo is not None:
            filters["min_tempo"] = float(min_tempo)
        if max_tempo is not None:
            filters["max_tempo"] = float(max_tempo)

    if "tempo" in hard and not isinstance(hard["tempo"], str):
        min_tempo, max_tempo = _extract_min_max(hard["tempo"])
        if min_tempo is not None:
            filters["min_tempo"] = float(min_tempo)
        if max_tempo is not None:
            filters["max_tempo"] = float(max_tempo)

    if "min_tempo" in hard and hard["min_tempo"] is not None:
        filters["min_tempo"] = float(hard["min_tempo"])
    if "max_tempo" in hard and hard["max_tempo"] is not None:
        filters["max_tempo"] = float(hard["max_tempo"])

    # Energy mapping:
    # Supports:
    # - energy: "low" / "medium" / "high"
    # - energy: {"min": 0.2, "max": 0.8}
    # - min_energy / max_energy direct keys
    if "energy" in hard:
        _apply_bucket_or_range(
            filters=filters,
            raw_value=hard["energy"],
            bucket_map=ENERGY_BUCKETS,
            min_key="min_energy",
            max_key="max_energy",
        )

    if "min_energy" in hard and hard["min_energy"] is not None:
        filters["min_energy"] = float(hard["min_energy"])
    if "max_energy" in hard and hard["max_energy"] is not None:
        filters["max_energy"] = float(hard["max_energy"])

    # Danceability mapping:
    # Supports:
    # - danceability: "low" / "medium" / "high"
    # - danceability: {"min": ..., "max": ...}
    # - min_danceability / max_danceability direct keys
    if "danceability" in hard:
        _apply_bucket_or_range(
            filters=filters,
            raw_value=hard["danceability"],
            bucket_map=DANCEABILITY_BUCKETS,
            min_key="min_danceability",
            max_key="max_danceability",
        )

    if "min_danceability" in hard and hard["min_danceability"] is not None:
        filters["min_danceability"] = float(hard["min_danceability"])
    if "max_danceability" in hard and hard["max_danceability"] is not None:
        filters["max_danceability"] = float(hard["max_danceability"])

    return filters

# #### Might need to be moved to a higher level orchestration layer if we want to do hybrid retrieval with both relational and vector sources in parallel. ####
# def build_retrieval_payload(intent: VibeIntent) -> Dict[str, Any]:
#     """
#     Convenience helper for later hybrid retrieval.

#     Returns a payload with:
#     - db_filters: for relational retrieval
#     - semantic_query: for Chroma/vector retrieval
#     - soft_preferences: for ranking
#     - exclusions: for filtering/reranking
#     """
#     return {
#         "db_filters": vibe_intent_to_relational_filters(intent),
#         "semantic_query": intent.semantic_query,
#         "soft_preferences": intent.soft_preferences,
#         "exclusions": intent.exclusions,
#     }