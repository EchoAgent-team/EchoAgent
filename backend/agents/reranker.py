"""
Deterministic reranking utilities for EchoAgent.

This module reranks fused retrieval candidates using transparent signals:
- base retrieval score from candidate fusion
- vector distance/rank
- soft preference matches
- exclusion penalties

It is designed to run as a LangGraph node after candidate_fusion_node.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

from .vibe_intent import VibeIntent


DEFAULT_SOFT_PREFERENCE_WEIGHT = 0.4
DEFAULT_SEMANTIC_WEIGHT = 0.25
DEFAULT_EXCLUSION_PENALTY = 2.0


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _tokens(values: Iterable[Any]) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        text = _clean_text(value)
        if text:
            out.add(text)
    return out


def _candidate_metadata(candidate: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    for key in ("metadata", "relational_candidate"):
        value = candidate.get(key)
        if isinstance(value, dict):
            metadata.update(value)

    vector_candidate = candidate.get("vector_candidate")
    if isinstance(vector_candidate, dict):
        vector_metadata = vector_candidate.get("metadata")
        if isinstance(vector_metadata, dict):
            metadata.update(vector_metadata)

    return metadata


def _candidate_terms(candidate: Dict[str, Any]) -> Set[str]:
    metadata = _candidate_metadata(candidate)
    terms: Set[str] = set()

    for key in ("title", "artist_name", "seed_genre", "album_title"):
        value = metadata.get(key)
        if value:
            terms.add(_clean_text(value))

    tags = metadata.get("top_tags_json")
    if isinstance(tags, dict):
        terms.update(_tokens(tags.keys()))
    elif isinstance(tags, list):
        terms.update(_tokens(tags))
    elif isinstance(tags, str):
        terms.update(_tokens(re.split(r"[|,]", tags)))

    vector_candidate = candidate.get("vector_candidate")
    if isinstance(vector_candidate, dict):
        vector_metadata = vector_candidate.get("metadata") or {}
        for key in ("genres_csv", "tags_csv"):
            value = vector_metadata.get(key)
            if isinstance(value, str):
                terms.update(_tokens(re.split(r"[|,]", value)))

    return {term for term in terms if term}


def _energy_bucket(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return _clean_text(value)

    try:
        energy = float(value)
    except (TypeError, ValueError):
        return ""

    if energy < 0.4:
        return "low"
    if energy < 0.7:
        return "medium"
    return "high"


def _semantic_score(candidate: Dict[str, Any]) -> float:
    vector_candidate = candidate.get("vector_candidate")
    if not isinstance(vector_candidate, dict):
        return 0.0

    distance = vector_candidate.get("vector_distance")
    if isinstance(distance, (int, float)) and distance >= 0:
        return 1.0 / (1.0 + float(distance))

    rank = vector_candidate.get("vector_rank")
    if isinstance(rank, (int, float)) and rank > 0:
        return 1.0 / float(rank)

    return 0.0


def _soft_preference_score(
    candidate: Dict[str, Any],
    soft_preferences: Dict[str, Any],
) -> float:
    if not soft_preferences:
        return 0.0

    metadata = _candidate_metadata(candidate)
    candidate_terms = _candidate_terms(candidate)
    score = 0.0

    preferred_terms = []
    for key in (
        "moods",
        "themes",
        "genres_prefer",
        "artists_prefer",
        "tags",
        "era",
    ):
        preferred_terms.extend(_as_list(soft_preferences.get(key)))

    for term in _tokens(preferred_terms):
        if term in candidate_terms:
            score += 1.0

    preferred_energy = soft_preferences.get("energy")
    if preferred_energy:
        if _energy_bucket(metadata.get("energy")) == _clean_text(preferred_energy):
            score += 1.0

    preferred_tempo = soft_preferences.get("tempo_bpm")
    if preferred_tempo is not None and metadata.get("tempo") is not None:
        try:
            diff = abs(float(metadata["tempo"]) - float(preferred_tempo))
            score += max(0.0, 1.0 - diff / 60.0)
        except (TypeError, ValueError):
            pass

    return score


def _exclusion_penalty(
    candidate: Dict[str, Any],
    exclusions: Dict[str, Any],
) -> float:
    if not exclusions:
        return 0.0

    candidate_terms = _candidate_terms(candidate)
    penalty = 0.0

    excluded_terms = []
    for key in ("genres_exclude", "artists_exclude", "tags_exclude"):
        excluded_terms.extend(_as_list(exclusions.get(key)))

    for term in _tokens(excluded_terms):
        if term in candidate_terms:
            penalty += 1.0

    return penalty


def rerank_candidates(
    candidates: List[Dict[str, Any]],
    intent: VibeIntent,
    playlist_plan: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    if not isinstance(intent, VibeIntent):
        raise TypeError("intent must be a VibeIntent")

    playlist_plan = playlist_plan or {}

    soft_weight = playlist_plan.get(
        "soft_preference_weight",
        DEFAULT_SOFT_PREFERENCE_WEIGHT,
    )
    semantic_weight = playlist_plan.get(
        "semantic_weight",
        DEFAULT_SEMANTIC_WEIGHT,
    )
    exclusion_weight = playlist_plan.get(
        "exclusion_penalty",
        DEFAULT_EXCLUSION_PENALTY,
    )

    ranked = []

    for candidate in candidates:
        base_score = candidate.get("score", candidate.get("retrieval_score", 0.0))
        semantic_score = _semantic_score(candidate)
        soft_score = _soft_preference_score(candidate, intent.soft_preferences)
        exclusion_penalty = _exclusion_penalty(candidate, intent.exclusions)

        final_score = (
            float(base_score)
            + semantic_weight * semantic_score
            + soft_weight * soft_score
            - exclusion_weight * exclusion_penalty
        )

        ranked.append(
            {
                **candidate,
                "score": final_score,
                "retrieval_score": candidate.get("retrieval_score", base_score),
                "ranking_debug": {
                    "base_score": base_score,
                    "semantic_score": semantic_score,
                    "soft_preference_score": soft_score,
                    "exclusion_penalty": exclusion_penalty,
                },
            }
        )

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def reranker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    raw_plan = state.get("playlist_plan")
    ranked_candidates = rerank_candidates(
        candidates=state.get("fused_candidates", []),
        intent=state["intent"],
        playlist_plan=raw_plan.to_dict() if raw_plan is not None else {},
    )

    return {
        "ranked_candidates": ranked_candidates,
    }
