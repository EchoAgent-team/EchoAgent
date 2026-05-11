"""
Candidate fusion utilities for EchoAgent.

This module merges relational and vector retrieval outputs into one
track-keyed candidate list with transparent, deterministic scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List


RELATIONAL_WEIGHT = 1.0
VECTOR_WEIGHT = 1.0


def fuse_candidates(
    relational_candidates: List[Dict[str, Any]],
    vector_candidates: List[Dict[str, Any]],
    relational_weight: float = RELATIONAL_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
) -> List[Dict[str, Any]]:
    """
    Merge relational + vector candidates into one list.

    For now:
    - relational match = useful hard-filter signal
    - vector rank = semantic/vibe signal
    - tracks found in both should score higher
    """
    fused = {}

    for track in relational_candidates:
        track_id = track.get("track_id")
        if not track_id:
            continue

        fused[track_id] = {
            "track_id": track_id,
            "metadata": track,
            "sources": ["relational"],
            "relational_candidate": track,
            "vector_candidate": None,
            "retrieval_score": relational_weight,
        }

    for i, track in enumerate(vector_candidates):
        track_id = track.get("track_id")

        if not track_id and isinstance(track.get("metadata"), dict):
            track_id = track["metadata"].get("track_id")

        if not track_id:
            continue

        rank = track.get("vector_rank", i + 1)
        if not isinstance(rank, (int, float)) or rank <= 0:
            rank = i + 1

        vector_score = vector_weight * (1.0 / rank)

        if track_id in fused:
            if "vector" not in fused[track_id]["sources"]:
                fused[track_id]["sources"].append("vector")

            fused[track_id]["vector_candidate"] = track
            fused[track_id]["retrieval_score"] += vector_score

        else:
            fused[track_id] = {
                "track_id": track_id,
                "metadata": track.get("metadata", {}),
                "sources": ["vector"],
                "relational_candidate": None,
                "vector_candidate": track,
                "retrieval_score": vector_score,
            }

    return sorted(fused.values(), key=lambda x: x["retrieval_score"], reverse=True)


def candidate_fusion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    playlist_plan = state.get("playlist_plan", {})

    relational_weight = playlist_plan.get("relational_weight", RELATIONAL_WEIGHT)
    vector_weight = playlist_plan.get("semantic_weight", VECTOR_WEIGHT)

    fused_candidates = fuse_candidates(
        relational_candidates=state.get("relational_candidates", []),
        vector_candidates=state.get("vector_candidates", []),
        relational_weight=relational_weight,
        vector_weight=vector_weight,
    )

    return {
        "fused_candidates": fused_candidates,
        "fused_candidate_count": len(fused_candidates),
    }
