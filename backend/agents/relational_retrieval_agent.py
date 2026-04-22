"""
Relational retrieval agent for EchoAgent.

This module is a thin orchestration layer between:
- parsed/validated VibeIntent objects
- relational retrieval mapping
- SQLite / SQLAlchemy querying

It is designed to be flexible in terms of where it can be called from:
- directly in backend services
- inside API routes
- as a LangGraph node
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.data.db import query_tracks
from .vibe_intent import VibeIntent
from .relational_retrieval_mapper import vibe_intent_to_relational_filters


def retrieve_relational_candidates(
    intent: VibeIntent,
    limit: int = 100,
    database_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve candidate tracks from the relational database using the
    SQL-friendly subset of VibeIntent.hard_constraints.

    Args:
        intent:
            Parsed and validated VibeIntent.
        limit:
            Maximum number of candidates to return.
        database_url:
            Optional override for the DB URL.

    Returns:
        List of serialized track dictionaries.
    """
    filters = vibe_intent_to_relational_filters(intent)

    candidates = query_tracks(
        filters=filters,
        limit=limit,
        database_url=database_url,
    )

    return candidates


def retrieve_relational_bundle(
    intent: VibeIntent,
    limit: int = 100,
    database_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper that returns both the filters used and the retrieved
    candidates. Useful for debugging, evaluation, and later graph state updates.

    Returns:
        {
            "relational_filters": ...,
            "relational_candidates": ...,
            "candidate_count": ...
        }
    """
    filters = vibe_intent_to_relational_filters(intent)

    candidates = query_tracks(
        filters=filters,
        limit=limit,
        database_url=database_url,
    )

    return {
        "relational_filters": filters,
        "relational_candidates": candidates,
        "candidate_count": len(candidates),
    }


def relational_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-friendly node function.

    Expected input state keys:
        - "intent": VibeIntent
        - optional "relational_limit": int
        - optional "database_url": str

    Returns state updates:
        - "relational_filters"
        - "relational_candidates"
        - "relational_candidate_count"
    """
    intent = state["intent"]
    limit = state.get("relational_limit", 100)
    database_url = state.get("database_url")

    if not isinstance(intent, VibeIntent):
        raise TypeError("state['intent'] must be a VibeIntent instance")

    filters = vibe_intent_to_relational_filters(intent)

    candidates = query_tracks(
        filters=filters,
        limit=limit,
        database_url=database_url,
    )

    return {
        "relational_filters": filters,
        "relational_candidates": candidates,
        "relational_candidate_count": len(candidates),
    }