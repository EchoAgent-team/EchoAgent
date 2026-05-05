from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from backend.agents.vibe_intent import VibeIntent


class PlaylistPlan(TypedDict, total=False):
    """Planner output that controls retrieval, ranking, and playlist shape."""

    playlist_size: int
    n_vector: int
    n_relational: int

    semantic_weight: float
    relational_weight: float
    soft_preference_weight: float
    novelty_weight: float

    artist_repeat_penalty: float
    genre_concentration_penalty: float
    exclusion_penalty: float


class CriticReport(TypedDict, total=False):
    """Critic output for accepting or revising a playlist."""

    accept: bool
    reason: str
    suggested_adjustments: Dict[str, Any]


class PlaylistGraphState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    user_prompt: str

    intent: VibeIntent
    playlist_plan: PlaylistPlan
    critic_report: CriticReport

    relational_filters: Dict[str, Any]
    relational_candidates: List[Dict[str, Any]]
    relational_candidate_count: int

    vector_query: str
    vector_candidates: List[Dict[str, Any]]
    vector_candidate_count: int

    fused_candidates: List[Dict[str, Any]]
    ranked_candidates: List[Dict[str, Any]]
    playlist: List[Dict[str, Any]]

    database_url: Optional[str]
    chroma_persist_directory: Optional[str]
    retry_count: int
    max_retries: int

    trace: List[Dict[str, Any]]
    errors: List[str]
