from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class RecommendRequest(BaseModel):
    prompt: str


class PlaylistTrack(BaseModel):
    track_id: str
    title: Optional[str] = None
    artist_name: Optional[str] = None
    album_title: Optional[str] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    tags: list[str] = []
    score: float
    sources: list[str] = []


class CriticReport(BaseModel):
    accept: bool
    reason: str
    suggested_adjustments: dict[str, Any] = {}


class RecommendDebug(BaseModel):
    intent: dict[str, Any]
    plan: dict[str, Any]
    relational_candidate_count: int
    vector_candidate_count: int
    fused_candidate_count: int
    retry_count: int
    builder_fallback_used: bool = False
    builder_fallback_reason: Optional[str] = None
    critic_report: Optional[CriticReport] = None


class RecommendResponse(BaseModel):
    prompt: str
    playlist: list[PlaylistTrack]
    debug: RecommendDebug
