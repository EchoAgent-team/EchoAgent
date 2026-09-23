from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.agents.planner_agent import PlannerAgent
from backend.agents.playlist_builder import PlaylistBuilderAgent
from backend.agents.playlist_graph import run_playlist_graph
from backend.agents.prompt_parser import PromptParser
from backend.agents.vibe_intent import VibeIntent
from backend.api.schemas import (
    CriticReport,
    PlaylistTrack,
    RecommendDebug,
    RecommendRequest,
    RecommendResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependencies — read the singletons built once at startup (see main.py's
# lifespan). Routed through Depends() rather than module globals so tests can
# swap them out via app.dependency_overrides without touching real Groq/DB.
# ---------------------------------------------------------------------------

def get_llm_client(request: Request) -> Any:
    client = getattr(request.app.state, "llm_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="LLM client is not configured (missing GROQ_API_KEY).")
    return client


def get_planner_agent(request: Request) -> PlannerAgent:
    agent = getattr(request.app.state, "planner_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Planner agent is not configured (missing GROQ_API_KEY).")
    return agent


def get_playlist_builder_agent(request: Request) -> PlaylistBuilderAgent:
    agent = getattr(request.app.state, "playlist_builder_agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Playlist builder agent is not configured (missing GROQ_API_KEY).")
    return agent


def get_prompt_schema_path(request: Request) -> str:
    return request.app.state.prompt_schema_path


def get_database_url(request: Request) -> Optional[str]:
    return getattr(request.app.state, "database_url", None)


def get_chroma_persist_directory(request: Request) -> Optional[str]:
    return getattr(request.app.state, "chroma_persist_directory", None)


# ---------------------------------------------------------------------------
# final_state -> RecommendResponse mapping
# ---------------------------------------------------------------------------

def _extract_tags(top_tags_json: Any, limit: int = 10) -> List[str]:
    if not isinstance(top_tags_json, dict):
        return []
    sorted_tags = sorted(top_tags_json.items(), key=lambda kv: kv[1], reverse=True)
    return [tag for tag, _ in sorted_tags[:limit]]


def _extract_genre(metadata: Dict[str, Any]) -> Optional[str]:
    seed_genre = metadata.get("seed_genre")
    if seed_genre:
        return seed_genre
    genres_csv = metadata.get("genres_csv") or ""
    if genres_csv:
        return genres_csv.split("|")[0] or None
    return None


def _to_playlist_track(candidate: Dict[str, Any]) -> PlaylistTrack:
    metadata = candidate.get("metadata") or {}
    return PlaylistTrack(
        track_id=candidate.get("track_id"),
        title=metadata.get("title"),
        artist_name=metadata.get("artist_name"),
        album_title=metadata.get("album_title"),
        year=metadata.get("year"),
        genre=_extract_genre(metadata),
        tags=_extract_tags(metadata.get("top_tags_json")),
        score=float(candidate.get("score", 0.0)),
        sources=list(candidate.get("sources") or []),
    )


def _intent_to_dict(intent: VibeIntent) -> Dict[str, Any]:
    return {
        "semantic_query": intent.semantic_query,
        "hard_constraints": intent.hard_constraints,
        "soft_preferences": intent.soft_preferences,
        "exclusions": intent.exclusions,
    }


def _build_debug(final_state: Dict[str, Any]) -> RecommendDebug:
    plan = final_state.get("playlist_plan")
    intent = final_state.get("intent")
    critic_report = final_state.get("critic_report")

    return RecommendDebug(
        intent=_intent_to_dict(intent) if intent is not None else {},
        plan=plan.to_dict() if plan is not None else {},
        relational_candidate_count=final_state.get("relational_candidate_count", 0),
        vector_candidate_count=final_state.get("vector_candidate_count", 0),
        fused_candidate_count=final_state.get("fused_candidate_count", 0),
        retry_count=final_state.get("retry_count", 0),
        builder_fallback_used=final_state.get("builder_fallback_used", False),
        builder_fallback_reason=final_state.get("builder_fallback_reason"),
        critic_report=CriticReport(**critic_report) if critic_report else None,
    )


# ---------------------------------------------------------------------------
# Error classification
#
# run_playlist_graph() doesn't catch node exceptions, so a failure anywhere in
# the graph propagates here as a raw exception. Classify into the four
# documented failure modes; anything unrecognized falls through to 500.
# ---------------------------------------------------------------------------

def _classify_error(exc: Exception) -> HTTPException:
    message = str(exc)
    module = type(exc).__module__

    if isinstance(exc, ValueError) and "Failed to parse prompt" in message:
        return HTTPException(status_code=422, detail=f"Could not understand the request: {message}")

    if module.startswith("chromadb"):
        return HTTPException(status_code=503, detail=f"Vector database is unavailable: {message}")

    if module.startswith("sqlalchemy"):
        return HTTPException(status_code=503, detail=f"Relational database is unavailable: {message}")

    if module.startswith("groq"):
        return HTTPException(status_code=502, detail=f"LLM request failed: {message}")

    if isinstance(exc, ValueError) and (
        "PlannerAgent failed" in message
        or "PlaylistBuilderAgent failed" in message
        or "CriticAgent failed" in message
    ):
        return HTTPException(status_code=502, detail=f"LLM failed to produce a valid response: {message}")

    return HTTPException(status_code=500, detail=f"Unexpected error while generating playlist: {message}")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/recommend", response_model=RecommendResponse)
def recommend(
    body: RecommendRequest,
    llm_client: Any = Depends(get_llm_client),
    planner_agent: PlannerAgent = Depends(get_planner_agent),
    playlist_builder_agent: PlaylistBuilderAgent = Depends(get_playlist_builder_agent),
    prompt_schema_path: str = Depends(get_prompt_schema_path),
    database_url: Optional[str] = Depends(get_database_url),
    chroma_persist_directory: Optional[str] = Depends(get_chroma_persist_directory),
) -> RecommendResponse:
    prompt_parser = PromptParser(
        prompt_schema_path=prompt_schema_path,
        user_input=body.prompt,
        llm_client=llm_client,
    )

    try:
        final_state = run_playlist_graph(
            user_prompt=body.prompt,
            prompt_parser=prompt_parser,
            planner_agent=planner_agent,
            playlist_builder_agent=playlist_builder_agent,
            database_url=database_url,
            chroma_persist_directory=chroma_persist_directory,
        )
    except Exception as exc:
        logger.exception("Playlist generation failed for prompt=%r", body.prompt)
        raise _classify_error(exc) from exc

    playlist = final_state.get("playlist", [])
    if not playlist:
        raise HTTPException(
            status_code=404,
            detail="No tracks matched this request. Try broadening the prompt.",
        )

    critic_report = final_state.get("critic_report") or {}
    if critic_report.get("accept") is False:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "playlist_rejected",
                "message": (
                    "Could not generate a playlist that passed the quality check "
                    "within the allowed attempts. Try rephrasing your request."
                ),
                "reason": critic_report.get("reason") or "The playlist did not pass the quality check.",
                "retry_count": final_state.get("retry_count", 0),
            },
        )

    return RecommendResponse(
        prompt=body.prompt,
        playlist=[_to_playlist_track(c) for c in playlist],
        debug=_build_debug(final_state),
    )
