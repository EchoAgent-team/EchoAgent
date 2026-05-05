"""
Vector retrieval agent for EchoAgent.

This module is a thin orchestration layer between:
- parsed/validated VibeIntent objects
- semantic query text
- Chroma / vector embedding search

It is designed to be flexible in terms of where it can be called from:
- directly in backend services
- inside API routes
- as a LangGraph node
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from backend.data.embeddings import get_embedding_manager, get_vector_store
from .vibe_intent import VibeIntent


def _clean_result(result: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """
    Convert one raw Chroma result into a cleaner candidate dictionary.
    """
    metadata = result.get("metadata") or {}
    track_id = metadata.get("track_id") or metadata.get("song_id") or result.get("id")

    return {
        "track_id": track_id,
        "vector_rank": rank,
        "vector_distance": result.get("distance"),
        "metadata": metadata,
        "source": "vector",
    }


def retrieve_vector_candidates(
    intent: VibeIntent,
    limit: int = 100,
    persist_directory: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search the vector DB using the semantic query from a VibeIntent.

    Args:
        intent:
            Parsed user intent.
        limit:
            Number of similar tracks to return.
        persist_directory:
            Optional path to the persisted Chroma vector DB.
    """
    if not isinstance(intent, VibeIntent):
        raise TypeError("intent must be a VibeIntent")

    intent.normalize()
    intent.validate()

    embedding_manager = get_embedding_manager(persist_directory=persist_directory)

    raw_results = embedding_manager.query_track_embeddings(
        query_text=intent.semantic_query,
        top_k=limit,
    )

    candidates = []
    for rank, result in enumerate(raw_results, start=1):
        candidates.append(_clean_result(result, rank))

    return candidates


def retrieve_vector_bundle(
    intent: VibeIntent,
    limit: int = 100,
    persist_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Same search, but with useful debug fields for later graph/API work.
    """
    candidates = retrieve_vector_candidates(
        intent=intent,
        limit=limit,
        persist_directory=persist_directory,
    )

    return {
        "vector_query": intent.semantic_query,
        "vector_candidates": candidates,
        "vector_candidate_count": len(candidates),
    }


def vector_retrieval_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-style wrapper.

    Input state:
        {
            "intent": VibeIntent(...),
            "vector_limit": 100
        }
    """
    return retrieve_vector_bundle(
        intent=state["intent"],
        limit=state.get("vector_limit", 100),
        persist_directory=state.get("chroma_persist_directory"),
    )


def build_langchain_vector_retriever(search_k: int = 50):
    """
    Optional LangChain hook for later.

    We keep this separate from the main retrieval function so the core agent
    stays easy to understand.
    """
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": search_k})
