"""
Deterministic reranker for EchoAgent.

Applies PlaylistPlan weights to fused candidates and produces a scored,
sorted ranked_candidates list with per-track score components.

TODO: implement score computation using planner weights and VibeIntent signals.
"""

from __future__ import annotations

from typing import Any, Dict


def rerank_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stub — returns fused_candidates as-is until reranker is implemented."""
    return {
        "ranked_candidates": state.get("fused_candidates", []),
    }
