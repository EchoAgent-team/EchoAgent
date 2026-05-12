"""
Playlist builder for EchoAgent.

Selects the final N tracks from ranked_candidates, enforcing artist repetition
caps, genre diversity, and optional energy-flow shaping.

TODO: implement selection logic using PlaylistPlan constraints.
"""

from __future__ import annotations

from typing import Any, Dict


def build_playlist_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stub — slices top N from ranked_candidates until builder is implemented."""
    plan = state.get("playlist_plan")
    n = plan.playlist_size if plan else 20
    ranked = state.get("ranked_candidates", [])
    return {
        "playlist": ranked[:n],
    }
