"""
Critic Agent for EchoAgent.

Reviews the generated playlist against the original prompt and PlaylistPlan,
then returns a CriticReport: accept, reason, and optional suggested_adjustments
that the planner can incorporate on a retry.

TODO: implement LLM-backed critique with structured output validation.
"""

from __future__ import annotations

from typing import Any, Dict


def critique_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stub — always accepts until CriticAgent is implemented."""
    return {
        "critic_report": {
            "accept": True,
            "reason": "stub: critique not yet implemented",
            "suggested_adjustments": {},
        },
    }
