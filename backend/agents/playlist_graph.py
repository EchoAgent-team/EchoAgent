"""
LangGraph orchestration layer for EchoAgent playlist generation.

Graph shape:
    parse_intent → plan ─┬─ retrieve_relational ─┐
                         └─ retrieve_vector ─────┴→ fuse_candidates
                                                            → rerank
                                                            → build_playlist
                                                            → critique
                                                           ↙           ↘
                                                         plan          END
                                                      (retry, capped)

Agents (LLM-driven):  parse_intent, plan, build_playlist, critique
Deterministic nodes:  retrieve_relational, retrieve_vector, fuse_candidates, rerank
"""

from __future__ import annotations

from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from backend.agents.candidate_fuser import candidate_fusion_node
from backend.agents.graph_state import PlaylistGraphState
from backend.agents.prompt_parser import PromptParser
from backend.agents.planner_agent import PlannerAgent
from backend.agents.relational_retrieval_agent import relational_retrieval_node
from backend.agents.vector_retrieval_agent import vector_retrieval_node
from backend.agents.reranker import reranker_node
from backend.agents.playlist_builder import build_playlist_node, PlaylistBuilderAgent
from backend.agents.critic_agent import critique_node    # stub — not yet implemented


# ---------------------------------------------------------------------------
# Node: parse_intent
# ---------------------------------------------------------------------------

def parse_intent_node(state: PlaylistGraphState) -> Dict[str, Any]:
    """Convert raw user_prompt into a VibeIntent via PromptParser."""
    parser: PromptParser = state["prompt_parser"]
    intent = parser.parse(state["user_prompt"])
    return {"intent": intent}


# ---------------------------------------------------------------------------
# Node: plan
# ---------------------------------------------------------------------------

def plan_node(state: PlaylistGraphState) -> Dict[str, Any]:
    """
    Run PlannerAgent to produce a PlaylistPlan.

    On a critic retry the critic_report's suggested_adjustments are available
    in state but the planner currently re-derives the plan from scratch.
    Downstream callers can extend this to seed the LLM with the adjustments.
    """
    agent: PlannerAgent = state["planner_agent"]
    plan = agent.plan(
        user_prompt=state["user_prompt"],
        intent=state["intent"],
    )
    retry_count = state.get("retry_count", 0)
    return {
        "playlist_plan": plan,
        "retry_count": retry_count,
    }


# ---------------------------------------------------------------------------
# Node: retrieve_relational
# ---------------------------------------------------------------------------

def retrieve_relational_node(state: PlaylistGraphState) -> Dict[str, Any]:
    """Thin wrapper — delegates to the agent's node interface."""
    return relational_retrieval_node(state)


# ---------------------------------------------------------------------------
# Node: retrieve_vector
# ---------------------------------------------------------------------------

def retrieve_vector_node(state: PlaylistGraphState) -> Dict[str, Any]:
    """Thin wrapper — delegates to the agent's node interface."""
    return vector_retrieval_node(state)


# ---------------------------------------------------------------------------
# Conditional edge: route_critic
# ---------------------------------------------------------------------------

def route_critic(
    state: PlaylistGraphState,
) -> Literal["plan", "__end__"]:
    """
    After critique:
      - accept=True  → END
      - accept=False and retries remain → back to 'plan'
      - accept=False and retries exhausted → END (best-effort playlist stands)
    """
    report = state.get("critic_report", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if report.get("accept", True):
        return END

    if retry_count < max_retries:
        return "plan"

    return END


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_playlist_graph() -> StateGraph:
    """Assemble and compile the EchoAgent playlist graph."""
    graph = StateGraph(PlaylistGraphState)

    # --- nodes ---
    graph.add_node("parse_intent", parse_intent_node)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve_relational", retrieve_relational_node)
    graph.add_node("retrieve_vector", retrieve_vector_node)
    graph.add_node("fuse_candidates", candidate_fusion_node)
    graph.add_node("rerank", reranker_node)
    graph.add_node("build_playlist", build_playlist_node)
    graph.add_node("critique", critique_node)

    # --- entry point ---
    graph.set_entry_point("parse_intent")

    # --- linear edges ---
    graph.add_edge("parse_intent", "plan")

    # plan → parallel retrieval fan-out
    graph.add_edge("plan", "retrieve_relational")
    graph.add_edge("plan", "retrieve_vector")

    # parallel retrieval fan-in → fuse
    graph.add_edge("retrieve_relational", "fuse_candidates")
    graph.add_edge("retrieve_vector", "fuse_candidates")

    graph.add_edge("fuse_candidates", "rerank")
    graph.add_edge("rerank", "build_playlist")
    graph.add_edge("build_playlist", "critique")

    # --- critic routing (conditional) ---
    graph.add_conditional_edges(
        "critique",
        route_critic,
        {"plan": "plan", END: END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_playlist_graph(
    user_prompt: str,
    prompt_parser: PromptParser,
    planner_agent: PlannerAgent,
    playlist_builder_agent: PlaylistBuilderAgent,
    database_url: str | None = None,
    chroma_persist_directory: str | None = None,
    max_retries: int = 2,
) -> PlaylistGraphState:
    """
    Run the full playlist graph for a single user prompt.

    Args:
        user_prompt:               Raw natural-language request.
        prompt_parser:             Initialised PromptParser instance.
        planner_agent:             Initialised PlannerAgent instance.
        playlist_builder_agent:    Initialised PlaylistBuilderAgent instance.
        database_url:              Optional SQLAlchemy DB URL override.
        chroma_persist_directory:  Optional Chroma persistence path override.
        max_retries:               Maximum critic retry loops (default 2).

    Returns:
        Final PlaylistGraphState with state.playlist populated.
    """
    graph = build_playlist_graph()

    initial_state: PlaylistGraphState = {
        "user_prompt": user_prompt,
        "prompt_parser": prompt_parser,
        "planner_agent": planner_agent,
        "playlist_builder_agent": playlist_builder_agent,
        "database_url": database_url,
        "chroma_persist_directory": chroma_persist_directory,
        "retry_count": 0,
        "max_retries": max_retries,
        "trace": [],
        "errors": [],
    }

    return graph.invoke(initial_state)
