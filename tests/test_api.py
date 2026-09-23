from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.agents.vibe_intent import VibeIntent
from backend.api.main import app
from backend.api.routes import recommend as recommend_route


@dataclass
class DummyRetrievalLimits:
    """Minimal retrieval limit shape used by RecommendDebug."""

    n_vector: int = 25
    n_relational: int = 10


@dataclass
class DummyPlaylistPlan:
    """Small stand-in for PlaylistPlan with the API's to_dict contract."""

    playlist_size: int = 2
    semantic_weight: float = 0.6
    relational_weight: float = 0.2
    soft_preference_weight: float = 0.15
    novelty_weight: float = 0.05
    artist_repeat_penalty: float = 0.2
    genre_concentration_penalty: float = 0.15
    exclusion_penalty: float = 0.2
    retrieval_limits: DummyRetrievalLimits = field(default_factory=DummyRetrievalLimits)
    broaden_if_low_recall: bool = True
    diversity_strictness: str = "medium"
    rationale: str = "Prefer rainy late-night atmosphere with enough variety."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DummyLLMClient:
    """Avoids real LLM calls during API contract tests."""

    def generate(self, system_prompt: str, user_input: str, json_mode: bool = False) -> str:
        return "{}"


class DummyPlannerAgent:
    """Placeholder dependency; run_playlist_graph is mocked."""

    pass


class DummyPlaylistBuilderAgent:
    """Placeholder dependency; run_playlist_graph is mocked."""

    pass


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient with recommendation dependencies overridden."""

    app.dependency_overrides[recommend_route.get_llm_client] = lambda: DummyLLMClient()
    app.dependency_overrides[recommend_route.get_planner_agent] = lambda: DummyPlannerAgent()
    app.dependency_overrides[recommend_route.get_playlist_builder_agent] = (
        lambda: DummyPlaylistBuilderAgent()
    )
    app.dependency_overrides[recommend_route.get_prompt_schema_path] = (
        lambda: "backend/agents/prompt_schema.json"
    )
    app.dependency_overrides[recommend_route.get_database_url] = (
        lambda: "sqlite:///database/music_relational.db"
    )
    app.dependency_overrides[recommend_route.get_chroma_persist_directory] = (
        lambda: "database/chroma_db"
    )
    with TestClient(app) as test_client:
        yield test_client
        app.dependency_overrides.clear()


def _final_state(playlist: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a representative graph final_state for response mapping tests."""

    return {
        "intent": VibeIntent(
            semantic_query="late night rainy city drive",
            hard_constraints={},
            soft_preferences={"mood": ["rainy", "nocturnal"]},
            exclusions={},
        ),
        "playlist_plan": DummyPlaylistPlan(),
        "playlist": playlist
        if playlist is not None
        else [
            {
                "track_id": "TR001",
                "metadata": {
                    "title": "Night Drive",
                    "artist_name": "The Signals",
                    "album_title": "After Hours",
                    "year": 2011,
                    "seed_genre": "electronic",
                    "top_tags_json": {"rainy": 0.9, "night": 0.8, "driving": 0.7},
                },
                "score": 0.91,
                "sources": ["relational", "vector"],
            },
            {
                "track_id": "TR002",
                "metadata": {
                    "title": "Wet Asphalt",
                    "artist_name": "City Lights",
                    "genres_csv": "trip-hop|downtempo",
                    "top_tags_json": {"moody": 0.8},
                },
                "score": 0.82,
                "sources": ["vector"],
            },
        ],
        "relational_candidate_count": 10,
        "vector_candidate_count": 25,
        "fused_candidate_count": 30,
        "retry_count": 0,
    }


def test_health_returns_ok() -> None:
    """Health endpoint returns the simple liveness payload."""

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_returns_playlist_and_debug(monkeypatch, client: TestClient) -> None:
    """Recommend maps graph output into the public response contract."""

    captured: dict[str, Any] = {}

    def Test_run_playlist_graph(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _final_state()

    monkeypatch.setattr(recommend_route, "run_playlist_graph", Test_run_playlist_graph)

    response = client.post(
        "/recommend",
        json={"prompt": "late-night rainy city drive"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prompt"] == "late-night rainy city drive"
    assert body["playlist"] == [
        {
            "track_id": "TR001",
            "title": "Night Drive",
            "artist_name": "The Signals",
            "album_title": "After Hours",
            "year": 2011,
            "genre": "electronic",
            "tags": ["rainy", "night", "driving"],
            "score": 0.91,
            "sources": ["relational", "vector"],
        },
        {
            "track_id": "TR002",
            "title": "Wet Asphalt",
            "artist_name": "City Lights",
            "album_title": None,
            "year": None,
            "genre": "trip-hop",
            "tags": ["moody"],
            "score": 0.82,
            "sources": ["vector"],
        },
    ]
    assert body["debug"]["relational_candidate_count"] == 10
    assert body["debug"]["vector_candidate_count"] == 25
    assert body["debug"]["fused_candidate_count"] == 30
    assert body["debug"]["intent"]["semantic_query"] == "late night rainy city drive"
    assert body["debug"]["plan"]["playlist_size"] == 2

    assert captured["user_prompt"] == "late-night rainy city drive"
    assert captured["database_url"] == "sqlite:///database/music_relational.db"
    assert captured["chroma_persist_directory"] == "database/chroma_db"


def test_recommend_returns_404_for_empty_playlist(monkeypatch, client: TestClient) -> None:
    """An empty graph playlist becomes the documented 404 response."""

    monkeypatch.setattr(
        recommend_route,
        "run_playlist_graph",
        lambda **_: _final_state(playlist=[]),
    )

    response = client.post("/recommend", json={"prompt": "too narrow"})

    assert response.status_code == 404
    assert response.json() == {
        "detail": "No tracks matched this request. Try broadening the prompt."
    }


def test_recommend_classifies_prompt_parse_errors(monkeypatch, client: TestClient) -> None:
    """Prompt parser failures are exposed as client-side 422 errors."""

    def raise_parse_error(**_: Any) -> dict[str, Any]:
        raise ValueError("Failed to parse prompt after 3 attempts. Last error: bad json")

    monkeypatch.setattr(recommend_route, "run_playlist_graph", raise_parse_error)

    response = client.post("/recommend", json={"prompt": "???"})

    assert response.status_code == 422
    assert "Could not understand the request" in response.json()["detail"]


def test_recommend_requires_prompt_field(client: TestClient) -> None:
    """FastAPI validation rejects requests without the required prompt."""

    response = client.post("/recommend", json={})

    assert response.status_code == 422


@pytest.mark.parametrize("retry_count", [0, 2])
def test_recommend_returns_422_when_critic_rejects(monkeypatch, client, retry_count):
    state = _final_state()
    state["retry_count"] = retry_count
    state["critic_report"] = {
        "accept": False,
        "reason": "The playlist includes an excluded genre.",
        "suggested_adjustments": {"exclusion_penalty": 0.9},
    }
    monkeypatch.setattr(recommend_route, "run_playlist_graph", lambda **_: state)

    response = client.post("/recommend", json={"prompt": "no metal"})

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "playlist_rejected",
            "message": (
                "Could not generate a playlist that passed the quality check "
                "within the allowed attempts. Try rephrasing your request."
            ),
            "reason": "The playlist includes an excluded genre.",
            "retry_count": retry_count,
        }
    }


def test_recommend_returns_accepted_playlist_after_retry(monkeypatch, client):
    state = _final_state()
    state["retry_count"] = 2
    state["critic_report"] = {
        "accept": True, "reason": "Matches the request.", "suggested_adjustments": {},
    }
    monkeypatch.setattr(recommend_route, "run_playlist_graph", lambda **_: state)

    response = client.post("/recommend", json={"prompt": "rainy drive"})

    assert response.status_code == 200
    assert len(response.json()["playlist"]) == 2
    assert response.json()["debug"]["critic_report"]["accept"] is True


def test_empty_rejected_playlist_keeps_404(monkeypatch, client):
    state = _final_state(playlist=[])
    state["critic_report"] = {
        "accept": False, "reason": "No tracks.", "suggested_adjustments": {},
    }
    monkeypatch.setattr(recommend_route, "run_playlist_graph", lambda **_: state)

    response = client.post("/recommend", json={"prompt": "rare tracks"})

    assert response.status_code == 404
