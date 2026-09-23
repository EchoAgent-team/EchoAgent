"""Offline JSON and API-to-real-graph tests; only external I/O is scripted."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from groq import BadRequestError, RateLimitError

from backend.agents import playlist_graph
from backend.agents.critic_agent import CriticAgent
from backend.agents.json_output import JSONOutputError
from backend.agents.planner_agent import PlannerAgent, PlaylistPlan, RetrievalLimits
from backend.agents.playlist_builder import PlaylistBuilderAgent
from backend.agents.prompt_parser import LLMClient, PromptParser
from backend.agents.vibe_intent import VibeIntent
from backend.api.routes import recommend

SCHEMA = str(Path(__file__).resolve().parents[1] / "backend/agents/prompt_schema.json")
INTENT = {"semantic_query": "rainy drive", "hard_constraints": {}, "soft_preferences": {}, "exclusions": {}}
PLAN = PlaylistPlan(playlist_size=5, retrieval_limits=RetrievalLimits(50, 50), rationale="Rainy atmosphere.")
PLAN_JSON = {**PLAN.to_dict(), "retrieval_limits": {"vector": 50, "relational": 50}}
TRACKS = [{"track_id": f"TR{i}", "title": f"Track {i}", "artist_name": f"Artist {i}", "seed_genre": "ambient"} for i in range(5)]
BUILDER = {"selected_track_ids": [t["track_id"] for t in TRACKS], "energy_arc": "Gentle throughout.", "rationale": "Fits rain."}
ACCEPT = {"accept": True, "reason": "Matches the request.", "suggested_adjustments": {}}
REJECT = {"accept": False, "reason": "Too repetitive.", "suggested_adjustments": {"artist_repeat_penalty": 0.8}}


def completion(payload, finish_reason="stop"):
    content = json.dumps(payload) if isinstance(payload, dict) else payload
    return NS(choices=[NS(message=NS(content=content), finish_reason=finish_reason)],
              usage=NS(prompt_tokens=100, completion_tokens=40, total_tokens=140))


def provider_error(cls, status, code):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test/completions"))
    return cls("scripted provider failure", response=response, body={"code": code})


def scripted_client(responses):
    # Keep real generate/response handling; replace only the SDK transport.
    client = LLMClient.__new__(LLMClient)
    client.model_name = "test-model"
    client.temperature = 0.7
    client.max_new_tokens = 1024
    client.top_p = 1.0
    client.stream = False
    client.compound_custom = None
    client._mode = "groq"
    client._api_client = Mock()
    client._api_client.chat.completions.create.side_effect = responses
    return client


def calls(client):
    return client._api_client.chat.completions.create.call_args_list


def invoke_agent(name, client):
    intent = VibeIntent(**INTENT)
    if name == "parser":
        return PromptParser(SCHEMA, "rainy drive", client).parse()
    if name == "planner":
        return PlannerAgent(client).plan("rainy drive", intent)
    if name == "builder":
        return PlaylistBuilderAgent(client).build("rainy drive", intent, PLAN, TRACKS)
    return CriticAgent(client).critique("rainy drive", intent, PLAN, [])


VALID = {"parser": INTENT, "planner": PLAN_JSON, "builder": BUILDER, "critic": ACCEPT}


@pytest.mark.parametrize("agent", VALID)
@pytest.mark.parametrize("bad", ["not JSON", "{broken}", "", None, {}, "truncated", "provider_rejected"])
def test_all_agents_repair_without_disabling_json(agent, bad):
    if bad == "truncated":
        failure = completion(VALID[agent], "length")
    elif bad == "provider_rejected":
        failure = provider_error(BadRequestError, 400, "json_validate_failed")
    else:
        failure = completion(bad)
    client = scripted_client([failure, completion(VALID[agent])])
    invoke_agent(agent, client)
    assert len(calls(client)) == 2
    assert all(c.kwargs["response_format"] == {"type": "json_object"} for c in calls(client))
    assert "INVALID" in calls(client)[1].kwargs["messages"][0]["content"]
    if bad == "not JSON":
        assert "not JSON" in calls(client)[1].kwargs["messages"][0]["content"]


@pytest.mark.parametrize("agent", VALID)
def test_repairs_stop_after_three_attempts(agent):
    client = scripted_client([completion("bad")] * 3)
    with pytest.raises(ValueError, match="after 3 attempts"):
        invoke_agent(agent, client)
    assert len(calls(client)) == 3


@pytest.mark.parametrize("agent", VALID)
def test_rate_limit_is_not_a_json_repair(agent):
    error = provider_error(RateLimitError, 429, "rate_limit_exceeded")
    client = scripted_client([error])
    with pytest.raises(RateLimitError):
        invoke_agent(agent, client)
    assert len(calls(client)) == 1


def test_diagnostics_include_context_and_usage_without_content(caplog):
    client = scripted_client([completion(INTENT, "length"), completion(INTENT)])
    with caplog.at_level(logging.INFO, logger="backend.agents.json_output"):
        invoke_agent("parser", client)
    assert "agent=PromptParser attempt=1 model=test-model finish_reason=length" in caplog.text
    assert "agent=PromptParser attempt=2 model=test-model finish_reason=stop" in caplog.text
    assert "prompt_tokens=100 completion_tokens=40 total_tokens=140" in caplog.text
    assert "rainy drive" not in caplog.text


@pytest.mark.parametrize("reason", ["stop", "length", None])
def test_stream_completion_status_is_checked(reason):
    chunks = [NS(choices=[NS(delta=NS(content="{}"), finish_reason=reason)], usage=None)]
    client = scripted_client([iter(chunks)])
    client.stream = True
    if reason == "stop":
        assert client.generate("JSON", "test", json_mode=True) == "{}"
    else:
        with pytest.raises(JSONOutputError):
            client.generate("JSON", "test", json_mode=True)


def test_no_typeerror_plain_text_retry():
    client = Mock()
    client.generate.side_effect = TypeError("client error")
    with pytest.raises(ValueError, match="after 3 attempts"):
        invoke_agent("planner", client)
    assert client.generate.call_count == 3
    assert all(c.kwargs["json_mode"] is True for c in client.generate.call_args_list)


@pytest.fixture
def graph_api(monkeypatch):
    def run(responses, empty=False, max_retries=2):
        llm = scripted_client(responses)
        relational = Mock(return_value={"relational_candidates": [] if empty else TRACKS,
                                       "relational_candidate_count": 0 if empty else len(TRACKS)})
        vector_tracks = [
            {"track_id": t["track_id"], "metadata": t, "vector_rank": i + 1, "vector_distance": 0.1}
            for i, t in enumerate(TRACKS)
        ]
        vector = Mock(return_value={"vector_candidates": [] if empty else vector_tracks,
                                  "vector_candidate_count": 0 if empty else len(vector_tracks)})
        monkeypatch.setattr(playlist_graph, "relational_retrieval_node", relational)
        monkeypatch.setattr(playlist_graph, "vector_retrieval_node", vector)
        # Preserve the real graph runner; configure only its retry limit.
        def runner(**kwargs):
            return playlist_graph.run_playlist_graph(**kwargs, max_retries=max_retries)
        monkeypatch.setattr(recommend, "run_playlist_graph", runner)
        app = FastAPI()
        app.include_router(recommend.router)
        app.dependency_overrides = {
            recommend.get_llm_client: lambda: llm,
            recommend.get_planner_agent: lambda: PlannerAgent(llm),
            recommend.get_playlist_builder_agent: lambda: PlaylistBuilderAgent(llm),
            recommend.get_prompt_schema_path: lambda: SCHEMA,
            recommend.get_database_url: lambda: None,
            recommend.get_chroma_persist_directory: lambda: None,
        }
        with TestClient(app) as client:
            response = client.post("/recommend", json={"prompt": "rainy drive"})
        return response, llm, relational
    return run


def test_real_graph_immediate_acceptance(graph_api):
    response, llm, retrieval = graph_api([completion(x) for x in [INTENT, PLAN_JSON, BUILDER, ACCEPT]])
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["playlist"]) == 5
    assert all(t["sources"] == ["relational", "vector"] for t in body["playlist"])
    assert body["debug"]["retry_count"] == 0
    assert body["debug"]["fused_candidate_count"] == 5
    assert body["debug"]["builder_fallback_used"] is False
    assert body["debug"]["critic_report"]["accept"] is True
    assert len(calls(llm)) == 4
    assert retrieval.call_count == 1


def test_real_graph_replans_with_critic_feedback(graph_api):
    sequence = [INTENT, PLAN_JSON, BUILDER, REJECT, PLAN_JSON, BUILDER, ACCEPT]
    response, llm, retrieval = graph_api([completion(x) for x in sequence])
    assert response.status_code == 200, response.text
    assert response.json()["debug"]["retry_count"] == 1
    planner_input = calls(llm)[4].kwargs["messages"][1]["content"]
    assert "Too repetitive." in planner_input
    assert "artist_repeat_penalty" in planner_input and "0.8" in planner_input
    assert "PREVIOUS PLAN" in planner_input
    assert len(calls(llm)) == 7
    assert retrieval.call_count == 2


@pytest.mark.parametrize("limit", [0, 2, 5])
def test_real_graph_final_rejection_is_bounded(graph_api, limit):
    sequence = [INTENT] + [PLAN_JSON, BUILDER, REJECT] * (limit + 1)
    response, llm, retrieval = graph_api([completion(x) for x in sequence], max_retries=limit)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "playlist_rejected"
    assert response.json()["detail"]["retry_count"] == limit
    assert "playlist" not in response.json()
    assert len(calls(llm)) == 1 + 3 * (limit + 1)
    assert retrieval.call_count == limit + 1


def test_real_graph_empty_candidates_return_404(graph_api):
    sequence = [INTENT] + [PLAN_JSON, REJECT] * 3
    response, llm, _ = graph_api([completion(x) for x in sequence], empty=True)
    assert response.status_code == 404, response.text
    assert len(calls(llm)) == 7  # Builder never calls the model for an empty pool.


def test_real_graph_builder_fallback_visible_and_reviewed(graph_api):
    sequence = [INTENT, PLAN_JSON, "bad", "bad", "bad", ACCEPT]
    response, llm, _ = graph_api([completion(x) for x in sequence])
    assert response.status_code == 200, response.text
    debug = response.json()["debug"]
    assert debug["builder_fallback_used"] is True
    assert "after 3 attempts" in debug["builder_fallback_reason"]
    assert debug["critic_report"]["accept"] is True
    assert len(calls(llm)) == 6


def test_real_graph_fallback_flag_resets_after_successful_retry(graph_api):
    sequence = [INTENT, PLAN_JSON, "bad", "bad", "bad", REJECT, PLAN_JSON, BUILDER, ACCEPT]
    response, _, _ = graph_api([completion(x) for x in sequence])
    assert response.status_code == 200, response.text
    assert response.json()["debug"]["builder_fallback_used"] is False
    assert response.json()["debug"]["builder_fallback_reason"] is None


def test_real_graph_provider_error_does_not_become_builder_fallback(graph_api):
    sequence = [completion(INTENT), completion(PLAN_JSON), provider_error(RateLimitError, 429, "rate_limit_exceeded")]
    response, llm, _ = graph_api(sequence)
    assert response.status_code == 502, response.text
    assert "LLM request failed" in response.json()["detail"]
    assert len(calls(llm)) == 3


@pytest.mark.parametrize("prefix, expected", [([], 422), ([INTENT], 502), ([INTENT, PLAN_JSON, BUILDER], 502)])
def test_real_graph_exhausted_json_errors_are_clear(graph_api, prefix, expected):
    sequence = prefix + ["bad"] * 3
    response, llm, _ = graph_api([completion(x) for x in sequence])
    assert response.status_code == expected, response.text
    assert "after 3 attempts" in response.json()["detail"]
    assert len(calls(llm)) == len(sequence)


@pytest.mark.parametrize("agent, invalid", [
    ("parser", {**INTENT, "semantic_query": ""}),
    ("planner", {**PLAN_JSON, "playlist_size": 100}),
    ("builder", {**BUILDER, "selected_track_ids": ["unknown"] * 5}),
    ("builder", {**BUILDER, "selected_track_ids": ["TR0"] * 5}),
    ("critic", {**ACCEPT, "accept": "true"}),
    ("critic", {**REJECT, "suggested_adjustments": {"artist_repeat_penalty": 2}}),
])
def test_valid_json_still_requires_valid_fields(agent, invalid):
    client = scripted_client([completion(invalid), completion(VALID[agent])])
    invoke_agent(agent, client)
    assert len(calls(client)) == 2


def test_hf_api_requests_json_and_checks_truncation():
    client = scripted_client([])
    client._mode = "hf_api"
    client._api_client.chat_completion.return_value = completion(INTENT, "length")
    with pytest.raises(JSONOutputError, match="truncated"):
        client.generate("JSON", "rain", json_mode=True)
    assert client._api_client.chat_completion.call_args.kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("eos_id, truncated", [(99, True), (3, False)])
def test_local_generation_checks_new_token_budget(eos_id, truncated):
    from contextlib import nullcontext
    import numpy as np

    class Inputs(dict):
        def to(self, device):
            return self

    client = scripted_client([])
    client._mode = "local"
    client.max_new_tokens = 2
    client._torch = NS(no_grad=nullcontext)
    client._tokens = Mock(return_value=Inputs(input_ids=np.array([[10, 11, 12]])))
    client._tokens.decode.return_value = "{}"
    client._model = Mock(device="cpu", generation_config=NS(eos_token_id=eos_id))
    client._model.generate.return_value = np.array([[10, 11, 12, 2, 3]])
    if truncated:
        with pytest.raises(JSONOutputError, match="truncated"):
            client.generate("JSON", "rain", json_mode=True)
    else:
        assert client.generate("JSON", "rain", json_mode=True) == "{}"
    assert list(client._tokens.decode.call_args.args[0]) == [2, 3]


def test_real_graph_json_rejection_recovers_in_same_agent(graph_api):
    sequence = [completion(INTENT), completion(PLAN_JSON),
                provider_error(BadRequestError, 400, "json_validate_failed"),
                completion(BUILDER), completion(ACCEPT)]
    response, llm, _ = graph_api(sequence)
    assert response.status_code == 200, response.text
    assert response.json()["debug"]["builder_fallback_used"] is False
    assert len(calls(llm)) == 5
    assert all(c.kwargs["response_format"] == {"type": "json_object"} for c in calls(llm))
