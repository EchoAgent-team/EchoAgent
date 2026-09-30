from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from dotenv import load_dotenv

from backend.agents.critic_agent import CriticAgent
from backend.agents.planner_agent import PlannerAgent
from backend.agents.playlist_builder import PlaylistBuilderAgent
from backend.agents.prompt_parser import LLMClient
from backend.api.routes import health, recommend

PROMPT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "agents" / "prompt_schema.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Build the LLM-backed agents once at startup and reuse them across requests.

    Two clients on separate Groq model buckets: a light one (parser, planner)
    and a heavy one (builder, critic) where taste judgment matters. The agents are
    stateless w.r.t. any single prompt, so they're safe to share. PromptParser
    bakes user_input into __init__, so a fresh one is built per request instead
    (see routes/recommend.py).
    """
    api_key = os.environ.get("GROQ_API_KEY")

    def build_client(model_name: str) -> LLMClient:
        return LLMClient(
            model_name=model_name,
            device=None,
            api_key=api_key,
            endpoint=None,
            temperature=float(os.environ.get("GROQ_TEMPERATURE", "0.7")),
            max_new_tokens=int(os.environ.get("GROQ_MAX_TOKENS", "1024")),
            provider="groq",
            reasoning_effort=os.environ.get("GROQ_REASONING_EFFORT", "low"),
        )

    light_client = heavy_client = None
    if api_key:
        light_client = build_client(os.environ.get("GROQ_MODEL_LIGHT", "openai/gpt-oss-20b"))
        heavy_client = build_client(os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"))

    app.state.light_llm_client = light_client
    app.state.planner_agent = PlannerAgent(llm_client=light_client) if light_client else None
    app.state.playlist_builder_agent = PlaylistBuilderAgent(llm_client=heavy_client) if heavy_client else None
    app.state.critic_agent = CriticAgent(llm_client=heavy_client) if heavy_client else None
    app.state.prompt_schema_path = str(PROMPT_SCHEMA_PATH)
    app.state.database_url = os.environ.get("DATABASE_URL")
    app.state.chroma_persist_directory = os.environ.get("CHROMA_PERSIST_DIRECTORY")

    yield


app = FastAPI(title="EchoAgent API", lifespan=lifespan)

app.include_router(health.router)
app.include_router(recommend.router)
