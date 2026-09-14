from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

from backend.agents.planner_agent import PlannerAgent
from backend.agents.playlist_builder import PlaylistBuilderAgent
from backend.agents.prompt_parser import LLMClient
from backend.api.routes import health, recommend

PROMPT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "agents" / "prompt_schema.json"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Build the LLM-backed agents once at startup and reuse them across requests.

    PlannerAgent/PlaylistBuilderAgent are stateless w.r.t. any single prompt, so
    they're safe to share. PromptParser bakes user_input into __init__, so a
    fresh one is built per request instead (see routes/recommend.py).
    """
    api_key = os.environ.get("GROQ_API_KEY")

    llm_client = None
    if api_key:
        llm_client = LLMClient(
            model_name=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            device=None,
            api_key=api_key,
            endpoint=None,
            temperature=float(os.environ.get("GROQ_TEMPERATURE", "0.7")),
            max_new_tokens=int(os.environ.get("GROQ_MAX_TOKENS", "1024")),
            provider="groq",
        )

    app.state.llm_client = llm_client
    app.state.planner_agent = PlannerAgent(llm_client=llm_client) if llm_client else None
    app.state.playlist_builder_agent = PlaylistBuilderAgent(llm_client=llm_client) if llm_client else None
    app.state.prompt_schema_path = str(PROMPT_SCHEMA_PATH)
    app.state.database_url = os.environ.get("DATABASE_URL")
    app.state.chroma_persist_directory = os.environ.get("CHROMA_PERSIST_DIRECTORY")

    yield


app = FastAPI(title="EchoAgent API", lifespan=lifespan)

app.include_router(health.router)
app.include_router(recommend.router)
