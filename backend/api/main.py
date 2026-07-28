from fastapi import FastAPI

from backend.api.routes import health

app = FastAPI(title="EchoAgent API")

app.include_router(health.router)
