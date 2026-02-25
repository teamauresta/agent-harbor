"""
Harbor — FastAPI entry point.
Multi-tenant AI web chat agent framework powered by Chatwoot + LangGraph.
"""
import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from webhook.handler import router as webhook_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger()
settings = get_settings()

app = FastAPI(
    title="Harbor",
    description="AI web chat agent framework — Chatwoot + LangGraph",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Webhook routes: POST /webhook/{client_id}
app.include_router(webhook_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "harbor"}


@app.get("/")
async def root():
    return {
        "service": "Harbor",
        "description": "AI web chat agents powered by Chatwoot + LangGraph",
        "version": "0.1.0",
    }


if __name__ == "__main__":
    log.info("harbor.starting", port=settings.port)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
    )
