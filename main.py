"""
main.py — Application entry point.

Wires together:
- FastAPI app with all routes
- Task router with all module registrations
- Prompt template registration
- Database initialization
- Logging setup
"""

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from utils.logger import get_logger

logger = get_logger("main")


# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle handler.
    Runs startup logic before yielding, and cleanup on shutdown.
    """
    logger.info("=" * 60)
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("=" * 60)

    # 1. Initialize database
    from storage.database import db
    db.init()

    # 2. Register all prompt templates
    from prompts.resume_prompts import register_resume_prompts
    from prompts.assistant_prompts import (
        register_assistant_prompts,
        register_job_app_prompts,
        register_code_prompts,
    )
    register_resume_prompts()
    register_assistant_prompts()
    register_job_app_prompts()
    register_code_prompts()
    logger.info("Prompt templates registered")

    # 3. Register module handlers with the task router
    from core.task_router import router
    from modules.resume.handler import resume_handler
    from modules.assistant.handler import assistant_handler
    from modules.job_application.handler import job_application_handler
    from modules.code_gen.handler import code_gen_handler
    from modules.learning.handler import learning_handler

    router.register("resume", resume_handler.handle)
    router.register("assistant", assistant_handler.handle)
    router.register("job_application", job_application_handler.handle)
    router.register("code", code_gen_handler.handle)
    router.register("learning", learning_handler.handle)
    logger.info("Task router registered modules: %s", router.available_tasks())

    # 4. Initialize vector memory store
    from storage.vector_store import vector_store
    vector_store.init()

    # 5. Check Ollama availability
    from core.llm_service import llm_service
    if llm_service.is_available():
        models = llm_service.list_models()
        logger.info("Ollama is available. Models: %s", models)
    else:
        logger.warning(
            "Ollama is NOT reachable at %s. "
            "Start it with: ollama serve",
            settings.OLLAMA_BASE_URL,
        )

    logger.info("System ready. API running at http://%s:%d%s", settings.API_HOST, settings.API_PORT, settings.API_PREFIX)

    yield  # App runs here

    # Shutdown
    logger.info("Shutting down %s", settings.APP_NAME)
    db.prune_old_records()


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Local multi-purpose AI system powered by Ollama (Mistral/LLaMA 3). "
        "Modules: Resume Optimization, Job Application, General Assistant, Code Generation."
    ),
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────
from api.routes.task import task_route
from api.routes.resume import resume_route
from api.routes.other_routes import (
    assistant_route,
    job_app_route,
    code_route,
    history_route,
)
from api.routes.learning import learning_route

app.include_router(task_route, prefix=settings.API_PREFIX)
app.include_router(resume_route, prefix=settings.API_PREFIX)
app.include_router(assistant_route, prefix=settings.API_PREFIX)
app.include_router(job_app_route, prefix=settings.API_PREFIX)
app.include_router(code_route, prefix=settings.API_PREFIX)
app.include_router(history_route, prefix=f"{settings.API_PREFIX}/history")
app.include_router(learning_route, prefix=settings.API_PREFIX)


# ─── Health & Root ────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "system": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": f"{settings.API_PREFIX}/docs",
        "health": "/health",
    })


@app.get("/health", tags=["System"])
async def health_check():
    from core.llm_service import llm_service
    from core.task_router import router

    ollama_ok = llm_service.is_available()
    return JSONResponse({
        "status": "healthy" if ollama_ok else "degraded",
        "ollama": "connected" if ollama_ok else "unreachable",
        "model": settings.DEFAULT_MODEL,
        "registered_tasks": router.available_tasks(),
        "version": settings.APP_VERSION,
    })


@app.get(f"{settings.API_PREFIX}/status", tags=["System"])
async def system_status():
    from core.llm_service import llm_service
    from storage.database import db

    return {
        "ollama_available": llm_service.is_available(),
        "available_models": llm_service.list_models(),
        "active_model": settings.DEFAULT_MODEL,
        "database": settings.DATABASE_URL,
        "recent_tasks": db.get_task_history(limit=5),
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
