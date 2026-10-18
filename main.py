import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()  # before importing modules that read the environment

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import database
from controllers.evaluation_controller import router as evaluation_router
from services.rag_service import RAGService

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("prismeval")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.init_db()

    for name in ("COZE_API_TOKEN", "COZE_BOT_ID"):
        if not os.getenv(name):
            logger.warning("%s is not set; /api/v1/evaluate will fail until it is.", name)

    # Build the RAG index now so the first request isn't slow.
    try:
        await RAGService.retrieve_context("warm up the retrieval index", 256, 0.5)
    except Exception:
        logger.exception("RAG warm-up failed; it will retry on the first request.")

    yield


app = FastAPI(
    title="PrismEval",
    description="Multi-agent evaluation with hybrid RAG and an EQI meta-evaluator.",
    version="1.0.0",
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluation_router)


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )