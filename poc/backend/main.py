"""
Gossip Handshake Protocol — POC Backend

FastAPI server that demonstrates the Gossip Handshake Protocol:
inference-time routing of queries to specialist LoRA adapters.
"""

import logging
import time
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import CORS_ORIGINS, DOMAIN_INFO, ADAPTER_DIRS, ADAPTER_NAMES, MODEL_IDS
from models import (
    QueryRequest,
    QueryResponse,
    RoutingResult,
    GenerationResult,
    DomainInfo,
    ModelStatus,
    SwitchModelRequest,
    SampleQuestion,
)
from routers import keyword_route, cosine_route, SAMPLE_QUESTIONS
from inference import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SCALE = os.environ.get("GH_MODEL_SCALE", "0.5B")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("Starting Gossip Handshake POC backend...")
    logger.info("Loading model at scale: %s", DEFAULT_SCALE)
    engine.load_model(DEFAULT_SCALE)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Gossip Handshake Protocol — POC",
    description=(
        "Interactive demonstration of the Gossip Handshake Protocol: "
        "routing-based LoRA adapter selection for decentralised knowledge sharing."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — restricted to configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": engine.is_loaded,
        "scale": engine.current_scale,
    }


@app.get("/api/domains", response_model=list[DomainInfo])
async def list_domains():
    """List available domains and their adapter status."""
    result = []
    for key, info in DOMAIN_INFO.items():
        result.append(
            DomainInfo(
                key=key,
                label=info["label"],
                description=info["description"],
                icon=info["icon"],
                color=info["color"],
                adapter_available=key in engine.loaded_adapters,
            )
        )
    return result


@app.get("/api/model-status", response_model=ModelStatus)
async def model_status():
    """Get current model loading status."""
    scale = engine.current_scale or DEFAULT_SCALE
    return ModelStatus(
        model_scale=scale,
        model_id=MODEL_IDS.get(scale, "unknown"),
        loaded=engine.is_loaded,
        domains_loaded=engine.loaded_adapters,
        merged_loaded=engine.merged_loaded,
    )


@app.get("/api/sample-questions", response_model=list[SampleQuestion])
async def sample_questions():
    """Get sample questions for each domain."""
    return [SampleQuestion(**sq) for sq in SAMPLE_QUESTIONS]


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query through the Gossip Handshake Protocol.

    1. Route the query to the correct specialist domain
    2. Generate a response from the specialist adapter
    3. Optionally generate from the TIES-merged adapter for comparison
    """
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Check if the requested model scale matches
    if request.model_scale != engine.current_scale:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Requested scale {request.model_scale} but "
                f"{engine.current_scale} is loaded. "
                "Use /api/switch-model to switch."
            ),
        )

    total_start = time.perf_counter()

    # --- Step 1: Route ---
    if request.router_type == "keyword":
        route_result = keyword_route(request.query)
    else:
        if not engine.domain_centroids:
            engine.compute_centroids(SAMPLE_QUESTIONS)
        route_result = cosine_route(
            request.query,
            engine.model,
            engine.tokenizer,
            engine.domain_centroids,
        )

    domain = route_result["domain"]
    domain_info = DOMAIN_INFO.get(domain, {})

    routing = RoutingResult(
        domain=domain,
        domain_label=domain_info.get("label", domain),
        domain_icon=domain_info.get("icon", "❓"),
        domain_color=domain_info.get("color", "#888888"),
        keyword_matches=route_result.get("keyword_matches", {}),
        scores=route_result["scores"],
        confidence=route_result["confidence"],
        router_type=request.router_type,
    )

    # --- Step 2: Generate specialist response ---
    if domain not in engine.loaded_adapters:
        raise HTTPException(
            status_code=404,
            detail=f"Adapter for domain '{domain}' is not loaded.",
        )

    response_text, gen_time = engine.generate(
        query=request.query,
        adapter_name=domain,
        temperature=request.temperature,
    )

    specialist = GenerationResult(
        response=response_text,
        adapter_used=domain,
        generation_time_ms=gen_time,
    )

    # --- Step 3 (optional): Generate merged response ---
    merged = None
    if request.compare_merged and engine.merged_loaded:
        merged_text, merged_time = engine.generate(
            query=request.query,
            adapter_name="merged",
            temperature=request.temperature,
        )
        merged = GenerationResult(
            response=merged_text,
            adapter_used="merged (TIES)",
            generation_time_ms=merged_time,
        )

    total_ms = (time.perf_counter() - total_start) * 1000

    return QueryResponse(
        routing=routing,
        specialist=specialist,
        merged=merged,
        model_scale=request.model_scale,
        total_time_ms=round(total_ms, 1),
    )


@app.post("/api/switch-model", response_model=ModelStatus)
async def switch_model(request: SwitchModelRequest):
    """
    Switch the loaded model scale.

    This unloads the current model and loads the new one.
    Takes ~30-60 seconds depending on hardware.
    """
    logger.info("Switching model to %s...", request.model_scale)
    engine.load_model(request.model_scale)

    scale = engine.current_scale or request.model_scale
    return ModelStatus(
        model_scale=scale,
        model_id=MODEL_IDS.get(scale, "unknown"),
        loaded=engine.is_loaded,
        domains_loaded=engine.loaded_adapters,
        merged_loaded=engine.merged_loaded,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
