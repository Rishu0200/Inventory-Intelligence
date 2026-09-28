"""
FastAPI application entry point.
Models and ChromaDB are loaded once at startup via lifespan context.
"""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.schemas import HealthResponse
from api.routes import query, alerts, forecast, auth as auth_routes, data as data_routes


# ── App state (shared across requests) ───────────────────────────────────────

class AppState:
    xgb_model   = None
    anomaly_model = None
    anomaly_scaler = None
    chroma_docs  = 0
    models_ready = False


app_state = AppState()


# ── Lifespan: load heavy resources once ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    def _load():
        from cache.redis_client import cache_set
        print("⚙️  Loading ML models (background)...")
        try:
            from knowledge.feature_store.demand_model import load_model
            app_state.xgb_model = load_model()
            app_state.models_ready = app_state.xgb_model is not None
            cache_set("models_ready", app_state.models_ready, ttl_seconds=None)
            print(f"   XGBoost: {'✓' if app_state.models_ready else '✗ not found'}")
        except Exception as e:
            print(f"   XGBoost load failed: {e}")

        try:
            from knowledge.feature_store.anomaly_model import load_anomaly_model
            app_state.anomaly_model, app_state.anomaly_scaler = load_anomaly_model()
            print(f"   Isolation Forest: {'✓' if app_state.anomaly_model else '✗ not found'}")
        except Exception as e:
            print(f"   Anomaly model load failed: {e}")

        try:
            from knowledge.vector_store.embedder import collection_count
            app_state.chroma_docs = collection_count(settings.chroma_collection_pos)
            print(f"   ChromaDB: {app_state.chroma_docs} chunks")
        except Exception as e:
            print(f"   ChromaDB not loaded: {e}")

        print(f"🚀  Models ready — demo_mode={'ON' if settings.demo_mode else 'OFF'}")

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    print("🚀  API responding. Models loading in background...")
    yield
    print("🛑  Shutting down.")


    



# ── App creation ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Inventory Intelligence API",
    description=(
        "Agentic AI + RAG inventory system for Uninox Houseware.\n\n"
        "Built with LangGraph · ChromaDB · XGBoost · FastAPI"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
# ── Mount routers ─────────────────────────────────────────────────────────────

app.include_router(query.router,    prefix="/api", tags=["Query"])
app.include_router(alerts.router,   prefix="/api", tags=["Alerts"])
app.include_router(forecast.router, prefix="/api", tags=["Forecast"])
app.include_router(auth_routes.router, prefix="/api", tags=["Auth"])
app.include_router(data_routes.router, prefix="/api", tags=["Data"])


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/ping", response_model=HealthResponse, tags=["Health"])
def ping():
    from cache.redis_client import cache_get
    # Check Redis first (works across processes/instances); fall back to this
    # process's own in-memory flag if Redis has nothing yet (e.g. right at boot).
    models_ready = cache_get("models_ready")
    if models_ready is None:
        models_ready = app_state.models_ready
    return HealthResponse(
        status="ok",
        demo_mode=settings.demo_mode,
        models_ready=bool(models_ready),
        chroma_docs=app_state.chroma_docs,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host,
                port=settings.api_port, reload=True)
