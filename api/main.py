"""
FastAPI application entry point.
Models and ChromaDB are loaded once at startup via lifespan context.
"""
from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.routes import query, alerts, forecast
from api.schemas import HealthResponse


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
        print("⚙️  Loading ML models (background)...")
        try:
            from knowledge.feature_store.demand_model import load_model
            app_state.xgb_model = load_model()
            app_state.models_ready = app_state.xgb_model is not None
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────────

app.include_router(query.router,    prefix="/api", tags=["Query"])
app.include_router(alerts.router,   prefix="/api", tags=["Alerts"])
app.include_router(forecast.router, prefix="/api", tags=["Forecast"])


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/ping", response_model=HealthResponse, tags=["Health"])
def ping():
    return HealthResponse(
        status="ok",
        demo_mode=settings.demo_mode,
        models_ready=app_state.models_ready,
        chroma_docs=app_state.chroma_docs,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host,
                port=settings.api_port, reload=True)
