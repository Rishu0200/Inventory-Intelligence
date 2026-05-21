"""
Demand Forecasting Agent.
Calls get_forecast() + retrieve_docs() to answer demand-related queries.
"""
from __future__ import annotations
from orchestrator.tools import get_forecast, retrieve_docs
from config import settings


def demand_agent_node(state: dict) -> dict:
    """
    LangGraph node — handles demand forecasting queries.
    Returns updated state with tool_result and rag_context.
    """
    sku_id = state.get("sku_id", "")
    query  = state.get("query", "")

    # ── If no SKU identified, pick the top SKU by demand ─────────────────────
    if not sku_id:
        sku_id = _pick_top_sku()

    # ── Call forecast tool ────────────────────────────────────────────────────
    try:
        forecast_str = get_forecast.invoke({"sku_id": sku_id, "horizon": 3})
    except Exception as e:
        forecast_str = f"Forecast unavailable for {sku_id}: {e}"

    # ── RAG: retrieve seasonal/historical context ─────────────────────────────
    try:
        rag_query  = f"demand history seasonal sales {sku_id}"
        rag_result = retrieve_docs.invoke({
            "query":    rag_query,
            "doc_type": "PO",
            "k":        3,
        })
    except Exception:
        rag_result = ""

    # ── If no LLM, build a structured text response directly ─────────────────
    if not settings.use_llm:
        response = (
            f"📊 Demand Forecast — {sku_id}\n\n"
            f"{forecast_str}\n\n"
            f"{'📄 Historical Context:' + chr(10) + rag_result[:400] if rag_result else ''}"
        ).strip()
        return {"tool_result": response, "rag_context": rag_result}

    return {"tool_result": forecast_str, "rag_context": rag_result}


def _pick_top_sku() -> str:
    """Return the SKU with highest average demand from history."""
    try:
        import pandas as pd
        from config import Paths
        df = pd.read_csv(Paths.DATA_RAW / "demand_history.csv")
        top = df.groupby("sku_id")["net_units"].mean().idxmax()
        return str(top)
    except Exception:
        return "SC-001"
