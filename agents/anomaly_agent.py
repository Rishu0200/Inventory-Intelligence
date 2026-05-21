"""
Anomaly Detection Agent.
Runs Isolation Forest, then RAG-retrieves historical context to explain anomalies.
"""
from __future__ import annotations
from orchestrator.tools import detect_sku_anomaly, retrieve_docs
from config import settings


def anomaly_agent_node(state: dict) -> dict:
    """
    LangGraph node — detects and explains demand/inventory anomalies.
    """
    sku_id = state.get("sku_id", "")
    query  = state.get("query", "")

    # ── Detect anomalies ──────────────────────────────────────────────────────
    if sku_id:
        try:
            anomaly_str = detect_sku_anomaly.invoke({"sku_id": sku_id})
        except Exception as e:
            anomaly_str = f"Anomaly detection failed for {sku_id}: {e}"
    else:
        anomaly_str = _scan_all_anomalies()

    # ── RAG: retrieve context around flagged periods ──────────────────────────
    rag_result = ""
    try:
        rag_query  = f"unusual demand spike stockout {sku_id} {query[:40]}"
        rag_result = retrieve_docs.invoke({
            "query":    rag_query,
            "doc_type": "all",
            "k":        4,
        })
    except Exception:
        pass

    # ── Non-LLM structured response ───────────────────────────────────────────
    if not settings.use_llm:
        response = f"🔍 Anomaly Analysis\n\n{anomaly_str}"
        if rag_result and rag_result != "No relevant documents found.":
            response += f"\n\n📄 Historical Context:\n{rag_result[:500]}"
        return {"tool_result": response, "rag_context": rag_result}

    return {"tool_result": anomaly_str, "rag_context": rag_result}


def _scan_all_anomalies() -> str:
    """Scan all SKUs for anomalies and return a summary."""
    try:
        from knowledge.feature_store.anomaly_model import detect_anomalies, load_anomaly_model
        model, scaler = load_anomaly_model()

        if model is None:
            return (
                "⚠️ Anomaly model not trained yet.\n"
                "Run: python scripts/train_models.py"
            )

        anomalies = detect_anomalies(sku_id=None)
        if not anomalies:
            return "✓ No anomalies detected across all SKUs."

        lines = [f"🚨 {len(anomalies)} anomalous data point(s) detected:\n"]
        for a in anomalies[:8]:
            period = str(a.get("period", ""))[:7]
            lines.append(
                f"  • {a['sku_id']} ({a.get('product_name','')[:20]}) "
                f"| Period: {period} "
                f"| Units: {a['net_units']} "
                f"| 3M-Avg: {a['roll_mean_3']:.0f} "
                f"| Score: {a['anomaly_score']:.3f}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Anomaly scan failed: {e}"
