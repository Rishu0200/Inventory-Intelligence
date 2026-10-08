"""
LangChain @tool definitions.
"""
from __future__ import annotations
import pandas as pd
from langchain_core.tools import tool
from orchestrator.reorder_logic import get_reorder_result
from config import Paths
from db.session import get_session
from db.models import InventorySnapshot, DemandRecord, SupplierTerm, Supplier
from knowledge.feature_store.demand_model import forecast_sku, load_model
from knowledge.feature_store.anomaly_model import detect_anomalies
from knowledge.vector_store.retriever import retrieve, format_context
from config import settings


_xgb_model = None

def _get_model():
    global _xgb_model
    if _xgb_model is None:
        _xgb_model = load_model()
    return _xgb_model


@tool
def get_forecast(sku_id: str, horizon: int = 3) -> str:
    """
    Forecast monthly demand for a SKU over `horizon` months.
    """
    result = forecast_sku(sku_id, horizon=horizon, model=_get_model())
    if not result["forecast"]:
        return f"No forecast data available for SKU {sku_id}."
    lines = [f"Demand forecast for {sku_id} (next {horizon} months):"]
    for i, (fc, lo, hi) in enumerate(zip(result["forecast"], result["lower"], result["upper"]), 1):
        lines.append(f"  Month +{i}: {fc:.0f} units  (90% CI: {lo:.0f} – {hi:.0f})")
    return "\n".join(lines)


@tool
def check_stock(sku_id: str) -> str:
    """
    Check current stock level, reorder point, and days-of-stock for a SKU.
    """
    with get_session() as session:
        r = session.query(InventorySnapshot).filter_by(sku_id=sku_id).first()
        res = get_reorder_result(session, sku_id) if r is not None else None

        if r is None or res is None:
            return f"SKU {sku_id} not found in inventory."

        alert   = "⚠️ REORDER NEEDED" if res.needs_reorder else "✓ OK"
        return (
            f"Inventory — {sku_id} ({r.item_name}):\n"
            f"  On-Hand: {r.qty_on_hand}  |  WIP: {r.qty_wip}  |  In-Transit: {r.qty_in_transit}\n"
            f"  Total Available: {r.total_available}  |  Reorder Point: {r.reorder_point}({res.source})\n"
            f"  Days of Stock: {r.days_of_stock}  |  Status: {r.status}\n"
            f"  Gap vs ROP: {res.gap:+.0f} units  — {alert}"
        )


@tool
def compute_rop(sku_id: str) -> str:
    """
    Compute the Reorder Point (ROP) for a SKU:
    ROP = (avg_daily_demand × lead_time_days) + safety_stock
    Safety stock scales demand variability to the lead-time window.
    SKUs with no demand history use the inventory table's reorder point.
    """
    with get_session() as session:
        res = get_reorder_result(session, sku_id)

    if res is None:
        return f"SKU {sku_id} not found in inventory."

    alert = "⚠️ REORDER NOW" if res.needs_reorder else "✓ Stock OK"

    if res.source == "static":
        return (
            f"ROP Analysis — {sku_id}:\n"
            f"  No demand history, so using the inventory table's reorder point.\n"
            f"  Reorder point: {res.rop:.0f} units\n"
            f"  Current available: {res.current:.0f} units\n"
            f"  {alert}"
        )

    return (
        f"ROP Analysis — {sku_id}:\n"
        f"  Avg daily demand: {res.avg_daily:.1f} units\n"
        f"  Lead time: {res.lead_time:.0f} days"
        f"{' (assumed — no supplier record found)' if res.lead_time_assumed else ''}\n"
        f"  Safety stock (95% SL): {res.safety_stock:.0f} units\n"
        f"  Computed ROP: {res.rop:.0f} units\n"
        f"  Current available: {res.current:.0f} units\n"
        f"  {alert}"
    )


@tool
def retrieve_docs(query: str, doc_type: str = "all", k: int = 5) -> str:
    """
    Semantic search over PO and supplier catalog documents.
    doc_type: "PO", "catalog", or "all"
    "all" splits k evenly across both collections so PO results can't
    starve out catalog results.
    """
    from config import settings as s
    results = []

    if doc_type == "PO":
        results = retrieve(query, s.chroma_collection_pos, k=k)
    elif doc_type == "catalog":
        results = retrieve(query, s.chroma_collection_catalogs, k=k)
    else:
        half = max(1, k // 2)
        po_results  = retrieve(query, s.chroma_collection_pos, k=half)
        cat_results = retrieve(query, s.chroma_collection_catalogs, k=k - half)
        results = po_results + cat_results

    return format_context(results[:k])


@tool
def get_supplier_info(supplier_id: str) -> str:
    """
    Retrieve full supplier information including terms, lead time, and on-time rate.
    """
    with get_session() as session:
        d = session.get(Supplier, supplier_id)
        t = session.query(SupplierTerm).filter_by(supplier_id=supplier_id).first()

    if d is None and t is None:
        return f"Supplier {supplier_id} not found."

    parts = []
    if d is not None:
        parts.append(f"Supplier: {d.supplier_name} ({supplier_id})")
        parts.append(f"  City: {d.city}, {d.state}")
        parts.append(f"  Category: {d.category}")
        parts.append(f"  On-time rate: {d.on_time_rate_pct}%")
        parts.append(f"  Payment terms: {d.payment_terms}")
    if t is not None:
        parts.append(f"  Lead time: {t.lead_time_days} days")
        parts.append(f"  MOQ: {t.min_order_qty} units")
        parts.append(f"  Credit: {t.credit_days} days")
        parts.append(f"  Advance: {t.advance_pct}%")
        parts.append(f"  Penalty: {t.penalty_clause}")
        parts.append(f"  Notes: {t.notes}")
    return "\n".join(parts)


@tool
def detect_sku_anomaly(sku_id: str) -> str:
    """
    Detect demand anomalies for a specific SKU using Isolation Forest.
    """
    anomalies = detect_anomalies(sku_id=sku_id)
    if not anomalies:
        return f"No anomalies detected for SKU {sku_id}."
    lines = [f"Anomalies detected for {sku_id}:"]
    for a in anomalies[:5]:
        period = str(a.get("period", ""))[:7]
        lines.append(
            f"  {period}: {a['net_units']} units sold "
            f"(3-month avg: {a['roll_mean_3']:.0f}), score: {a['anomaly_score']:.3f}"
        )
    return "\n".join(lines)


ALL_TOOLS = [get_forecast, check_stock, compute_rop, retrieve_docs,
             get_supplier_info, detect_sku_anomaly]
