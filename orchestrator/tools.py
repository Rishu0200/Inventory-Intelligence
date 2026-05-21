"""
LangChain @tool definitions.
All agents import tools from here — single source of truth for signatures.
"""
from __future__ import annotations
import pandas as pd
from langchain_core.tools import tool

from config import Paths
from knowledge.feature_store.demand_model import forecast_sku, load_model
from knowledge.feature_store.anomaly_model import detect_anomalies
from knowledge.vector_store.retriever import retrieve, format_context
from config import settings


# Pre-load model once (shared across tool calls)
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
    Returns a formatted string with forecast values and confidence intervals.
    Example: get_forecast("TBW-001", horizon=3)
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
    Returns whether the SKU needs reordering.
    Example: check_stock("RSH-001")
    """
    df = pd.read_csv(Paths.DATA_RAW / "inventory_history.csv")
    row = df[df["sku_id"] == sku_id]
    if row.empty:
        return f"SKU {sku_id} not found in inventory."
    r = row.iloc[0]
    rop_gap = r["total_available"] - r["reorder_point"]
    alert   = "⚠️ REORDER NEEDED" if rop_gap <= 0 else "✓ OK"
    return (
        f"Inventory — {sku_id} ({r['item_name']}):\n"
        f"  On-Hand: {r['qty_on_hand']}  |  WIP: {r['qty_wip']}  |  In-Transit: {r['qty_in_transit']}\n"
        f"  Total Available: {r['total_available']}  |  Reorder Point: {r['reorder_point']}\n"
        f"  Days of Stock: {r['days_of_stock']}  |  Status: {r['status']}\n"
        f"  Gap vs ROP: {rop_gap:+.0f} units  — {alert}"
    )


@tool
def compute_rop(sku_id: str) -> str:
    """
    Compute the Reorder Point (ROP) for a SKU using:
    ROP = (avg_daily_demand × lead_time_days) + safety_stock
    Example: compute_rop("TBP-001")
    """
    demand_df = pd.read_csv(Paths.DATA_RAW / "demand_history.csv")
    terms_df  = pd.read_csv(Paths.DATA_RAW / "supplier_terms.csv")
    inv_df    = pd.read_csv(Paths.DATA_RAW / "inventory_history.csv")

    sku_demand = demand_df[demand_df["sku_id"] == sku_id]["net_units"]
    if sku_demand.empty:
        return f"No demand history for {sku_id}."

    avg_monthly  = sku_demand.mean()
    avg_daily    = avg_monthly / 30.0
    std_monthly  = sku_demand.std(ddof=0)
    safety_stock = 1.64 * std_monthly   # 95% service level

    # Lead time from supplier_terms
    term_row = terms_df[terms_df["skus_supplied"].str.contains(sku_id, na=False)]
    lead_time = float(term_row["lead_time_days"].iloc[0]) if not term_row.empty else 30.0

    rop = avg_daily * lead_time + safety_stock
    inv_row = inv_df[inv_df["sku_id"] == sku_id]
    current = float(inv_row["total_available"].iloc[0]) if not inv_row.empty else 0

    alert = "⚠️ REORDER NOW" if current < rop else "✓ Stock OK"
    return (
        f"ROP Analysis — {sku_id}:\n"
        f"  Avg daily demand: {avg_daily:.1f} units\n"
        f"  Lead time: {lead_time:.0f} days\n"
        f"  Safety stock (95% SL): {safety_stock:.0f} units\n"
        f"  Computed ROP: {rop:.0f} units\n"
        f"  Current available: {current:.0f} units\n"
        f"  {alert}"
    )


@tool
def retrieve_docs(query: str, doc_type: str = "all", k: int = 5) -> str:
    """
    Semantic search over PO and supplier catalog documents.
    doc_type: "PO", "catalog", or "all"
    Example: retrieve_docs("payment terms Krishna Basket", doc_type="catalog")
    """
    from config import settings as s
    results = []
    if doc_type in ("PO", "all"):
        results += retrieve(query, s.chroma_collection_pos,   k=k, where={"doc_type": "PO"} if doc_type == "PO" else None)
    if doc_type in ("catalog", "all"):
        results += retrieve(query, s.chroma_collection_catalogs, k=k)
    return format_context(results[:k])


@tool
def get_supplier_info(supplier_id: str) -> str:
    """
    Retrieve full supplier information including terms, lead time, and on-time rate.
    Example: get_supplier_info("SUP-001")
    """
    terms = pd.read_csv(Paths.DATA_RAW / "supplier_terms.csv")
    direc = pd.read_csv(Paths.DATA_RAW / "supplier_directory.csv")

    t = terms[terms["supplier_id"] == supplier_id]
    d = direc[direc["supplier_id"] == supplier_id]

    if t.empty and d.empty:
        return f"Supplier {supplier_id} not found."

    parts = []
    if not d.empty:
        r = d.iloc[0]
        parts.append(f"Supplier: {r['supplier_name']} ({supplier_id})")
        parts.append(f"  City: {r['city']}, {r['state']}")
        parts.append(f"  Category: {r['category']}")
        parts.append(f"  On-time rate: {r['on_time_rate_pct']}%")
        parts.append(f"  Payment terms: {r['payment_terms']}")
    if not t.empty:
        r = t.iloc[0]
        parts.append(f"  Lead time: {r['lead_time_days']} days")
        parts.append(f"  MOQ: {r['min_order_qty']} units")
        parts.append(f"  Credit: {r['credit_days']} days")
        parts.append(f"  Advance: {r['advance_pct']}%")
        parts.append(f"  Penalty: {r['penalty_clause']}")
        parts.append(f"  Notes: {r['notes']}")
    return "\n".join(parts)


@tool
def detect_sku_anomaly(sku_id: str) -> str:
    """
    Detect demand anomalies for a specific SKU using Isolation Forest.
    Example: detect_sku_anomaly("SC-001")
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
