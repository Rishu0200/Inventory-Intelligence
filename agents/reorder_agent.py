"""
Reorder Point Agent.
Checks current stock vs computed ROP and raises alerts.
"""
from __future__ import annotations
import pandas as pd
from orchestrator.tools import check_stock, compute_rop, retrieve_docs
from config import settings, Paths
from db.session import get_session
from db.models import InventorySnapshot


def reorder_agent_node(state: dict) -> dict:
    """
    LangGraph node — handles stock-level and reorder-point queries.
    If a specific SKU is mentioned, analyses that SKU only.
    Otherwise, scans all SKUs and returns a summary of alerts.
    """
    sku_id = state.get("sku_id", "")

    if sku_id:
        result = _analyse_single(sku_id)
    else:
        result = _scan_all_skus()

    # RAG context: pull relevant PO documents for the SKU
    rag_result = ""
    try:
        rag_key = f"purchase order {sku_id} reorder" if sku_id else "low stock reorder"
        rag_result = retrieve_docs.invoke({"query": rag_key, "doc_type": "PO", "k": 3})
    except Exception:
        pass

    return {"tool_result": result, "rag_context": rag_result}


def _analyse_single(sku_id: str) -> str:
    try:
        stock_info = check_stock.invoke({"sku_id": sku_id})
        rop_info   = compute_rop.invoke({"sku_id": sku_id})
        return f"{stock_info}\n\n{rop_info}"
    except Exception as e:
        return f"Could not analyse {sku_id}: {e}"


def _scan_all_skus() -> str:
    """Reads the inventory snapshot from the database, not a CSV."""
    try:
        with get_session() as session:
            rows = session.query(InventorySnapshot).all()
        alerts = [r for r in rows if r.total_available <= r.reorder_point]

        if not alerts:
            return "✓ All SKUs are above their reorder points. No immediate action needed."

        lines = [f"⚠️  {len(alerts)} SKU(s) at or below reorder point:\n"]
        for row in alerts:
            gap = row.total_available - row.reorder_point
            lines.append(
                f"  • {row.sku_id} ({row.item_name}): "
                f"Available={row.total_available:.0f}  ROP={row.reorder_point:.0f}  "
                f"Gap={gap:+.0f}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Unable to scan inventory: {e}"
