"""
Supplier Agent.
Retrieves supplier terms, catalog info, and recommends best vendor for a SKU.
"""
from __future__ import annotations
import pandas as pd
from orchestrator.tools import get_supplier_info, retrieve_docs
from config import Paths, settings
from db.session import get_session
from db.models import Supplier, SupplierTerm

def supplier_agent_node(state: dict) -> dict:
    """
    LangGraph node — answers supplier-related queries.
    """
    sku_id = state.get("sku_id", "")
    query  = state.get("query", "")

    # Find supplier(s) for this SKU
    supplier_ids = _find_suppliers_for_sku(sku_id) if sku_id else _all_supplier_ids()

    # Build supplier info string
    if supplier_ids:
        parts = []
        for sup_id in supplier_ids[:3]:   # max 3 suppliers
            try:
                info = get_supplier_info.invoke({"supplier_id": sup_id})
                parts.append(info)
            except Exception:
                pass
        supplier_text = "\n\n".join(parts) if parts else "No supplier data found."
    else:
        supplier_text = "No supplier mapping found for the specified SKU."


    if len(supplier_ids) > 1:
        supplier_text += "\n\n" + _recommend_best(supplier_ids, sku_id)

    # RAG: retrieve catalog documents
    rag_result = ""
    try:
        rag_query  = f"supplier catalog pricing terms {sku_id} {query[:60]}"
        rag_result = retrieve_docs.invoke({
            "query":    rag_query,
            "doc_type": "catalog",
            "k":        3,
        })
    except Exception:
        pass

    return {"tool_result": supplier_text, "rag_context": rag_result}


def _find_suppliers_for_sku(sku_id: str) -> list[str]:
    """Return supplier IDs that supply a given SKU."""
    try:
        with get_session() as session:
            rows = session.query(SupplierTerm.supplier_id, SupplierTerm.skus_supplied).all()
        return [
            r.supplier_id for r in rows
            if r.skus_supplied and sku_id.lower() in r.skus_supplied.lower()
        ]
    except Exception:
        return []

def _all_supplier_ids() -> list[str]:
    try:
        with get_session() as session:
            rows = session.query(Supplier.supplier_id).limit(5).all()
        return [r[0] for r in rows]
    except Exception:
        return []


def _recommend_best(supplier_ids: list[str], sku_id: str) -> str:
    """Simple rule-based recommendation: shortest lead time + best on-time rate."""
    try:
        with get_session() as session:
            rows = (
                session.query(Supplier, SupplierTerm)
                .join(SupplierTerm, SupplierTerm.supplier_id == Supplier.supplier_id)
                .filter(Supplier.supplier_id.in_(supplier_ids))
                .all()
            )

        if not rows:
            return ""

        lead_times = pd.Series([term.lead_time_days for _, term in rows])
        on_times   = pd.Series([sup.on_time_rate_pct for sup, _ in rows])
        scores     = (-lead_times.rank() + on_times.rank())

        best_idx = int(scores.idxmax())
        best_sup, best_term = rows[best_idx]

        return (
            f"🏆 Recommended supplier for {sku_id}: "
            f"{best_sup.supplier_name} "
            f"(Lead time: {best_term.lead_time_days} days, "
            f"On-time rate: {best_sup.on_time_rate_pct}%)"
        )
    except Exception:
        return ""
