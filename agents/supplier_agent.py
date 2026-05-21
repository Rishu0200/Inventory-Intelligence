"""
Supplier Agent.
Retrieves supplier terms, catalog info, and recommends best vendor for a SKU.
"""
from __future__ import annotations
import pandas as pd
from orchestrator.tools import get_supplier_info, retrieve_docs
from config import Paths, settings


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

    # Add comparison/recommendation if multiple suppliers
    if len(supplier_ids) > 1 and not settings.use_llm:
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
        terms = pd.read_csv(Paths.DATA_RAW / "supplier_terms.csv")
        mask  = terms["skus_supplied"].fillna("").str.contains(sku_id, case=False)
        return terms[mask]["supplier_id"].tolist()
    except Exception:
        return []


def _all_supplier_ids() -> list[str]:
    try:
        direc = pd.read_csv(Paths.DATA_RAW / "supplier_directory.csv")
        return direc["supplier_id"].tolist()[:5]
    except Exception:
        return []


def _recommend_best(supplier_ids: list[str], sku_id: str) -> str:
    """Simple rule-based recommendation: shortest lead time + best on-time rate."""
    try:
        terms = pd.read_csv(Paths.DATA_RAW / "supplier_terms.csv")
        direc = pd.read_csv(Paths.DATA_RAW / "supplier_directory.csv")
        merged = terms.merge(direc, on="supplier_id", how="left")
        subset = merged[merged["supplier_id"].isin(supplier_ids)].copy()

        if subset.empty:
            return ""

        subset["score"] = (
            -subset["lead_time_days"].rank() +
            subset["on_time_rate_pct"].rank()
        )
        best = subset.loc[subset["score"].idxmax()]
        return (
            f"🏆 Recommended supplier for {sku_id}: "
            f"{best.get('supplier_name', best['supplier_id'])} "
            f"(Lead time: {best['lead_time_days']} days, "
            f"On-time rate: {best['on_time_rate_pct']}%)"
        )
    except Exception:
        return ""
