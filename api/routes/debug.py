"""
GET /api/debug/connectivity — admin-only, reports whether each external
service (DB, Chroma Cloud) actually responds. Temporary — remove once the
current issue is diagnosed.
"""
from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy import text

from auth.dependencies import require_admin
from db.models import User, SupplierTerm
from db.session import get_session

router = APIRouter()


@router.get("/debug/connectivity")
def check_connectivity(user: User = Depends(require_admin)):
    results = {}

    try:
        with get_session() as s:
            s.execute(text("SELECT 1"))
        results["database_connection"] = "ok"
    except Exception as e:
        results["database_connection"] = f"FAILED: {e}"

    try:
        with get_session() as s:
            count = s.query(SupplierTerm).count()
            rsh = s.query(SupplierTerm).filter(
                SupplierTerm.skus_supplied.contains("RSH-001")
            ).first()
        results["supplier_terms_row_count"] = count
        results["rsh_001_match"] = rsh.supplier_id if rsh else "NOT FOUND"
    except Exception as e:
        results["supplier_terms_query"] = f"FAILED: {e}"

    try:
        from knowledge.vector_store.embedder import collection_count
        from config import settings
        results["chroma_catalog_doc_count"] = collection_count(settings.chroma_collection_catalogs)
    except Exception as e:
        results["chroma_catalog_doc_count"] = f"FAILED: {e}"

    return results