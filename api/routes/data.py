"""
Read-only data endpoints for the frontend, so it never touches the database
or CSVs directly (required on Streamlit Cloud, which shares no filesystem
with the API).
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas import InventoryItem, DemandPoint, DemandHistoryResponse
from auth.dependencies import get_current_user
from db.models import User, InventorySnapshot, DemandRecord
from db.session import get_session

router = APIRouter()


@router.get("/skus", response_model=list[str])
def list_skus(user: User = Depends(get_current_user)):
    """All SKU ids that have demand history (used for the forecast dropdown)."""
    with get_session() as session:
        rows = session.query(DemandRecord.sku_id).distinct().all()
    return sorted(r[0] for r in rows)


@router.get("/inventory", response_model=list[InventoryItem])
def get_inventory(user: User = Depends(get_current_user)):
    """Current inventory snapshot, one row per SKU."""
    with get_session() as session:
        rows = session.query(InventorySnapshot).order_by(InventorySnapshot.sku_id).all()
        return [
            InventoryItem(
                sku_id=r.sku_id,
                item_name=r.item_name or r.sku_id,
                qty_on_hand=r.qty_on_hand or 0,
                total_available=r.total_available or 0,
                reorder_point=r.reorder_point or 0,
                days_of_stock=r.days_of_stock,
                status=r.status or "",
            )
            for r in rows
        ]


@router.get("/demand/{sku_id}", response_model=DemandHistoryResponse)
def get_demand_history(
    sku_id: str,
    months: int = Query(default=12, ge=1, le=60),
    user: User = Depends(get_current_user),
):
    """Most recent `months` of net units for one SKU, oldest first."""
    with get_session() as session:
        rows = (
            session.query(DemandRecord.period, DemandRecord.net_units)
            .filter(DemandRecord.sku_id == sku_id)
            .order_by(DemandRecord.period.desc())
            .limit(months)
            .all()
        )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No demand history for SKU '{sku_id}'.")

    rows = sorted(rows, key=lambda r: r[0])
    return DemandHistoryResponse(
        sku_id=sku_id,
        points=[DemandPoint(period=r[0].strftime("%Y-%m"), net_units=float(r[1])) for r in rows],
    )