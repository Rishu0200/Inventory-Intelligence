from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from db.models import DemandRecord, InventorySnapshot, SupplierTerm

SERVICE_Z = 1.64              
DEFAULT_LEAD_TIME_DAYS = 30.0


@dataclass
class ReorderResult:
    sku_id: str
    item_name: str
    current: float
    rop: float
    source: str                       
    avg_daily: float | None = None
    safety_stock: float | None = None
    lead_time: float | None = None
    lead_time_assumed: bool = False
    status: str = ""

    @property
    def gap(self) -> float:
        return self.current - self.rop

    @property
    def needs_reorder(self) -> bool:
        return self.current <= self.rop


def _demand_stats(session, sku_ids=None) -> dict[str, tuple[float, float]]:
    """sku_id -> (avg_monthly_units, std_monthly_units)"""
    q = session.query(DemandRecord.sku_id, DemandRecord.net_units)
    if sku_ids is not None:
        q = q.filter(DemandRecord.sku_id.in_(sku_ids))
    rows = q.all()
    if not rows:
        return {}
    df = pd.DataFrame([(r[0], r[1]) for r in rows], columns=["sku_id", "net_units"])
    g = df.groupby("sku_id")["net_units"]
    avg, std = g.mean(), g.std(ddof=0)
    return {sku: (float(avg[sku]), float(std[sku])) for sku in avg.index}


def _lead_times(session) -> dict[str, float]:
    """sku_id -> supplier lead time in days (skus_supplied is '; '-separated)."""
    out: dict[str, float] = {}
    for skus, lead in session.query(
        SupplierTerm.skus_supplied, SupplierTerm.lead_time_days
    ).all():
        for sku in (skus or "").split(";"):
            sku = sku.strip()
            if sku and lead is not None:
                out.setdefault(sku, float(lead))
    return out


def _build(inv, stats: dict, lead_times: dict) -> ReorderResult:
    current = float(inv.total_available or 0)
    item_name = inv.item_name or inv.sku_id
    status = inv.status or ""

    stat = stats.get(inv.sku_id)
    if stat is None:
        return ReorderResult(
            sku_id=inv.sku_id, item_name=item_name, current=current,
            rop=float(inv.reorder_point or 0), source="static", status=status,
        )

    avg_monthly, std_monthly = stat
    lead = lead_times.get(inv.sku_id)
    assumed = lead is None
    if assumed:
        lead = DEFAULT_LEAD_TIME_DAYS

    avg_daily = avg_monthly / 30.0
    std_daily = std_monthly / 30.0
    safety = SERVICE_Z * std_daily * (lead ** 0.5)
    rop = avg_daily * lead + safety

    return ReorderResult(
        sku_id=inv.sku_id, item_name=item_name, current=current, rop=rop,
        source="computed", avg_daily=avg_daily, safety_stock=safety,
        lead_time=lead, lead_time_assumed=assumed, status=status,
    )


def get_reorder_result(session, sku_id: str) -> ReorderResult | None:
    inv = session.query(InventorySnapshot).filter_by(sku_id=sku_id).first()
    if inv is None:
        return None
    return _build(inv, _demand_stats(session, [sku_id]), _lead_times(session))


def get_all_reorder_results(session) -> list[ReorderResult]:
    invs = session.query(InventorySnapshot).order_by(InventorySnapshot.sku_id).all()
    stats, leads = _demand_stats(session), _lead_times(session)
    return [_build(i, stats, leads) for i in invs]