"""
GET /api/alerts — Returns current reorder and anomaly alerts for all SKUs.
Results cached for 10 minutes to avoid repeated model inference.
"""
from __future__ import annotations
import pandas as pd
from fastapi import APIRouter, Depends
from datetime import datetime, timedelta
from api.schemas import AlertItem, AlertsResponse
from config import Paths
from auth.dependencies import get_current_user
from db.models import User, InventorySnapshot
from db.session import get_session
from cache.redis_client import cache_get, cache_set, cache_delete
from orchestrator.reorder_logic import get_all_reorder_results

router = APIRouter()

_CACHE_KEY = "alerts:all"
_CACHE_TTL_SECONDS = 600

# ── Simple in-memory TTL cache ────────────────────────────────────────────────
_cache: dict = {}
_TTL = timedelta(minutes=10)


def _get_cached(key: str):
    if key in _cache:
        data, ts = _cache[key]
        if datetime.now() - ts < _TTL:
            return data
    return None


def _set_cached(key: str, data):
    _cache[key] = (data, datetime.now())


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/alerts", response_model=AlertsResponse)
def get_alerts(refresh: bool = False, user: User = Depends(get_current_user)):
    if refresh:
        cache_delete(_CACHE_KEY)
    else:
        cached = cache_get(_CACHE_KEY)
        if cached:
            return AlertsResponse(**cached)

    alerts: list[AlertItem] = []
    alerts += _reorder_alerts()
    alerts += _anomaly_alerts()

    # Sort: high severity first
    severity_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 3))

    response = AlertsResponse(
        total_alerts=len(alerts),
        reorder_alerts=sum(1 for a in alerts if a.alert_type == "reorder"),
        anomaly_alerts=sum(1 for a in alerts if a.alert_type == "anomaly"),
        alerts=alerts,
    )
    cache_set(_CACHE_KEY, response.model_dump(mode="json"), ttl_seconds=_CACHE_TTL_SECONDS)
    return response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reorder_alerts() -> list[AlertItem]:
    alerts: list[AlertItem] = []
    try:
        with get_session() as session:
            results = get_all_reorder_results(session)

            for r in results:
                if not r.needs_reorder:
                    continue

                ratio = (r.current / r.rop) if r.rop else 0
                severity = "high" if ratio < -0.5 else "medium" if ratio < 0.8 else "low"
                alerts.append(AlertItem(
                sku_id=r.sku_id,
                item_name=r.item_name,
                alert_type="reorder",
                severity=severity,
                current_stock=r.current,
                reorder_point=round(r.rop, 1),
                gap=round(r.gap, 1),
                message=(
                    f"Stock {r.current:.0f} ≤ ROP {r.rop:.0f} ({r.source}). "
                    f"Place order immediately."
                ),
            ))
    except Exception as e:
        print(f"[alerts] Reorder scan failed: {e}")
    return alerts


def _anomaly_alerts() -> list[AlertItem]:
    alerts: list[AlertItem] = []
    try:
        from knowledge.feature_store.anomaly_model import detect_anomalies, load_anomaly_model
        model, _ = load_anomaly_model()
        if model is None:
            return []

        anomalies = detect_anomalies(sku_id=None)
        for a in anomalies[:10]:    # cap at 10
            score    = float(a.get("anomaly_score", 0))
            severity = "high" if score < -0.2 else "medium" if score < -0.1 else "low"
            period   = str(a.get("period", ""))[:7]
            alerts.append(AlertItem(
                sku_id=str(a["sku_id"]),
                item_name=str(a.get("product_name", a["sku_id"]))[:30],
                alert_type="anomaly",
                severity=severity,
                anomaly_score=round(score, 4),
                message=(
                    f"Unusual demand in {period}: "
                    f"{a['net_units']} units vs 3M avg {a['roll_mean_3']:.0f}."
                ),
            ))
    except Exception as e:
        print(f"[alerts] Anomaly scan failed: {e}")
    return alerts
