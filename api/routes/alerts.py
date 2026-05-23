"""
GET /api/alerts — Returns current reorder and anomaly alerts for all SKUs.
Results cached for 10 minutes to avoid repeated model inference.
"""
from __future__ import annotations
from datetime import datetime, timedelta

import pandas as pd
from fastapi import APIRouter

from api.schemas import AlertItem, AlertsResponse
from config import Paths

router = APIRouter()

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
def get_alerts(refresh: bool = False):
    """
    Scan all SKUs and return:
    - Reorder alerts: SKUs at or below their reorder point
    - Anomaly alerts: SKUs flagged by Isolation Forest

    Use ?refresh=true to bypass the 10-minute cache.
    """
    cached = None if refresh else _get_cached("alerts")
    if cached:
        return cached

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
    _set_cached("alerts", response)
    return response


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reorder_alerts() -> list[AlertItem]:
    alerts = []
    try:
        inv = pd.read_csv(Paths.DATA_RAW / "inventory_history.csv")
        below_rop = inv[inv["total_available"] <= inv["reorder_point"]]

        for _, row in below_rop.iterrows():
            gap      = float(row["total_available"] - row["reorder_point"])
            severity = "high" if gap < -50 else "medium" if gap < 0 else "low"
            alerts.append(AlertItem(
                sku_id=str(row["sku_id"]),
                item_name=str(row.get("item_name", row["sku_id"])),
                alert_type="reorder",
                severity=severity,
                current_stock=float(row["total_available"]),
                reorder_point=float(row["reorder_point"]),
                gap=round(gap, 1),
                message=(
                    f"Stock {row['total_available']:.0f} ≤ ROP {row['reorder_point']:.0f}. "
                    f"Place order immediately ({row.get('status','')})."
                ),
            ))
    except Exception as e:
        print(f"[alerts] Reorder scan failed: {e}")
    return alerts


def _anomaly_alerts() -> list[AlertItem]:
    alerts = []
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
