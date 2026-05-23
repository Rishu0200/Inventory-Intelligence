"""
GET /api/forecast/{sku_id} — Demand forecast endpoint.
Returns point forecast + 90% confidence intervals for N months ahead.
"""
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query

from api.schemas import ForecastResponse, ForecastPoint
from knowledge.feature_store.demand_model import forecast_sku, load_model

router = APIRouter()
_model = None   # lazy-loaded


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@router.get("/forecast/{sku_id}", response_model=ForecastResponse)
def get_forecast_endpoint(
    sku_id:  str,
    horizon: int = Query(default=3, ge=1, le=12,
                         description="Forecast horizon in months (1-12)"),
):
    """
    Monthly demand forecast for a specific SKU.

    **sku_id examples:** SC-001, TBP-001, RSH-001, CHM-001

    Returns point forecast and 90% confidence intervals for each future month.
    If the model isn't trained yet, falls back to a rolling-mean estimate.
    """
    result = forecast_sku(sku_id=sku_id, horizon=horizon, model=_get_model())

    if not result["forecast"]:
        raise HTTPException(
            status_code=404,
            detail=f"SKU '{sku_id}' not found in demand history. "
                   f"Check data/raw/demand_history.csv for valid SKU IDs.",
        )

    points = [
        ForecastPoint(
            month_offset=i + 1,
            forecast=fc,
            lower_ci=lo,
            upper_ci=hi,
        )
        for i, (fc, lo, hi) in enumerate(
            zip(result["forecast"], result["lower"], result["upper"])
        )
    ]

    return ForecastResponse(
        sku_id=sku_id,
        horizon=horizon,
        points=points,
        model="XGBoost" if _get_model() else "Rolling Mean (fallback)",
    )


@router.get("/forecast", response_model=list[ForecastResponse], tags=["Forecast"])
def get_all_forecasts(
    horizon: int = Query(default=3, ge=1, le=6),
):
    """
    Forecast for all SKUs in demand history.
    Useful for dashboard overview.
    """
    import pandas as pd
    from config import Paths

    try:
        df = pd.read_csv(Paths.DATA_RAW / "demand_history.csv")
        sku_ids = df["sku_id"].unique().tolist()
    except Exception:
        sku_ids = ["SC-001", "TBP-001", "PBP-001", "RSH-001", "CHM-001"]

    model = _get_model()
    results = []
    for sku in sku_ids:
        r = forecast_sku(sku_id=sku, horizon=horizon, model=model)
        if r["forecast"]:
            results.append(ForecastResponse(
                sku_id=sku,
                horizon=horizon,
                points=[
                    ForecastPoint(month_offset=i+1, forecast=fc, lower_ci=lo, upper_ci=hi)
                    for i, (fc, lo, hi) in enumerate(zip(r["forecast"], r["lower"], r["upper"]))
                ],
                model="XGBoost" if model else "Rolling Mean",
            ))
    return results
