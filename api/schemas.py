"""
Pydantic v2 models for all API I/O.
Auto-generates OpenAPI docs at /docs — screenshot this for your portfolio.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


# ── Request models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          description="Natural language inventory query",
                          examples=["How many units of TBP-001 will we need next month?"])


# ── Response models ────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    question:        str
    intent:          str = Field(description="Classified intent: demand/reorder/supplier/anomaly")
    sku_id:          str = Field(description="Extracted SKU ID, empty if not found")
    answer:          str
    rag_context_used: bool


class ForecastPoint(BaseModel):
    month_offset: int    = Field(description="+1, +2, +3 months from today")
    forecast:     float
    lower_ci:     float  = Field(description="90% lower confidence interval")
    upper_ci:     float  = Field(description="90% upper confidence interval")


class ForecastResponse(BaseModel):
    sku_id:   str
    horizon:  int
    points:   list[ForecastPoint]
    model:    str = "XGBoost"
    currency: str = "INR"


class AlertItem(BaseModel):
    sku_id:            str
    item_name:         str
    alert_type:        str  = Field(description="reorder | anomaly")
    severity:          str  = Field(description="high | medium | low")
    current_stock:     Optional[float] = None
    reorder_point:     Optional[float] = None
    gap:               Optional[float] = None
    anomaly_score:     Optional[float] = None
    message:           str


class AlertsResponse(BaseModel):
    total_alerts:   int
    reorder_alerts: int
    anomaly_alerts: int
    alerts:         list[AlertItem]


class HealthResponse(BaseModel):
    status:       str = "ok"
    demo_mode:    bool
    models_ready: bool
    chroma_docs:  int
