"""
db/models.py — SQLAlchemy ORM models.
"""
from __future__ import annotations
from datetime import datetime, date

from sqlalchemy import (
    String, Integer, Float, Date, DateTime, ForeignKey, Text, Boolean, Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Demand ──────────────────────────────────────────────────────────────────

class DemandRecord(Base):
    __tablename__ = "demand_records"

    id           : Mapped[int]      = mapped_column(primary_key=True, autoincrement=True)
    record_id    : Mapped[str]      = mapped_column(String(20), unique=True, index=True)
    period       : Mapped[date]     = mapped_column(Date, index=True)   # first-of-month
    month        : Mapped[str]      = mapped_column(String(10))
    year         : Mapped[int]      = mapped_column(Integer)
    sku_id       : Mapped[str]      = mapped_column(String(20), index=True)
    product_name : Mapped[str]      = mapped_column(String(120))
    abc_class    : Mapped[str]      = mapped_column(String(1))
    units_sold   : Mapped[int]      = mapped_column(Integer)
    units_returned: Mapped[int]     = mapped_column(Integer)
    net_units    : Mapped[int]      = mapped_column(Integer)
    channel      : Mapped[str]      = mapped_column(String(30))
    revenue_inr  : Mapped[float]    = mapped_column(Float)

    __table_args__ = (Index("ix_demand_sku_period", "sku_id", "period"),)


# ── Inventory (snapshot, not history — one row per SKU as-of snapshot_date) ──

class InventorySnapshot(Base):
    __tablename__ = "inventory_snapshots"

    id              : Mapped[int]   = mapped_column(primary_key=True, autoincrement=True)
    sku_id          : Mapped[str]   = mapped_column(String(20), index=True)
    item_name       : Mapped[str]   = mapped_column(String(120))
    item_type       : Mapped[str]   = mapped_column(String(30))
    abc_class       : Mapped[str]   = mapped_column(String(1))
    unit            : Mapped[str]   = mapped_column(String(20))
    qty_on_hand     : Mapped[float] = mapped_column(Float)
    qty_wip         : Mapped[float] = mapped_column(Float)
    qty_in_transit  : Mapped[float] = mapped_column(Float)
    total_available : Mapped[float] = mapped_column(Float)
    reorder_point   : Mapped[float] = mapped_column(Float)
    days_of_stock   : Mapped[float] = mapped_column(Float)
    status          : Mapped[str]   = mapped_column(String(20))
    snapshot_date   : Mapped[date]  = mapped_column(Date, index=True)
    location        : Mapped[str]   = mapped_column(String(60), nullable=True)


# ── Suppliers ─────────────────────────────────────────────────────────────────

class Supplier(Base):
    __tablename__ = "suppliers"

    supplier_id      : Mapped[str]   = mapped_column(String(20), primary_key=True)
    supplier_name    : Mapped[str]   = mapped_column(String(120))
    city             : Mapped[str]   = mapped_column(String(60))
    state            : Mapped[str]   = mapped_column(String(60))
    contact_person   : Mapped[str]   = mapped_column(String(80))
    phone            : Mapped[str]   = mapped_column(String(20))
    category         : Mapped[str]   = mapped_column(String(60))
    payment_terms    : Mapped[str]   = mapped_column(String(80))
    on_time_rate_pct : Mapped[float] = mapped_column(Float)
    bank             : Mapped[str]   = mapped_column(String(80), nullable=True)
    gstin            : Mapped[str]   = mapped_column(String(20), nullable=True)
    status           : Mapped[str]   = mapped_column(String(20))

    terms: Mapped["SupplierTerm"] = relationship(back_populates="supplier", uselist=False)


class SupplierTerm(Base):
    __tablename__ = "supplier_terms"

    id                : Mapped[int]   = mapped_column(primary_key=True, autoincrement=True)
    supplier_id       : Mapped[str]   = mapped_column(ForeignKey("suppliers.supplier_id"), index=True)
    supplier_name     : Mapped[str]   = mapped_column(String(120))
    skus_supplied     : Mapped[str]   = mapped_column(String(200))   # "; "-separated, matches CSV today
    lead_time_days    : Mapped[float] = mapped_column(Float)
    min_order_qty     : Mapped[float] = mapped_column(Float)
    credit_days       : Mapped[float] = mapped_column(Float)
    advance_pct       : Mapped[float] = mapped_column(Float, nullable=True)
    price_valid_until : Mapped[date]  = mapped_column(Date, nullable=True)
    on_time_rate_pct  : Mapped[float] = mapped_column(Float)
    penalty_clause    : Mapped[str]   = mapped_column(String(120), nullable=True)
    last_audit_date   : Mapped[date]  = mapped_column(Date, nullable=True)
    notes             : Mapped[str]   = mapped_column(Text, nullable=True)

    supplier: Mapped["Supplier"] = relationship(back_populates="terms")


# ── Purchase Orders ───────────────────────────────────────────────────────────

class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    id                : Mapped[int]   = mapped_column(primary_key=True, autoincrement=True)
    po_number         : Mapped[str]   = mapped_column(String(30), unique=True, index=True)
    po_date           : Mapped[date]  = mapped_column(Date, index=True)
    supplier_id       : Mapped[str]   = mapped_column(ForeignKey("suppliers.supplier_id"), index=True)
    supplier_name     : Mapped[str]   = mapped_column(String(120))
    sku_id            : Mapped[str]   = mapped_column(String(20), index=True)
    item_name         : Mapped[str]   = mapped_column(String(120))
    category          : Mapped[str]   = mapped_column(String(60))
    qty_ordered       : Mapped[float] = mapped_column(Float)
    unit              : Mapped[str]   = mapped_column(String(20))
    rate_per_unit_inr : Mapped[float] = mapped_column(Float)
    po_value_inr      : Mapped[float] = mapped_column(Float)
    expected_delivery : Mapped[date]  = mapped_column(Date, nullable=True)
    actual_delivery   : Mapped[date]  = mapped_column(Date, nullable=True)
    delay_days        : Mapped[float] = mapped_column(Float, nullable=True)
    status            : Mapped[str]   = mapped_column(String(30))
    remarks           : Mapped[str]   = mapped_column(Text, nullable=True)


# ── Users (JWT auth) ──────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id            : Mapped[int]      = mapped_column(primary_key=True, autoincrement=True)
    email         : Mapped[str]      = mapped_column(String(120), unique=True, index=True)
    hashed_password: Mapped[str]     = mapped_column(String(200))
    full_name     : Mapped[str]      = mapped_column(String(120), nullable=True)
    role          : Mapped[str]      = mapped_column(String(20), default="viewer")  # "viewer" | "admin"
    is_active     : Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at    : Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ── Query audit log (also doubles as thumbs up/down feedback store) ──────────

class QueryLog(Base):
    __tablename__ = "query_logs"

    id              : Mapped[int]      = mapped_column(primary_key=True, autoincrement=True)
    user_id         : Mapped[int]      = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    question        : Mapped[str]      = mapped_column(Text)
    intent          : Mapped[str]      = mapped_column(String(20))
    sku_id          : Mapped[str]      = mapped_column(String(20), nullable=True)
    answer          : Mapped[str]      = mapped_column(Text)
    rag_context_used: Mapped[bool]     = mapped_column(Boolean)
    feedback        : Mapped[str]      = mapped_column(String(10), nullable=True)  # "up" | "down" | None
    created_at      : Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


# ── Alerts (persisted so history/trend is queryable, not just live-computed) ─

class Alert(Base):
    __tablename__ = "alerts"

    id             : Mapped[int]      = mapped_column(primary_key=True, autoincrement=True)
    sku_id         : Mapped[str]      = mapped_column(String(20), index=True)
    item_name      : Mapped[str]      = mapped_column(String(120))
    alert_type     : Mapped[str]      = mapped_column(String(20))   # "reorder" | "anomaly"
    severity       : Mapped[str]      = mapped_column(String(10))   # "high" | "medium" | "low"
    current_stock  : Mapped[float]    = mapped_column(Float, nullable=True)
    reorder_point  : Mapped[float]    = mapped_column(Float, nullable=True)
    gap            : Mapped[float]    = mapped_column(Float, nullable=True)
    anomaly_score  : Mapped[float]    = mapped_column(Float, nullable=True)
    message        : Mapped[str]      = mapped_column(Text)
    created_at     : Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


# ── Eval harness results (intent accuracy / retrieval hit-rate / groundedness) ─

class EvalRun(Base):
    __tablename__ = "eval_runs"

    id               : Mapped[int]      = mapped_column(primary_key=True, autoincrement=True)
    run_at           : Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    git_commit       : Mapped[str]      = mapped_column(String(40), nullable=True)
    intent_accuracy  : Mapped[float]    = mapped_column(Float, nullable=True)
    retrieval_hit_rate: Mapped[float]   = mapped_column(Float, nullable=True)
    groundedness_pass_rate: Mapped[float] = mapped_column(Float, nullable=True)
    n_queries        : Mapped[int]      = mapped_column(Integer, nullable=True)
    notes            : Mapped[str]      = mapped_column(Text, nullable=True)